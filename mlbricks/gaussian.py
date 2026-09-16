"""Ready-made Bolt causal language model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .bolt import Bolt
from .components import Embedding, FFN, LayerNorm, LMHead, RMSNorm, Residual
from .ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from .residualbrick import ResController
from .position import make_position
from .runtime import (
    normalize_backend, backend_report as collect_backend_report,
    build_execution_plan, compile_module, prepare_module_execution, reset_execution_route, predict_module,
)
from .cache import GaussCache


@dataclass
class GaussianConfig:
    vocab_size: int
    context: int = 2048
    layers: int = 6
    dim: int = 384
    heads: int = 6
    latent_dim: int = 32
    position: str = "none"
    ffn: str = "standard"
    residual: str = "standard"
    norm: str = "rmsnorm"
    activation: str = "gelu"
    ffn_mult: int = 4
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    backend: str = "auto"
    eps: float = 1e-6
    ffn_state_dim: int = 256
    ffn_depth_embedding_dim: int = 64
    ffn_virtual_refinements: int = 2
    ffn_virtual_hidden_dim: int = 128
    ffn_micro_hidden_dim: int = 64
    ffn_micro_refinements: int = 1
    residual_update_ratio: float = 0.18
    residual_stream_ratio: float = 1.08
    residual_update_softness: float = 8.0
    residual_stream_softness: float = 8.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context", "layers", "dim", "heads", "latent_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        self.backend = normalize_backend(self.backend, warn_legacy=True)
        self.ffn = str(self.ffn).strip().lower().replace("-", "_")
        aliases = {
            "default": "standard", "ffn": "standard", "standard": "standard",
            "saffn": "saffn", "stateaware": "saffn", "state_aware": "saffn", "ffnbrick": "saffn",
            "virtual": "virtual", "virtual_state_aware": "virtual", "virtual_ffnbrick": "virtual",
            "micro": "micro", "micro_virtual": "micro", "micro_ffnbrick": "micro",
        }
        if self.ffn not in aliases:
            raise ValueError("ffn must be standard, saffn, virtual, or micro")
        self.ffn = aliases[self.ffn]
        self.residual = str(self.residual).strip().lower().replace("-", "_")
        if self.residual in {"default", "residual"}:
            self.residual = "standard"
        if self.residual in {"controller", "res_controller"}:
            self.residual = "rescontroller"
        if self.residual not in {"standard", "rescontroller"}:
            raise ValueError("residual must be standard or rescontroller")
        self.norm = str(self.norm).strip().lower().replace("-", "_")
        if self.norm not in {"rmsnorm", "rms", "layernorm", "layer_norm"}:
            raise ValueError("norm must be rmsnorm or layernorm")

    # ESA-style compatibility names.
    @property
    def block(self) -> int:
        return self.context

    @property
    def n_layer(self) -> int:
        return self.layers

    @property
    def embd(self) -> int:
        return self.dim

    @property
    def head(self) -> int:
        return self.heads


def _norm(kind: str, dim: int, bias: bool):
    if kind in {"rms", "rmsnorm"}:
        return RMSNorm(dim)
    return LayerNorm(dim, bias=bias)


def _residual(cfg: GaussianConfig):
    if cfg.residual == "standard":
        return Residual(cfg.dropout)
    return ResController(
        update_ratio=cfg.residual_update_ratio,
        stream_ratio=cfg.residual_stream_ratio,
        update_softness=cfg.residual_update_softness,
        stream_softness=cfg.residual_stream_softness,
        backend=cfg.backend,
    )


def _ffn(cfg: GaussianConfig, layer_index: int):
    if cfg.ffn == "standard":
        return FFN(
            cfg.dim,
            cfg.ffn_mult * cfg.dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
    if cfg.ffn == "saffn":
        return StateAwareFFN(
            d_model=cfg.dim,
            state_dim=cfg.ffn_state_dim,
            depth_embedding_dim=cfg.ffn_depth_embedding_dim,
            layer_index=layer_index,
            total_layers=cfg.layers,
            backend=cfg.backend,
        )
    if cfg.ffn == "virtual":
        return VirtualStateAwareFFN(
            d_model=cfg.dim,
            state_dim=cfg.ffn_state_dim,
            depth_embedding_dim=cfg.ffn_depth_embedding_dim,
            layer_index=layer_index,
            total_layers=cfg.layers,
            virtual_refinements=cfg.ffn_virtual_refinements,
            virtual_hidden_dim=cfg.ffn_virtual_hidden_dim,
            backend=cfg.backend,
        )
    return MicroVirtualFFN(
        d_model=cfg.dim,
        hidden_dim=cfg.ffn_micro_hidden_dim,
        refinements=cfg.ffn_micro_refinements,
        backend=cfg.backend,
    )


class _GaussianBlock(nn.Module):
    def __init__(self, cfg: GaussianConfig, layer_index: int, rotary=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm1 = _norm(cfg.norm, cfg.dim, cfg.bias)
        self.attn = Bolt(
            cfg.dim,
            cfg.heads,
            latent_dim=cfg.latent_dim,
            bias=cfg.bias,
            dropout=cfg.dropout,
            causal=True,
            backend=cfg.backend,
            eps=cfg.eps,
            position=rotary,
        )
        self.residual1 = _residual(cfg)
        self.norm2 = _norm(cfg.norm, cfg.dim, cfg.bias)
        self.ffn = _ffn(cfg, layer_index)
        self.residual2 = _residual(cfg)
        self._compiled_finish = None

    def _ffn_update(self, x, mixer_update, previous_mixer, previous_state):
        if isinstance(self.ffn, MicroVirtualFFN):
            refined = self.ffn.refine(x)
            return refined - x, previous_state
        if isinstance(self.ffn, (StateAwareFFN, VirtualStateAwareFFN)):
            if previous_mixer is None:
                previous_mixer = torch.zeros_like(mixer_update)
            if previous_state is None:
                previous_state = self.ffn.initial_state(x)
            return self.ffn(x, mixer_update, previous_mixer, previous_state)
        result = self.ffn(x)
        if isinstance(result, tuple):
            return result[0], previous_state
        return result, previous_state

    def finish(self, x, mixer_update, previous_mixer=None, previous_state=None):
        x = self.residual1(x, mixer_update)
        ffn_update, next_state = self._ffn_update(
            self.norm2(x), mixer_update, previous_mixer, previous_state
        )
        x = self.residual2(x, ffn_update)
        return x, mixer_update, next_state

    def compile_decode_region(self, *, mode: str = "default"):
        if not hasattr(torch, "compile"):
            return self
        try:
            self._compiled_finish = torch.compile(
                self.finish, mode=mode, fullgraph=False, dynamic=False
            )
        except Exception:
            self._compiled_finish = None
        return self

    def _finish_decode(self, x, mixer_update, previous_mixer=None, previous_state=None):
        if self._compiled_finish is None:
            return self.finish(x, mixer_update, previous_mixer, previous_state)
        try:
            return self._compiled_finish(x, mixer_update, previous_mixer, previous_state)
        except Exception:
            # Compilation is an optimization, never a correctness requirement.
            self._compiled_finish = None
            return self.finish(x, mixer_update, previous_mixer, previous_state)

    def forward(self, x, previous_mixer=None, previous_state=None):
        mixer_update = self.attn(self.norm1(x))
        return self.finish(x, mixer_update, previous_mixer, previous_state)

    @torch.no_grad()
    def prefill(
        self, x, previous_mixer=None, previous_state=None, *,
        start_pos: int = 0, cache_buffer: GaussCache | None = None,
    ):
        normalized = self.norm1(x)
        mixer_update, (c, rho) = self.attn.prefill_with_cache(
            normalized, start_pos=start_pos
        )
        cache = cache_buffer.load_prefix(c, rho) if cache_buffer is not None else (c, rho)
        x, mixer_update, next_state = self.finish(
            x, mixer_update, previous_mixer, previous_state
        )
        return x, cache, mixer_update, next_state

    @torch.no_grad()
    def step(self, x, cache, previous_mixer=None, previous_state=None, *, position: int):
        normalized = self.norm1(x)
        q, c_now, rho_now = self.attn.project_decode_state(
            normalized, start_pos=position
        )
        if isinstance(cache, GaussCache):
            mixer_update = self.attn.decode_append_projected(
                q, c_now, rho_now, cache.c, cache.rho, position=position
            )[:, None, :]
            cache.length = max(cache.length, int(position) + 1)
            next_cache = cache
        else:
            # Backward-compatible external prefill/decode API. Optimized model
            # generation uses GaussCache and never takes this torch.cat path.
            c_old, rho_old = cache
            c = torch.cat([c_old, c_now], dim=2)
            rho = torch.cat([rho_old, rho_now], dim=2)
            mixer_update = self.attn.decode_projected(
                q, c, rho, used_length=c.size(2)
            )[:, None, :]
            next_cache = (c, rho)
        x, mixer_update, next_state = self._finish_decode(
            x, mixer_update, previous_mixer, previous_state
        )
        return x, next_cache, mixer_update, next_state


class Gaussian(nn.Module):
    """Complete causal LM made from :class:`Bolt` blocks.

    Backend is ``auto`` implicitly.  Users normally omit it and only specify
    ``native`` or ``pytorch`` when they deliberately want to override dispatch.
    """

    def __init__(self, config: GaussianConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            # Accept ESA-style names as a convenience.
            aliases = {"block": "context", "n_layer": "layers", "embd": "dim", "head": "heads"}
            for old, new in aliases.items():
                if old in kwargs and new not in kwargs:
                    kwargs[new] = kwargs.pop(old)
            config = GaussianConfig(**kwargs)
        elif kwargs:
            raise TypeError("Pass either config=... or keyword configuration, not both")
        self.config = config
        self.backend = config.backend

        additive, rotary = make_position(
            config.position, dim=config.dim, max_seq_len=config.context
        )
        self.position = additive
        self.token_embedding = Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [_GaussianBlock(config, i, rotary=rotary) for i in range(config.layers)]
        )
        self.final_norm = _norm(config.norm, config.dim, config.bias)
        self.lm_head = LMHead(
            config.dim,
            config.vocab_size,
            bias=False,
            tie_to=self.token_embedding if config.tie_embeddings else None,
        )
        self._prepared_generation = None
        self._compile_generation_regions = False
        self._generation_compile_mode = "default"

    def _embed(self, input_ids: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.size(1) + start_pos > self.config.context:
            raise ValueError("sequence exceeds configured context")
        x = self.token_embedding(input_ids)
        if self.position is not None:
            try:
                x = self.position(x, start_pos=start_pos)
            except TypeError:
                x = self.position(x)
        return self.drop(x)

    def forward(self, input_ids: torch.Tensor, *, targets: torch.Tensor | None = None):
        x = self._embed(input_ids)
        previous_mixer = None
        ffn_state = None
        for block in self.blocks:
            x, previous_mixer, ffn_state = block(x, previous_mixer, ffn_state)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        self._mlbricks_requested_backend = value
        self._mlbricks_model_route = "operator" if value == "auto" else value
        self._mlbricks_model_route_reason = f"explicit:{value}"
        self._mlbricks_model_timings = None
        self.backend = value
        self.config.backend = value
        if recursive:
            for module in self.modules():
                if module is self:
                    continue
                setter = getattr(module, "set_backend", None)
                if callable(setter):
                    try:
                        setter(value, recursive=False)
                    except TypeError:
                        setter(value)
        return self

    def backend_report(self):
        return collect_backend_report(self)

    def execution_plan(self):
        return getattr(self, "_mlbricks_execution_plan", build_execution_plan(self))

    def prepare_execution(self, *sample_args, sample_kwargs=None, warmup=5, trials=20, candidates=("operator", "native", "pytorch"), force=False):
        return prepare_module_execution(
            self, *sample_args, sample_kwargs=sample_kwargs, warmup=warmup,
            trials=trials, candidates=tuple(candidates), force=force,
        )

    def predict(self, *args, device="auto", dtype="auto", calibrate=True, **kwargs):
        """One-call optimized inference with automatic placement and planning."""
        return predict_module(
            self, *args, device=device, dtype=dtype, calibrate=calibrate, **kwargs
        )

    def reset_execution(self):
        reset_execution_route(self)
        return self

    def compile(
        self, *, mode: str = "default", dynamic: bool | None = None,
        fullgraph: bool = False, strict: bool = False,
    ):
        """Compile graphable PyTorch regions and keep native ops as boundaries."""
        compile_module(
            self, mode=mode, dynamic=dynamic, fullgraph=fullgraph, strict=strict
        )
        self._compile_generation_regions = True
        self._generation_compile_mode = str(mode)
        return self

    def prepare_generation(
        self, *, batch_size: int = 1, max_context: int | None = None,
        device=None, dtype=None, warmup_native: bool = True,
    ):
        """Preallocate fixed-capacity Gauss caches for low-overhead decode."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        capacity = self.config.context if max_context is None else int(max_context)
        if capacity <= 0 or capacity > self.config.context:
            raise ValueError("max_context must be in [1, config.context]")
        p = next(self.parameters())
        device = p.device if device is None else torch.device(device)
        dtype = p.dtype if dtype is None else dtype
        caches = [
            GaussCache.allocate(
                batch_size, block.attn.num_heads, capacity, block.attn.latent_dim,
                device=device, dtype=dtype,
            )
            for block in self.blocks
        ]
        self._prepared_generation = {
            "batch_size": int(batch_size),
            "capacity": capacity,
            "device": str(device),
            "dtype": dtype,
            "caches": caches,
        }
        if warmup_native and device.type == "cuda" and self.backend != "pytorch":
            # Pay extension import/load cost before the autoregressive loop.
            try:
                from ._backend import load_cuda_extension
                load_cuda_extension()
            except Exception:
                if self.backend == "native":
                    raise
        if device.type == "cuda" and self._compile_generation_regions:
            for block in self.blocks:
                block.compile_decode_region(mode=self._generation_compile_mode)
        return self

    def _generation_caches(self, input_ids: torch.Tensor, capacity: int):
        p = next(self.parameters())
        prepared = self._prepared_generation
        if (prepared is None
            or prepared["batch_size"] != input_ids.size(0)
            or prepared["capacity"] < capacity
            or prepared["device"] != str(input_ids.device)
            or prepared["dtype"] != p.dtype):
            self.prepare_generation(
                batch_size=input_ids.size(0), max_context=capacity,
                device=input_ids.device, dtype=p.dtype, warmup_native=True,
            )
            prepared = self._prepared_generation
        return prepared["caches"]

    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor, *, cache_buffers: list[GaussCache] | None = None
    ):
        x = self._embed(input_ids, start_pos=0)
        caches = []
        previous_mixer = None
        ffn_state = None
        for i, block in enumerate(self.blocks):
            cache_buffer = None if cache_buffers is None else cache_buffers[i]
            x, cache, previous_mixer, ffn_state = block.prefill(
                x, previous_mixer, ffn_state, start_pos=0, cache_buffer=cache_buffer
            )
            caches.append(cache)
        logits = self.lm_head(self.final_norm(x))
        return logits, caches

    @torch.no_grad()
    def decode_step(self, input_ids: torch.Tensor, caches, *, position: int):
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.shape[1] != 1:
            raise ValueError("decode_step expects one token per batch")
        x = self._embed(input_ids, start_pos=position)
        new_caches = []
        previous_mixer = None
        ffn_state = None
        for block, cache in zip(self.blocks, caches):
            x, next_cache, previous_mixer, ffn_state = block.step(
                x, cache, previous_mixer, ffn_state, position=position
            )
            new_caches.append(next_cache)
        logits = self.lm_head(self.final_norm(x))
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids
        total = input_ids.size(1) + max_new_tokens
        if total > self.config.context:
            raise ValueError("generation would exceed configured context")
        was_training = self.training
        self.eval()

        # Fixed-capacity cache + token storage: no growing torch.cat in the hot
        # autoregressive loop. Cache tensors keep stable addresses as well.
        cache_buffers = self._generation_caches(input_ids, total)
        logits, caches = self.prefill(input_ids, cache_buffers=cache_buffers)
        tokens = torch.empty(
            input_ids.size(0), total, device=input_ids.device, dtype=input_ids.dtype
        )
        tokens[:, : input_ids.size(1)].copy_(input_ids)
        next_logits = logits[:, -1, :]
        for i in range(max_new_tokens):
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scores = next_logits / float(temperature)
                if top_k is not None and 0 < top_k < scores.size(-1):
                    cutoff = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                next_token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            write_pos = input_ids.size(1) + i
            tokens[:, write_pos:write_pos + 1].copy_(next_token)
            if i + 1 == max_new_tokens:
                break
            step_logits, caches = self.decode_step(
                next_token, caches, position=write_pos
            )
            next_logits = step_logits[:, -1, :]
        if was_training:
            self.train()
        return tokens

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"vocab_size={c.vocab_size}, context={c.context}, layers={c.layers}, "
            f"dim={c.dim}, heads={c.heads}, latent_dim={c.latent_dim}, "
            f"position={c.position!r}, backend={self.backend!r}"
        )
