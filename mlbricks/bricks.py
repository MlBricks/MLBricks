"""Composable layer-by-layer neural-network builder for MLBricks."""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
import torch.nn.functional as F

from .components import Embedding, FFN, LMHead, LayerNorm, RMSNorm, Residual
from .bolt import Attention, Bolt
from .esa import ESA
from .ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from .position import RoPE, make_position
from .runtime import (
    normalize_backend, backend_report as collect_backend_report,
    build_execution_plan, compile_module, prepare_module_execution, reset_execution_route, predict_module,
)
from .cache import GaussCache, KVCache


def _make_norm(spec, dim: int):
    if isinstance(spec, nn.Module):
        return spec
    name = str(spec or "rmsnorm").strip().lower().replace("-", "_")
    if name in {"rms", "rmsnorm"}:
        return RMSNorm(dim)
    if name in {"layernorm", "layer_norm", "ln"}:
        return LayerNorm(dim)
    raise ValueError("norm must be rmsnorm, layernorm, or an nn.Module")


def _call_optional(module_or_fn, x):
    if module_or_fn is None:
        return x
    return module_or_fn(x)


class Brick(nn.Module):
    """One configurable MLBricks model layer.

    A Brick is intentionally composition-first: ``mixer``, ``ffn``, ``norm``
    and ``residual`` can all be user-created modules.  This lets researchers
    mix ESA, Bolt, standard Attention and custom operations in one
    model without adding a new monolithic architecture class.
    """

    def __init__(
        self,
        *,
        mixer: nn.Module,
        ffn: nn.Module | None = None,
        norm1: nn.Module | str = "rmsnorm",
        norm2: nn.Module | str = "rmsnorm",
        residual: nn.Module | None = None,
        residual2: nn.Module | None = None,
        position=None,
        pre: nn.Module | Callable | None = None,
        post: nn.Module | Callable | None = None,
        dim: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(mixer, nn.Module):
            raise TypeError("mixer must be an nn.Module")
        inferred = dim or getattr(mixer, "d_model", None) or getattr(mixer, "embd", None)
        if inferred is None:
            raise ValueError("dim is required when mixer does not expose d_model/embd")
        self.dim = int(inferred)
        self.mixer = mixer
        self.ffn = ffn
        self.norm1 = _make_norm(norm1, self.dim)
        self.norm2 = _make_norm(norm2, self.dim)
        self.residual1 = residual if residual is not None else Residual()
        self.residual2 = residual2 if residual2 is not None else (
            residual if residual is not None else Residual()
        )
        self.pre = pre
        self.post = post
        self.position = position
        self._compiled_finish = None
        if position is not None:
            _, rotary = make_position(position, dim=self.dim, max_seq_len=65536)
            if rotary is not None:
                setter = getattr(self.mixer, "set_position", None)
                if not callable(setter):
                    raise ValueError("RoPE requires a mixer with set_position(), such as Attention/Bolt")
                setter(rotary)

    def _mix(self, x):
        out = self.mixer(x)
        return out[0] if isinstance(out, tuple) else out

    def _ffn_update(self, x, mixer_update, previous_mixer, previous_state):
        if self.ffn is None:
            return torch.zeros_like(x), previous_state
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
        if self.ffn is not None:
            x = self.residual2(x, ffn_update)
        x = _call_optional(self.post, x)
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
            self._compiled_finish = None
            return self.finish(x, mixer_update, previous_mixer, previous_state)

    def forward(self, x, previous_mixer=None, previous_state=None):
        x = _call_optional(self.pre, x)
        update = self._mix(self.norm1(x))
        return self.finish(x, update, previous_mixer, previous_state)

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        targets = [self.mixer, self.ffn, self.residual1, self.residual2]
        for module in targets:
            if module is None:
                continue
            setter = getattr(module, "set_backend", None)
            if callable(setter):
                try:
                    setter(value, recursive=recursive)
                except TypeError:
                    setter(value)
        return self

    @torch.no_grad()
    def prefill(
        self, x, previous_mixer=None, previous_state=None, *, start_pos: int = 0,
        cache_buffer=None,
    ):
        x = _call_optional(self.pre, x)
        normalized = self.norm1(x)
        state = None
        if isinstance(self.mixer, Bolt):
            update, (c, rho) = self.mixer.prefill_with_cache(
                normalized, start_pos=start_pos
            )
            if isinstance(cache_buffer, GaussCache):
                cache_buffer.load_prefix(c, rho)
                state = ("gauss", cache_buffer)
            else:
                state = ("gauss", c, rho)
        elif isinstance(self.mixer, Attention):
            update, (k, v) = self.mixer.prefill_with_cache(
                normalized, start_pos=start_pos
            )
            if isinstance(cache_buffer, KVCache):
                cache_buffer.load_prefix(k, v)
                state = ("attention", cache_buffer)
            else:
                state = ("attention", k, v)
        elif isinstance(self.mixer, ESA):
            update, esa_state = self.mixer.prefill(normalized)
            state = ("esa", esa_state)
        else:
            update = self._mix(normalized)
            state = ("stateless", None)
        x, update, next_ffn = self.finish(x, update, previous_mixer, previous_state)
        return x, state, update, next_ffn

    @torch.no_grad()
    def step(self, x, state, previous_mixer=None, previous_state=None, *, position: int):
        x = _call_optional(self.pre, x)
        normalized = self.norm1(x)
        kind = state[0]
        if kind == "gauss":
            q, c_now, rho_now = self.mixer.project_decode_state(
                normalized, start_pos=position
            )
            if len(state) == 2 and isinstance(state[1], GaussCache):
                cache = state[1]
                update = self.mixer.decode_append_projected(
                    q, c_now, rho_now, cache.c, cache.rho, position=position
                )[:, None, :]
                cache.length = max(cache.length, int(position) + 1)
                next_state = ("gauss", cache)
            else:
                _, c_old, rho_old = state
                c = torch.cat([c_old, c_now], dim=2)
                rho = torch.cat([rho_old, rho_now], dim=2)
                update = self.mixer.decode_projected(
                    q, c, rho, used_length=c.size(2)
                )[:, None, :]
                next_state = ("gauss", c, rho)
        elif kind == "attention":
            q, k_now, v_now = self.mixer.project_decode_state(
                normalized, start_pos=position
            )
            if len(state) == 2 and isinstance(state[1], KVCache):
                cache = state[1]
                update = self.mixer.decode_append_projected(
                    q, k_now, v_now, cache.k, cache.v, position=position
                )[:, None, :]
                cache.length = max(cache.length, int(position) + 1)
                next_state = ("attention", cache)
            else:
                _, k_old, v_old = state
                k = torch.cat([k_old, k_now], dim=2)
                v = torch.cat([v_old, v_now], dim=2)
                update = self.mixer.decode_projected(
                    q, k, v, used_length=k.size(2)
                )[:, None, :]
                next_state = ("attention", k, v)
        elif kind == "esa":
            update, esa_state = self.mixer.decode_step(normalized[:, 0, :], state[1])
            if update.ndim == 2:
                update = update[:, None, :]
            next_state = ("esa", esa_state)
        else:
            update = self._mix(normalized)
            next_state = state
        x, update, next_ffn = self._finish_decode(
            x, update, previous_mixer, previous_state
        )
        return x, next_state, update, next_ffn


class Bricks(nn.Module):
    """Custom layer-by-layer MLBricks model pipeline."""

    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        context: int,
        layers: list[Brick] | tuple[Brick, ...],
        embedding: nn.Module | str = "standard",
        position=None,
        final_norm: nn.Module | str = "rmsnorm",
        lm_head: nn.Module | str = "tied",
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or dim <= 0 or context <= 0:
            raise ValueError("vocab_size, dim, and context must be positive")
        if not layers:
            raise ValueError("Bricks requires at least one Brick layer")
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.context = int(context)
        self.backend = normalize_backend(backend, warn_legacy=True)
        self.layers = nn.ModuleList(layers)
        for layer in self.layers:
            if layer.dim != self.dim:
                raise ValueError(f"every Brick must use dim={self.dim}")

        if isinstance(embedding, str):
            if embedding.strip().lower() not in {"standard", "embedding"}:
                raise ValueError("embedding string must be 'standard' or pass an nn.Module")
            self.embedding = Embedding(self.vocab_size, self.dim)
        elif isinstance(embedding, nn.Module):
            self.embedding = embedding
        else:
            raise TypeError("embedding must be 'standard' or nn.Module")

        additive, rotary = make_position(position, dim=self.dim, max_seq_len=self.context)
        self.position = additive
        if rotary is not None:
            for layer in self.layers:
                setter = getattr(layer.mixer, "set_position", None)
                if callable(setter):
                    setter(rotary)

        self.drop = nn.Dropout(dropout)
        self.final_norm = _make_norm(final_norm, self.dim)
        if isinstance(lm_head, str):
            name = lm_head.strip().lower()
            if name not in {"tied", "standard"}:
                raise ValueError("lm_head must be tied, standard, or an nn.Module")
            self.lm_head = LMHead(
                self.dim,
                self.vocab_size,
                bias=False,
                tie_to=self.embedding if name == "tied" else None,
            )
        elif isinstance(lm_head, nn.Module):
            self.lm_head = lm_head
        else:
            raise TypeError("lm_head must be a string or nn.Module")
        if self.backend != "auto":
            self.set_backend(self.backend)
        self._prepared_generation = None
        self._compile_generation_regions = False
        self._generation_compile_mode = "default"

    def _embed(self, input_ids, *, start_pos: int = 0):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if start_pos + input_ids.size(1) > self.context:
            raise ValueError("sequence exceeds configured context")
        x = self.embedding(input_ids)
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
        for layer in self.layers:
            x, previous_mixer, ffn_state = layer(x, previous_mixer, ffn_state)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        self._mlbricks_requested_backend = value
        self._mlbricks_model_route = "operator" if value == "auto" else value
        self._mlbricks_model_route_reason = f"explicit:{value}"
        self._mlbricks_model_timings = None
        self.backend = value
        if recursive:
            for layer in self.layers:
                layer.set_backend(value, recursive=True)
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
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        capacity = self.context if max_context is None else int(max_context)
        if capacity <= 0 or capacity > self.context:
            raise ValueError("max_context must be in [1, context]")
        p = next(self.parameters())
        device = p.device if device is None else torch.device(device)
        dtype = p.dtype if dtype is None else dtype
        buffers = []
        for layer in self.layers:
            mixer = layer.mixer
            if isinstance(mixer, Bolt):
                buffers.append(GaussCache.allocate(
                    batch_size, mixer.num_heads, capacity, mixer.latent_dim,
                    device=device, dtype=dtype,
                ))
            elif isinstance(mixer, Attention):
                buffers.append(KVCache.allocate(
                    batch_size, mixer.num_heads, capacity, mixer.head_dim,
                    device=device, dtype=dtype,
                ))
            else:
                buffers.append(None)
        self._prepared_generation = {
            "batch_size": int(batch_size), "capacity": capacity,
            "device": str(device), "dtype": dtype, "buffers": buffers,
        }
        if warmup_native and device.type == "cuda" and self.backend != "pytorch":
            try:
                from ._backend import load_cuda_extension
                load_cuda_extension()
            except Exception:
                if self.backend == "native":
                    raise
        if device.type == "cuda" and self._compile_generation_regions:
            for layer in self.layers:
                layer.compile_decode_region(mode=self._generation_compile_mode)
        return self

    def _generation_buffers(self, input_ids: torch.Tensor, capacity: int):
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
        return prepared["buffers"]

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, *, cache_buffers=None):
        x = self._embed(input_ids, start_pos=0)
        states = []
        previous_mixer = None
        ffn_state = None
        for i, layer in enumerate(self.layers):
            cache_buffer = None if cache_buffers is None else cache_buffers[i]
            x, state, previous_mixer, ffn_state = layer.prefill(
                x, previous_mixer, ffn_state, start_pos=0, cache_buffer=cache_buffer
            )
            states.append(state)
        return self.lm_head(self.final_norm(x)), states

    @torch.no_grad()
    def decode_step(self, input_ids: torch.Tensor, states, *, position: int):
        if input_ids.ndim == 1:
            input_ids = input_ids[:, None]
        if input_ids.size(1) != 1:
            raise ValueError("decode_step expects one token")
        x = self._embed(input_ids, start_pos=position)
        next_states = []
        previous_mixer = None
        ffn_state = None
        for layer, state in zip(self.layers, states):
            x, state, previous_mixer, ffn_state = layer.step(
                x, state, previous_mixer, ffn_state, position=position
            )
            next_states.append(state)
        return self.lm_head(self.final_norm(x)), next_states

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int, *, temperature: float = 1.0, top_k: int | None = None):
        total = input_ids.size(1) + max_new_tokens
        if total > self.context:
            raise ValueError("generation would exceed configured context")
        if max_new_tokens <= 0:
            return input_ids
        was_training = self.training
        self.eval()
        buffers = self._generation_buffers(input_ids, total)
        logits, states = self.prefill(input_ids, cache_buffers=buffers)
        tokens = torch.empty(
            input_ids.size(0), total, device=input_ids.device, dtype=input_ids.dtype
        )
        tokens[:, : input_ids.size(1)].copy_(input_ids)
        next_logits = logits[:, -1, :]
        for i in range(max_new_tokens):
            if temperature <= 0:
                token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scores = next_logits / float(temperature)
                if top_k is not None and 0 < top_k < scores.size(-1):
                    cutoff = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                token = torch.multinomial(torch.softmax(scores, dim=-1), 1)
            pos = input_ids.size(1) + i
            tokens[:, pos:pos + 1].copy_(token)
            if i + 1 == max_new_tokens:
                break
            logits, states = self.decode_step(token, states, position=pos)
            next_logits = logits[:, -1, :]
        if was_training:
            self.train()
        return tokens

