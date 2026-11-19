from __future__ import annotations

import math
import contextlib
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .._reference import attention_forward_reference
from ..runtime import normalize_backend
from ..planner import EXECUTION_PLANNER
from ..position import RoPE
from .precision import native_bf16_supported
from .graph_policy import normalize_graph_policy

from .._backend import (
    WORKSPACES,
    KernelConfig,
    autotune,
    heuristic_config,
    load_cuda_extension,
)


def _validate_backend(backend: str) -> str:
    return normalize_backend(backend, warn_legacy=True)


def _planner_native_decode(
    owner: object,
    op: str,
    tensor: torch.Tensor,
    backend: str,
    eligible: bool,
    *,
    extra: tuple[object, ...] = (),
) -> bool:
    """Resolve an inference decode/preprocess route through the shared planner."""
    policy = _validate_backend(backend)
    if policy == "pytorch":
        return False
    if policy == "native":
        return bool(eligible)
    if not eligible:
        return False
    return EXECUTION_PLANNER.select_operator_once(
        owner, op, tensor, requested_backend="auto", native_available=True,
        native_supports_training=False, training=False, extra=extra,
    ) == "native"


def _resolve_rope(position, width: int):
    if position is None:
        return None
    if isinstance(position, RoPE):
        return position
    if isinstance(position, str) and position.strip().lower() in {"rope", "rotary"}:
        return RoPE(width)
    if isinstance(position, str) and position.strip().lower() in {"none", "off"}:
        return None
    raise ValueError("Attention position must be None, 'rope', or RoPE(...)")


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Mathematically standard scaled dot-product attention."""
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=(float(dropout_p) if training else 0.0),
        is_causal=bool(causal),
        scale=float(scale),
    )


class Attention(nn.Module):
    """Standard multi-head causal attention with native MLBricks decode."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = True,
        backend: str = "auto",
        autotune_kernels: bool = True,
        position=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.backend = _validate_backend(backend)
        self.autotune_kernels = bool(autotune_kernels)
        self.position = _resolve_rope(position, self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = _validate_backend(backend)
        EXECUTION_PLANNER.clear_owner_routes(self)
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        return "planner(auto forward) + native-decode"

    def set_position(self, position):
        self.position = _resolve_rope(position, self.head_dim)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [B,T,D]")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got {D}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if self.position is not None:
            q = self.position(q, start_pos=0)
            k = self.position(k, start_pos=0)

        # Quality-safe policy:
        # - gradient-enabled training uses the exact original 0.1.0 reference
        #   equation/order, preserving its forward and backward numerics.
        # - eval/no-grad prefill uses SDPA for the measured inference speedup.
        if self.training and torch.is_grad_enabled():
            y = attention_forward_reference(
                q, k, v,
                scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=self.causal,
                dropout_p=self.dropout,
                training=True,
            )
        else:
            y = _sdpa(
                q,
                k,
                v,
                scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=self.causal,
                dropout_p=self.dropout,
                training=self.training,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)

    @torch.no_grad()
    def project_cache_state(self, x: torch.Tensor, *, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x[:, None, :]
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        B, T, _ = x.shape
        qkv = self.qkv(x)
        _, k, v = qkv.chunk(3, dim=-1)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        if self.position is not None:
            k = self.position(k, start_pos=int(start_pos)).contiguous()
        return k, v

    @torch.no_grad()
    def prefill_with_cache(self, x: torch.Tensor, *, start_pos: int = 0):
        """Eval/prefill output and K/V cache from a single QKV projection."""
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        B, T, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if self.position is not None:
            q = self.position(q, start_pos=int(start_pos))
            k = self.position(k, start_pos=int(start_pos))
        y = _sdpa(
            q, k, v,
            scale=1.0 / math.sqrt(float(self.head_dim)),
            causal=self.causal, dropout_p=0.0, training=False,
        )
        y = self.out_proj(y.transpose(1, 2).contiguous().view(B, T, self.d_model))
        return y, (k.contiguous(), v.contiguous())

    @torch.no_grad()
    def project_decode_state(self, x: torch.Tensor, *, start_pos: int):
        """Project one decode token once, returning Q and appendable K/V."""
        if x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError("project_decode_state expects one token")
            x = x[:, 0, :]
        B = x.size(0)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, 1, self.head_dim)
        v = v.view(B, self.num_heads, 1, self.head_dim)
        if self.position is not None:
            q = self.position(q[:, :, None, :], start_pos=int(start_pos))[:, :, 0, :]
            k = self.position(k, start_pos=int(start_pos))
        return q.contiguous(), k.contiguous(), v.contiguous()

    @torch.no_grad()
    def decode_projected(
        self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, *,
        used_length: int | None = None, force_retune: bool = False,
    ) -> torch.Tensor:
        """Decode an already-projected Q against a fixed-capacity K/V cache."""
        B = q.size(0)
        capacity = int(k_cache.size(2))
        used = capacity if used_length is None else int(used_length)
        if used < 1 or used > capacity:
            raise ValueError("used_length must be in [1, cache_capacity]")
        native_eligible = (
            q.is_cuda and q.dtype == torch.float16
            and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
            and self.head_dim <= 64 and k_cache.is_contiguous() and v_cache.is_contiguous()
        )
        ext = load_cuda_extension() if _planner_native_decode(
            self, "attention_decode", q, self.backend, native_eligible,
            extra=(self.num_heads, used, self.head_dim, "projected"),
        ) else None
        if ext is not None and used != capacity and not hasattr(ext, "baseline_decode_out_used"):
            ext = None
        if self.backend == "native" and (not native_eligible or ext is None):
            raise RuntimeError("backend='native' requested but native projected decode is unavailable")
        if ext is None:
            y = _sdpa(
                q[:, :, None, :], k_cache[:, :, :used, :], v_cache[:, :, :used, :],
                scale=1.0 / math.sqrt(float(self.head_dim)), causal=False,
                dropout_p=0.0, training=False,
            )[:, :, 0, :]
        else:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="baseline", q=q, a=k_cache, b=v_cache, head_dim=self.head_dim,
                    ext=ext, force=force_retune,
                    used_length=used if used != capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="baseline", B=B, H=self.num_heads, T=used, W=self.head_dim
                )
            ws = WORKSPACES.get("baseline", B, self.num_heads, self.head_dim, cfg.splits, q.device)
            if used != capacity:
                ext.baseline_decode_out_used(
                    q, k_cache, v_cache, ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits, used,
                )
            else:
                ext.baseline_decode_out(
                    q, k_cache, v_cache, ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits,
                )
            y = ws.out
        return self.out_proj(y.reshape(B, self.d_model))

    @torch.no_grad()
    def decode_append_projected(
        self,
        q: torch.Tensor,
        k_now: torch.Tensor,
        v_now: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        position: int,
        force_retune: bool = False,
    ) -> torch.Tensor:
        """Append one projected K/V token and decode without Python cache copies.

        On supported FP16 CUDA inputs, the cache write and decode are issued by
        one native extension call on the current CUDA stream.  Other devices
        keep the exact PyTorch fallback semantics.
        """
        B = q.size(0)
        pos = int(position)
        capacity = int(k_cache.size(2))
        if pos < 0 or pos >= capacity:
            raise ValueError("position must be in [0, cache_capacity)")
        if k_now.dim() == 4:
            if k_now.size(2) != 1 or v_now.size(2) != 1:
                raise ValueError("k_now/v_now must contain exactly one token")
            k_now3 = k_now[:, :, 0, :]
            v_now3 = v_now[:, :, 0, :]
        else:
            k_now3, v_now3 = k_now, v_now
        if k_now3.shape != q.shape or v_now3.shape != q.shape:
            raise ValueError("projected q/k_now/v_now shapes must match [B,H,Dh]")

        native_eligible = (
            q.is_cuda and q.dtype == torch.float16
            and k_now3.dtype == torch.float16 and v_now3.dtype == torch.float16
            and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
            and self.head_dim <= 64
            and q.is_contiguous() and k_now3.is_contiguous() and v_now3.is_contiguous()
            and k_cache.is_contiguous() and v_cache.is_contiguous()
        )
        ext = load_cuda_extension() if _planner_native_decode(
            self, "attention_decode", q, self.backend, native_eligible,
            extra=(self.num_heads, pos + 1, self.head_dim, "append"),
        ) else None
        has_fused = bool(ext is not None and hasattr(ext, "baseline_decode_append_out"))

        if self.backend == "native" and (not native_eligible or ext is None):
            raise RuntimeError(
                "backend='native' requested but native append+decode is unavailable"
            )

        used = pos + 1
        if has_fused:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="baseline", q=q, a=k_cache, b=v_cache,
                    head_dim=self.head_dim, ext=ext, force=force_retune,
                    used_length=used if used != capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="baseline", B=B, H=self.num_heads, T=used, W=self.head_dim
                )
            ws = WORKSPACES.get(
                "baseline", B, self.num_heads, self.head_dim, cfg.splits, q.device
            )
            ext.baseline_decode_append_out(
                q, k_now3, v_now3, k_cache, v_cache,
                ws.pm, ws.pl, ws.po, ws.out,
                1.0 / math.sqrt(float(self.head_dim)),
                cfg.mode, cfg.splits, pos,
            )
            y = ws.out
            return self.out_proj(y.reshape(B, self.d_model))

        # Older extension / CPU / explicit PyTorch: preserve correctness.  If a
        # new extension is present but fused decode is unavailable, still use
        # its single cache-write kernel to avoid two separate copy_ launches.
        if ext is not None and hasattr(ext, "baseline_append_cache"):
            ext.baseline_append_cache(k_now3, v_now3, k_cache, v_cache, pos)
        else:
            k_cache[:, :, pos:pos + 1, :].copy_(k_now3[:, :, None, :])
            v_cache[:, :, pos:pos + 1, :].copy_(v_now3[:, :, None, :])
        return self.decode_projected(
            q, k_cache, v_cache, used_length=used, force_retune=force_retune
        )

    @torch.no_grad()
    def decode(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        force_retune: bool = False,
        position: int | None = None,
        used_length: int | None = None,
    ) -> torch.Tensor:
        if x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError("decode expects one token")
            x = x[:, 0, :]
        if x.dim() != 2 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape [B,{self.d_model}]")
        B = x.size(0)

        if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("k_cache/v_cache must have matching [B,H,T,Dh] shape")
        if (
            k_cache.size(0) != B
            or k_cache.size(1) != self.num_heads
            or k_cache.size(3) != self.head_dim
        ):
            raise ValueError("K/V cache shape does not match this Attention module")
        if k_cache.size(2) < 1:
            raise ValueError("decode cache must contain at least one token")
        if k_cache.device != x.device or v_cache.device != x.device:
            raise ValueError("x and K/V caches must be on the same device")
        cache_capacity = int(k_cache.size(2))
        used = cache_capacity if used_length is None else int(used_length)
        if used < 1 or used > cache_capacity:
            raise ValueError("used_length must be in [1, cache_capacity]")

        qkv = self.qkv(x)
        q = qkv[:, : self.d_model].view(B, self.num_heads, self.head_dim).contiguous()
        if self.position is not None:
            q_pos = used if position is None else int(position)
            q = self.position(q[:, :, None, :], start_pos=q_pos)[:, :, 0, :].contiguous()

        native_eligible = (
            q.is_cuda
            and q.dtype == torch.float16
            and k_cache.dtype == torch.float16
            and v_cache.dtype == torch.float16
            and self.head_dim <= 64
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
        )

        ext = None
        if _planner_native_decode(
            self, "attention_decode", q, self.backend, native_eligible,
            extra=(self.num_heads, used, self.head_dim, "decode"),
        ):
            ext = load_cuda_extension()

        native_used_api = bool(ext is not None and hasattr(ext, "baseline_decode_out_used"))
        if ext is not None and used != cache_capacity and not native_used_api:
            # Older prebuilt extension cannot distinguish logical length from
            # fixed cache capacity. Auto safely falls back without copying.
            ext = None
        if self.backend == "native" and (not native_eligible or ext is None):
            raise RuntimeError(
                "backend='native' requested but the native FP16 CUDA decode path is unavailable"
            )

        if ext is None:
            k_used = k_cache[:, :, :used, :]
            v_used = v_cache[:, :, :used, :]
            y = _sdpa(
                q[:, :, None, :],
                k_used,
                v_used,
                scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=False,
                dropout_p=0.0,
                training=False,
            )[:, :, 0, :]
        else:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="baseline",
                    q=q,
                    a=k_cache,
                    b=v_cache,
                    head_dim=self.head_dim,
                    ext=ext,
                    force=force_retune,
                    used_length=used if used != cache_capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="baseline",
                    B=B,
                    H=self.num_heads,
                    T=used,
                    W=self.head_dim,
                )
            ws = WORKSPACES.get(
                "baseline",
                B,
                self.num_heads,
                self.head_dim,
                cfg.splits,
                q.device,
            )
            if used != cache_capacity:
                ext.baseline_decode_out_used(
                    q, k_cache, v_cache,
                    ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)),
                    cfg.mode, cfg.splits, used,
                )
            else:
                ext.baseline_decode_out(
                    q,
                    k_cache,
                    v_cache,
                    ws.pm,
                    ws.pl,
                    ws.po,
                    ws.out,
                    1.0 / math.sqrt(float(self.head_dim)),
                    cfg.mode,
                    cfg.splits,
                )
            y = ws.out

        return self.out_proj(y.reshape(B, self.d_model))


class Bolt(nn.Module):
    """
    Unified BOLT attention: quantised MQA/GQA, Keyless shared C, or full QKV.

    The new modern modes all apply optional PRE-attention AVQ, using fixed
    multi-timescale decay or learned token-wise retention. No positional
    encodings are built in. A token-local Q/K transform may be provided by
    callers. Historical BOLT modes are no longer selectable.
    """

    def __init__(
        self,
        d_model: int | None = None,
        num_heads: int | None = None,
        *,
        embed_dim: int | None = None,
        latent_dim: int | None = None,
        # Attention choices: quantised (default), Keyless, base.
        mode: str = "quantised",
        cache: str = "normal",
        avq_dim: int | None = None,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = True,
        backend: str = "auto",
        autotune_kernels: bool = True,
        eps: float = 1e-6,
        use_sdpa: bool = True,
        position=None,
        native_full_sequence: bool = False,
        graph: str | bool = "auto",
        profile: str | None = None,
        state_memory: bool = True,
        avq: bool | str | None = True,
        retention: str = "decay",
        kv_heads: int | None = None,
        forget_gate_init: float = 0.95,
        head_output_gate: bool = True,
        device: str = 'auto',
        precision: str = 'fp32',
        attention_scale: str | None = None,
        memory_init: str | None = None,
        projection: str | None = None,
        avq_return_map: str = 'auto',
        avq_norm_eps: float | None = None,
    ):
        super().__init__()
        if not isinstance(state_memory, bool):
            raise TypeError('state_memory must be True or False')
        # An explicit AVQ switch changes recurrent state, not the attention architecture.
        if avq is not None:
            if isinstance(avq, bool):
                state_memory = bool(avq) and state_memory
            elif isinstance(avq, str) and avq.strip().lower() in ('enable', 'disable'):
                state_memory = avq.strip().lower() == 'enable' and state_memory
            else:
                raise ValueError('avq must be bool, enable, or disable')
        if retention not in ('decay', 'learned'):
            raise ValueError('retention must be decay or learned')
        self.retention = retention
        self.avq_mode = 'enable' if state_memory else 'disable'
        if profile is not None:
            if profile not in ('standard', 'compact'):
                raise ValueError('profile must be standard or compact')
            cache = 'normal' if profile == 'standard' else 'tight'
            if mode == 'legacy':
                raise ValueError('profile cannot be used with legacy mode')
            # Profile is a BOLT32 layout hint, not an attention-mode override.
        self.profile = profile
        self.graph = normalize_graph_policy(graph)
        self.precision = str(precision).lower()
        if self.precision not in ('auto','fp16','bf16','fp32'):
            raise ValueError('precision must be auto, fp16, bf16 or fp32')
        requested = str(device).strip().lower()
        if requested == 'auto':
            resolved_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            resolved_device = torch.device(device)
            if resolved_device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA was explicitly selected but is unavailable')
        # PyTorch-like constructor keyword: nn.MultiheadAttention(embed_dim=..., num_heads=...).
        # Preserve Bolt(384, 12) and Bolt(d_model=384, num_heads=12) unchanged.
        # Refuse ambiguous duplicate dimension keywords rather than choosing silently.
        if d_model is not None and embed_dim is not None:
            raise TypeError("Specify either 'd_model' or 'embed_dim', not both")
        if d_model is None:
            d_model = embed_dim
        if d_model is None or num_heads is None:
            raise TypeError("Bolt requires 'embed_dim' (or 'd_model') and 'num_heads'")
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError('embed_dim/d_model must be a positive integer')
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError('num_heads must be a positive integer')
        if d_model % num_heads:
            raise ValueError('embed_dim/d_model must be divisible by num_heads')
        # Public alias for compatibility with the naming in torch.nn.MultiheadAttention.
        self.embed_dim = int(d_model)
        # Preserve the public Bolt(d_model, num_heads, ...) API.
        # quantised/default = MQA/GQA; Keyless = one shared C; base = full MHA.
        requested_mode = str(mode).lower().strip() if mode is not None else 'quantised'
        requested_mode = {
            'enhanced_mqa': 'quantised', 'bolt32': 'keyless',
            'bolt': 'keyless', 'classic': 'base',
        }.get(requested_mode, requested_mode)
        # Version 2 modern BOLT: one Pre-AVQ location for every attention mode.
        # Historical per-head-C and post-AVQ attention are not selectable.
        if requested_mode in ('quality', 'fast'):
            if requested_mode == 'fast':
                state_memory = False
            requested_mode = 'keyless'
        if requested_mode in ('quantised', 'base', 'keyless'):
            from .enhanced_mqa import EnhancedMQA, ClassicMHA
            from .shared_c import SharedCKeyless
            if cache == 'tight' or profile == 'compact':
                raise ValueError('BOLT16/compact profiles are not part of the new dynamic-head API')
            if requested_mode == 'keyless' and kv_heads not in (None,):
                raise ValueError('Keyless has one shared C; kv_heads is not applicable')
            if requested_mode != 'keyless' and kv_heads is not None:
                if kv_heads < 1 or num_heads % kv_heads:
                    raise ValueError('kv_heads must divide num_heads')
                if requested_mode == 'base' and kv_heads != num_heads:
                    raise ValueError('base uses full K/V heads; use quantised for GQA')
            if latent_dim is not None and latent_dim != self.embed_dim // num_heads:
                raise ValueError('C/head dimensions are derived from d_model // num_heads')
            cls = {'quantised': EnhancedMQA, 'base': ClassicMHA,
                   'keyless': SharedCKeyless}[requested_mode]
            options = dict(
                forget_gate_init=forget_gate_init, head_output_gate=head_output_gate,
                state_memory=state_memory, bias=bias, dropout=dropout, causal=causal,
                backend=backend, position=position, avq_dim=avq_dim,
                retention=retention,
            )
            if requested_mode == 'quantised':
                options['kv_heads'] = 1 if kv_heads is None else kv_heads
            self._modern = cls(d_model, num_heads, **options)
            self.is_modern = True
            self.mode = requested_mode
            self._modern_architecture = requested_mode
            self.d_model = int(d_model)
            self.num_heads = int(num_heads)
            self.head_dim = self.d_model // self.num_heads
            self.kv_heads = getattr(self._modern, 'kv_heads', None) if requested_mode != 'keyless' else None
            self.latent_dim = self.head_dim
            self.avq_dim = self._modern.avq_dim
            self.state_memory = self._modern.state_memory
            self.causal = self._modern.causal
            self.dropout = self._modern.dropout
            self.cache_mode = 'dynamic'
            self.backend = self._modern.backend
            self.position = self._modern.position
            self.to(resolved_device)
            return
        raise ValueError('mode must be quantised, Keyless, or base; legacy BOLT was removed')


    def set_backend(self, backend: str, *, recursive: bool = True):
        self._modern.set_backend(backend, recursive=recursive)
        self.backend = self._modern.backend
        return self

    def set_graph(self, graph: str | bool):
        self.graph = normalize_graph_policy(graph)
        self._modern.set_graph(self.graph)
        return self

    def graph_allowed(self) -> bool:
        return self._modern.graph_allowed()

    def execution_info(self) -> dict:
        return self._modern.execution_info()

    def allocate_cache(self, *, batch_size: int, capacity: int, device=None, dtype=None):
        """Allocate a fixed-capacity, independent BOLT cached-decode session."""
        weight = next(self.parameters())
        device = weight.device if device is None else torch.device(device)
        if dtype is None:
            dtype = weight.dtype
            if device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
                dtype = torch.get_autocast_dtype('cuda')
            elif device.type == 'cuda' and self.precision != 'fp32':
                dtype = (torch.bfloat16 if self.precision == 'bf16' or
                         (self.precision == 'auto' and native_bf16_supported(device)) else torch.float16)
        return self._modern.allocate_cache(batch_size=batch_size, capacity=int(capacity),
                                           device=device, dtype=dtype)


    def resolved_backend(self) -> str:
        return self._modern.resolved_backend()

    def set_position(self, position):
        self._modern.set_position(position)
        self.position = self._modern.position
        return self

    def set_qk_transform(self, transform):
        return self.set_position(transform)

    @property
    def cache_bytes_per_head_token(self) -> float:
        return self._modern.cache_bytes_per_head_token

    @property
    def cache_bytes_per_token(self) -> int:
        return self._modern.cache_bytes_per_token

    @property
    def baseline_cache_bytes_per_head_token(self) -> int:
        return 4 * self.head_dim


    @property
    def cache_saving_percent(self) -> float:
        return 100.0 * (
            1.0
            - self.cache_bytes_per_head_token
            / self.baseline_cache_bytes_per_head_token
        )


    def _precision_context(self):
        current_device = next(self.parameters()).device
        # BoltModel already provides a model-wide autocast scope. Creating
        # one more autocast manager per BOLT layer is redundant and adds
        # Python dispatcher overhead to every training iteration.
        if current_device.type == 'cuda' and torch.is_autocast_enabled('cuda'):
            return contextlib.nullcontext()
        if current_device.type != 'cuda' or self.precision == 'fp32':
            return contextlib.nullcontext()
        if self.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is not supported on this GPU')
        dtype = (torch.bfloat16 if self.precision == 'bf16' else
                 (torch.bfloat16 if self.precision == 'auto' and native_bf16_supported(current_device)
                  else torch.float16))
        return torch.autocast('cuda', dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device, non_blocking=True)
        with self._precision_context():
            return self._modern(x)

    @torch.no_grad()
    def prefill_with_cache(self, x, *, start_pos=0):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device, non_blocking=True)
        with self._precision_context():
            return self._modern.prefill_with_cache(x, start_pos=start_pos)

    prefill = prefill_with_cache

    @torch.no_grad()
    def decode_step(self, x, cache):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device, non_blocking=True)
        with self._precision_context():
            return self._modern.decode_step(x, cache)

    @torch.no_grad()
    def decode_with_cache(self, x, cache, *, position):
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device, non_blocking=True)
        with self._precision_context():
            return self._modern.decode_with_cache(x, cache, position=position)

    def extra_repr(self) -> str:
        return self._modern.extra_repr()


# Canonical public API.
BoltAttention = Bolt
