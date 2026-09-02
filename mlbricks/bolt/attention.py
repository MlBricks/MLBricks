from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .._reference import attention_forward_reference, gauss_forward_reference
from ..runtime import normalize_backend
from ..planner import EXECUTION_PLANNER
from ..position import RoPE

from .._backend import (
    WORKSPACES,
    autotune,
    autotune_gauss_rope,
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
    Routed gated shared-latent Bolt attention.

    Mathematics is unchanged:
        U = X W_C
        G = X W_G
        C = U * (1 + tanh(G))
        rho = rsqrt(mean(C^2) + eps)
        scores = (Q C^T) * rho / sqrt(head_dim)
        output = softmax(scores) C
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        latent_dim: int = 32,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = True,
        backend: str = "auto",
        autotune_kernels: bool = True,
        eps: float = 1e-6,
        use_sdpa: bool = True,
        position=None,
        native_full_sequence: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        self.backend = _validate_backend(backend)
        self.autotune_kernels = bool(autotune_kernels)
        self.eps = float(eps)
        self.use_sdpa = bool(use_sdpa)
        self.position = _resolve_rope(position, self.latent_dim)
        self.native_full_sequence = bool(native_full_sequence)

        hr = self.num_heads * self.latent_dim
        self.q_proj = nn.Linear(d_model, hr, bias=bias)
        self.c_proj = nn.Linear(d_model, hr, bias=bias)
        self.g_proj = nn.Linear(d_model, hr, bias=bias)
        self.out_proj = nn.Linear(hr, d_model, bias=bias)

        if self.g_proj.bias is not None:
            nn.init.zeros_(self.g_proj.bias)
        nn.init.zeros_(self.g_proj.weight)

        # Inference-only packed Q/C/G cache. It preserves the original three
        # Parameters/state_dict keys; the concatenation is rebuilt only when a
        # source Parameter version changes.
        self._packed_qcg_cache = None
        self._packed_qcg_versions = None

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
        if self.native_full_sequence:
            return "planner(auto full-sequence) + native-decode"
        return "planner(auto forward) + native-decode"

    def set_position(self, position):
        self.position = _resolve_rope(position, self.latent_dim)
        return self

    @property
    def cache_bytes_per_head_token(self) -> int:
        return 2 * self.latent_dim + 2

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

    def _q_c(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x)
        u = self.c_proj(x)
        g = self.g_proj(x)
        c = u * (1.0 + torch.tanh(g))
        q = q.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        c = c.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        return q, c

    def _packed_qcg(self):
        """Return cached concatenated Q/C/G projection parameters for inference."""
        weights = (self.q_proj.weight, self.c_proj.weight, self.g_proj.weight)
        biases = (self.q_proj.bias, self.c_proj.bias, self.g_proj.bias)
        stamp = tuple(
            (int(w._version), str(w.device), str(w.dtype), int(w.data_ptr())) for w in weights
        ) + tuple(-1 if b is None else int(b._version) for b in biases)
        if self._packed_qcg_cache is None or self._packed_qcg_versions != stamp:
            weight = torch.cat([w.detach() for w in weights], dim=0).contiguous()
            bias = None if biases[0] is None else torch.cat([b.detach() for b in biases], dim=0).contiguous()
            self._packed_qcg_cache = (weight, bias)
            self._packed_qcg_versions = stamp
        return self._packed_qcg_cache

    @torch.no_grad()
    def _qcg_inference(self, x: torch.Tensor):
        weight, bias = self._packed_qcg()
        qcg = F.linear(x, weight, bias)
        hr = self.num_heads * self.latent_dim
        return qcg.split(hr, dim=-1)

    def _pytorch_full_eval_core(
        self, q_flat: torch.Tensor, u_flat: torch.Tensor, g_flat: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch full-sequence Bolt core used for one-time auto qualification."""
        B, T, _ = q_flat.shape
        c_flat = u_flat * (1.0 + torch.tanh(g_flat))
        q = q_flat.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        c = c_flat.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        rho = torch.rsqrt(c.float().square().mean(dim=-1) + self.eps)
        if self.use_sdpa:
            k = (c.float() * rho.unsqueeze(-1)).to(c.dtype)
            y = _sdpa(
                q, k, c, scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=self.causal, dropout_p=0.0, training=False,
            )
        else:
            scores = torch.matmul(q, c.transpose(-2, -1))
            scores = scores * rho.unsqueeze(-2)
            scores = scores * (1.0 / math.sqrt(float(self.head_dim)))
            if self.causal:
                mask = torch.ones(T, T, device=q_flat.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(c.dtype)
            y = torch.matmul(p, c)
        return y.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must have shape [B,T,D]")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected D={self.d_model}, got {D}")

        q_flat = self.q_proj(x)
        u_flat = self.c_proj(x)
        g_flat = self.g_proj(x)

        # Full-sequence native vision path. The custom C++ operator owns the
        # Bolt equation while its matmul/softmax operations dispatch through
        # ATen to CPU/CUDA and retain autograd. VisionBolt always uses
        # position=None internally; explicit Bolt RoPE keeps the proven Python
        # path below.
        if self.native_full_sequence and self.position is None and self.dropout == 0.0 and self.backend != "pytorch":
            fused_inference = bool(not self.training and not torch.is_grad_enabled())
            native_backend = self.backend

            # Element-wise auto: benchmark this Bolt instance/workload once,
            # freeze native or PyTorch, and never re-check transient load.
            if self.backend == "auto":
                extra = (
                    int(self.num_heads), int(self.latent_dim),
                    int(self.head_dim), bool(self.causal), fused_inference,
                )
                is_training = bool(torch.is_grad_enabled())
                frozen = EXECUTION_PLANNER.owner_routes(self).get(("bolt_full", is_training))
                if frozen in {"native", "pytorch"}:
                    route = frozen
                    native_ok = frozen == "native"
                    _vision_bolt_full = None
                else:
                    native_ok = False
                    _vision_bolt_full = None
                    try:
                        from ..vision_native import (
                            bolt_full as _vision_bolt_full,
                            available as _vision_native_available,
                            cuda_built as _vision_native_cuda_built,
                        )
                        native_ok = bool(_vision_native_available())
                        if q_flat.is_cuda:
                            native_ok = native_ok and bool(_vision_native_cuda_built())
                    except Exception:
                        native_ok = False

                if frozen not in {"native", "pytorch"} and not torch.is_grad_enabled() and native_ok and _vision_bolt_full is not None:
                    route = EXECUTION_PLANNER.qualify_operator_once(
                        self,
                        "bolt_full",
                        q_flat,
                        {
                            "native": lambda: _vision_bolt_full(
                                q_flat, u_flat, g_flat,
                                heads=self.num_heads, latent_dim=self.latent_dim,
                                head_dim=self.head_dim, eps=self.eps,
                                causal=self.causal, backend="native",
                                fused_inference=fused_inference, owner=self,
                            ),
                            "pytorch": lambda: self._pytorch_full_eval_core(
                                q_flat, u_flat, g_flat
                            ),
                        },
                        requested_backend="auto",
                        native_available=True,
                        native_supports_training=True,
                        training=False,
                        extra=extra,
                        default_auto="pytorch",
                    )
                elif frozen not in {"native", "pytorch"}:
                    route = EXECUTION_PLANNER.select_operator_once(
                        self,
                        "bolt_full",
                        q_flat,
                        requested_backend="auto",
                        native_available=native_ok,
                        native_supports_training=True,
                        training=bool(torch.is_grad_enabled()),
                        extra=extra,
                        default_auto="pytorch",
                    )
                native_backend = "native" if route == "native" else "pytorch"

            if native_backend == "native":
                try:
                    from ..vision_native import bolt_full as _vision_bolt_full
                    native_y = _vision_bolt_full(
                        q_flat, u_flat, g_flat,
                        heads=self.num_heads,
                        latent_dim=self.latent_dim,
                        head_dim=self.head_dim,
                        eps=self.eps,
                        causal=self.causal,
                        # Route already resolved above; avoid planner work in
                        # vision_native._use_native for steady-state auto.
                        backend="native",
                        fused_inference=fused_inference, owner=self,
                    )
                except RuntimeError:
                    if self.backend == "native":
                        raise
                    native_y = None
                if native_y is not None:
                    return self.out_proj(native_y)

        c_flat = u_flat * (1.0 + torch.tanh(g_flat))
        q = q_flat.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        c = c_flat.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2)
        rho = torch.rsqrt(c.float().square().mean(dim=-1) + self.eps)
        key_c = c
        if self.position is not None:
            q = self.position(q, start_pos=0)
            key_c = self.position(c, start_pos=0)

        # Quality-safe policy:
        # Keep the exact original Gauss equation/order whenever gradients are
        # enabled in training. This removes SDPA-induced gradient drift while
        # retaining SDPA for fast eval/prefill. No parameters or mathematics
        # are changed.
        if self.position is None and self.training and torch.is_grad_enabled():
            y = gauss_forward_reference(
                q, c,
                head_dim=self.head_dim,
                eps=self.eps,
                causal=self.causal,
                dropout_p=self.dropout,
                training=True,
            )
        elif self.use_sdpa:
            # Key-only normalization. With RoPE, only the key view rotates;
            # raw C remains the value so the shared latent cache stays compact.
            k = (key_c.float() * rho.unsqueeze(-1)).to(c.dtype)
            y = _sdpa(
                q, k, c,
                scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=self.causal,
                dropout_p=self.dropout,
                training=self.training,
            )
        else:
            scores = torch.matmul(q, key_c.transpose(-2, -1))
            scores = scores * rho.unsqueeze(-2)
            scores = scores * (1.0 / math.sqrt(float(self.head_dim)))
            if self.causal:
                mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(c.dtype)
            p = F.dropout(p, p=self.dropout, training=self.training)
            y = torch.matmul(p, c)

        y = y.transpose(1, 2).contiguous().view(
            B, T, self.num_heads * self.latent_dim
        )
        return self.out_proj(y)

    def _native_gate_rho(self, u: torch.Tensor, g: torch.Tensor):
        ext = None
        eligible = (
            u.is_cuda
            and u.dtype == torch.float16
            and g.dtype == torch.float16
            and self.latent_dim <= 64
        )
        if _planner_native_decode(
            self, "bolt_decode", u, self.backend, eligible,
            extra=(self.num_heads, self.latent_dim, "gate_rho"),
        ):
            ext = load_cuda_extension()
        if self.backend == "native" and (not eligible or ext is None):
            raise RuntimeError(
                "backend='native' requested but native Gauss preprocessing is unavailable"
            )
        if ext is None:
            c = u * (1.0 + torch.tanh(g))
            rho = torch.rsqrt(c.float().square().mean(dim=-1) + self.eps).to(c.dtype)
            return c, rho
        return ext.gauss_gate_rho(u.contiguous(), g.contiguous(), self.eps)

    @torch.no_grad()
    def project_cache_state(self, x: torch.Tensor, *, start_pos: int = 0):
        if x.dim() == 2:
            x = x[:, None, :]
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        B, T, _ = x.shape
        u = self.c_proj(x).view(B * T, self.num_heads, self.latent_dim).contiguous()
        g = self.g_proj(x).view(B * T, self.num_heads, self.latent_dim).contiguous()
        c, rho = self._native_gate_rho(u, g)
        c = c.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2).contiguous()
        rho = rho.view(B, T, self.num_heads).transpose(1, 2).contiguous()
        return c, rho

    @torch.no_grad()
    def prefill_with_cache(self, x: torch.Tensor, *, start_pos: int = 0):
        """Fast eval prefill: one packed Q/C/G GEMM produces output and cache."""
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        B, T, _ = x.shape
        q0, u0, g0 = self._qcg_inference(x)
        q = q0.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2).contiguous()
        u = u0.view(B * T, self.num_heads, self.latent_dim).contiguous()
        g = g0.view(B * T, self.num_heads, self.latent_dim).contiguous()
        c, rho = self._native_gate_rho(u, g)
        c = c.view(B, T, self.num_heads, self.latent_dim).transpose(1, 2).contiguous()
        rho = rho.view(B, T, self.num_heads).transpose(1, 2).contiguous()
        # Preserve the eval-forward normalization precision for the output.
        # The compact cache may keep rho in FP16, but prefill scoring uses the
        # FP32 rho value just like forward().
        rho_score = torch.rsqrt(c.float().square().mean(dim=-1) + self.eps)
        key_c = c
        if self.position is not None:
            q = self.position(q, start_pos=int(start_pos))
            key_c = self.position(c, start_pos=int(start_pos))
        if self.use_sdpa:
            k = (key_c.float() * rho_score.unsqueeze(-1)).to(c.dtype)
            y = _sdpa(
                q, k, c, scale=1.0 / math.sqrt(float(self.head_dim)),
                causal=self.causal, dropout_p=0.0, training=False,
            )
        else:
            scores = torch.matmul(q, key_c.transpose(-2, -1))
            scores = scores * rho_score.unsqueeze(-2)
            scores = scores * (1.0 / math.sqrt(float(self.head_dim)))
            if self.causal:
                mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, float("-inf"))
            p = torch.softmax(scores.float(), dim=-1).to(c.dtype)
            y = torch.matmul(p, c)
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.latent_dim)
        return self.out_proj(y), (c, rho)

    @torch.no_grad()
    def prefill(
        self, x: torch.Tensor, *, start_pos: int = 0
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Process a prefix and return outputs plus recurrent Bolt cache.

        The cache stores the compact shared-latent ``C`` tensor and its
        per-token normalization ``rho``. This is the public recurrent mixer
        interface used by compositional models such as SOUP.
        """
        return self.prefill_with_cache(x, start_pos=int(start_pos))

    @torch.no_grad()
    def decode_step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Decode one token and append it to the compact Bolt cache.

        ``x`` may be ``[B,D]`` or ``[B,1,D]``. The returned output always
        has shape ``[B,1,D]`` to match ESA/SOUP's recurrent mixer contract.
        """
        if x.dim() == 2:
            x = x[:, None, :]
        if x.dim() != 3 or x.size(1) != 1 or x.size(-1) != self.d_model:
            raise ValueError(f"decode_step expects x with shape [B,1,{self.d_model}]")
        if not isinstance(cache, (tuple, list)) or len(cache) != 2:
            raise TypeError("Bolt cache must be a (c_cache, rho_cache) pair")

        c_cache, rho_cache = cache
        if c_cache.dim() != 4:
            raise ValueError("c_cache must have shape [B,H,T,latent_dim]")
        if rho_cache.dim() != 3:
            raise ValueError("rho_cache must have shape [B,H,T]")
        if (
            c_cache.size(0) != x.size(0)
            or c_cache.size(1) != self.num_heads
            or c_cache.size(3) != self.latent_dim
            or rho_cache.shape != c_cache.shape[:3]
        ):
            raise ValueError("Bolt cache shape does not match this module or input batch")
        if c_cache.device != x.device or rho_cache.device != x.device:
            raise ValueError("x and Bolt cache must be on the same device")

        position = int(c_cache.size(2))
        q, c_now, rho_now = self.project_decode_state(x, start_pos=position)
        c_next = torch.cat((c_cache, c_now), dim=2)
        rho_next = torch.cat((rho_cache, rho_now), dim=2)
        y = self.decode_projected(
            q, c_next, rho_next, used_length=position + 1
        )[:, None, :]
        return y, (c_next, rho_next)

    lightning_prefill = prefill
    lightning_step = decode_step

    @torch.no_grad()
    def project_decode_state(self, x: torch.Tensor, *, start_pos: int):
        """One packed GEMM for current-token Q/C/G plus cache preprocessing."""
        if x.dim() == 3:
            if x.size(1) != 1:
                raise ValueError("project_decode_state expects one token")
            x = x[:, 0, :]
        B = x.size(0)
        q0, u0, g0 = self._qcg_inference(x)
        q = q0.view(B, self.num_heads, self.latent_dim).contiguous()
        u = u0.view(B, self.num_heads, self.latent_dim).contiguous()
        g = g0.view(B, self.num_heads, self.latent_dim).contiguous()
        c, rho = self._native_gate_rho(u, g)
        c = c[:, :, None, :].contiguous()
        rho = rho[:, :, None].contiguous()
        if self.position is not None:
            q = self.position(q[:, :, None, :], start_pos=int(start_pos))[:, :, 0, :].contiguous()
        return q, c, rho

    @torch.no_grad()
    def decode_projected(
        self, q: torch.Tensor, c_cache: torch.Tensor, rho_cache: torch.Tensor, *,
        used_length: int | None = None, force_retune: bool = False,
    ) -> torch.Tensor:
        B = q.size(0)
        capacity = int(c_cache.size(2))
        used = capacity if used_length is None else int(used_length)
        if used < 1 or used > capacity:
            raise ValueError("used_length must be in [1, cache_capacity]")
        tensor_eligible = (
            q.is_cuda and q.dtype == torch.float16
            and c_cache.dtype == torch.float16 and rho_cache.dtype == torch.float16
            and self.latent_dim <= 64 and self.head_dim <= 64
            and q.is_contiguous() and c_cache.is_contiguous() and rho_cache.is_contiguous()
        )
        rope_width = 0
        if isinstance(self.position, RoPE):
            rope_width = self.latent_dim if self.position.dim is None else min(
                int(self.position.dim), self.latent_dim
            )
            rope_width -= rope_width % 2
        ext = load_cuda_extension() if _planner_native_decode(
            self, "bolt_decode", q, self.backend, tensor_eligible,
            extra=(self.num_heads, used, self.latent_dim, self.head_dim, "projected"),
        ) else None
        plain_native = self.position is None and ext is not None
        rope_native = (
            rope_width >= 2 and ext is not None
            and hasattr(ext, "gauss_rope_decode_out_used")
        )
        if plain_native and used != capacity and not hasattr(ext, "gauss_decode_out_used"):
            plain_native = False
        if self.backend == "native" and not (plain_native or rope_native):
            raise RuntimeError(
                "backend='native' requested but native projected Gauss decode is unavailable"
            )

        if rope_native:
            if self.autotune_kernels:
                cfg = autotune_gauss_rope(
                    q=q, c=c_cache, rho=rho_cache, head_dim=self.head_dim, ext=ext,
                    rope_base=self.position.base, rope_dim=rope_width,
                    force=force_retune, used_length=used,
                )
            else:
                cfg = heuristic_config(
                    kind="gauss_rope", B=B, H=self.num_heads, T=used, W=self.latent_dim
                )
            ws = WORKSPACES.get(
                "gauss_rope", B, self.num_heads, self.latent_dim, cfg.splits, q.device
            )
            ext.gauss_rope_decode_out_used(
                q, c_cache, rho_cache, ws.pm, ws.pl, ws.po, ws.out,
                1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits, used,
                float(self.position.base), rope_width,
            )
            y = ws.out
        elif plain_native:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="gauss", q=q, a=c_cache, b=rho_cache, head_dim=self.head_dim,
                    ext=ext, force=force_retune,
                    used_length=used if used != capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="gauss", B=B, H=self.num_heads, T=used, W=self.latent_dim
                )
            ws = WORKSPACES.get(
                "gauss", B, self.num_heads, self.latent_dim, cfg.splits, q.device
            )
            if used != capacity:
                ext.gauss_decode_out_used(
                    q, c_cache, rho_cache, ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits, used,
                )
            else:
                ext.gauss_decode_out(
                    q, c_cache, rho_cache, ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits,
                )
            y = ws.out
        else:
            c_used = c_cache[:, :, :used, :]
            rho_used = rho_cache[:, :, :used]
            key_cache = c_used
            if self.position is not None:
                key_cache = self.position(c_used, start_pos=0)
            # Decode dispatch is representation-aware.  When Bolt is actually
            # compressed (latent_dim < head_dim), keep the historical compact
            # C+rho execution form.  Materializing K=rho*C for the whole cache
            # every token defeats the bandwidth/temporary-memory advantage that
            # compressed Bolt is designed to provide at long context.
            #
            # Parameter-matched / expanded Bolt keeps the SDPA route measured
            # faster for those shapes.  Both branches implement the same Bolt
            # equation and retain the same persistent C+rho cache.
            use_sdpa_decode = self.use_sdpa and self.latent_dim >= self.head_dim
            if use_sdpa_decode:
                k = (key_cache.float() * rho_used.unsqueeze(-1).float()).to(c_used.dtype)
                y = _sdpa(
                    q[:, :, None, :], k, c_used,
                    scale=1.0 / math.sqrt(float(self.head_dim)),
                    causal=False, dropout_p=0.0, training=False,
                )[:, :, 0, :]
            else:
                scores = torch.matmul(q[:, :, None, :], key_cache.transpose(-2, -1))
                scores = scores * rho_used[:, :, None, :].float()
                scores = scores * (1.0 / math.sqrt(float(self.head_dim)))
                p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
                y = torch.matmul(p, c_used)[:, :, 0, :]
        return self.out_proj(y.reshape(B, self.num_heads * self.latent_dim))

    @torch.no_grad()
    def decode_append_projected(
        self,
        q: torch.Tensor,
        c_now: torch.Tensor,
        rho_now: torch.Tensor,
        c_cache: torch.Tensor,
        rho_cache: torch.Tensor,
        *,
        position: int,
        force_retune: bool = False,
    ) -> torch.Tensor:
        """Append current C/rho and decode with minimal launch overhead.

        FP16 CUDA Gauss and Gauss+RoPE both use native fused append+decode.
        RoPE rotates only the key view inside CUDA; raw C remains the compact
        cache and value tensor, so no rotated-cache allocation is created.
        """
        B = q.size(0)
        pos = int(position)
        capacity = int(c_cache.size(2))
        if pos < 0 or pos >= capacity:
            raise ValueError("position must be in [0, cache_capacity)")
        if c_now.dim() == 4:
            if c_now.size(2) != 1:
                raise ValueError("c_now must contain exactly one token")
            c_now3 = c_now[:, :, 0, :]
        else:
            c_now3 = c_now
        if rho_now.dim() == 3:
            if rho_now.size(2) != 1:
                raise ValueError("rho_now must contain exactly one token")
            rho_now2 = rho_now[:, :, 0]
        else:
            rho_now2 = rho_now
        if c_now3.shape != q.shape or rho_now2.shape != q.shape[:2]:
            raise ValueError("projected q/C/rho shapes are incompatible")

        append_eligible = (
            q.is_cuda and q.dtype == torch.float16
            and c_now3.dtype == torch.float16 and rho_now2.dtype == torch.float16
            and c_cache.dtype == torch.float16 and rho_cache.dtype == torch.float16
            and self.latent_dim <= 64 and self.head_dim <= 64
            and q.is_contiguous() and c_now3.is_contiguous() and rho_now2.is_contiguous()
            and c_cache.is_contiguous() and rho_cache.is_contiguous()
        )
        ext = load_cuda_extension() if _planner_native_decode(
            self, "bolt_decode", q, self.backend, append_eligible,
            extra=(self.num_heads, pos + 1, self.latent_dim, self.head_dim, "append"),
        ) else None
        rope_width = 0
        if isinstance(self.position, RoPE):
            rope_width = self.latent_dim if self.position.dim is None else min(
                int(self.position.dim), self.latent_dim
            )
            rope_width -= rope_width % 2
        plain_fused = bool(
            self.position is None and ext is not None
            and hasattr(ext, "gauss_decode_append_out")
        )
        rope_fused = bool(
            rope_width >= 2 and ext is not None
            and hasattr(ext, "gauss_rope_decode_append_out")
        )
        if self.backend == "native" and not (plain_fused or rope_fused):
            raise RuntimeError(
                "backend='native' requested but native Gauss decode is unavailable"
            )

        used = pos + 1
        if rope_fused:
            if self.autotune_kernels:
                cfg = autotune_gauss_rope(
                    q=q, c=c_cache, rho=rho_cache, head_dim=self.head_dim, ext=ext,
                    rope_base=self.position.base, rope_dim=rope_width,
                    force=force_retune, used_length=used,
                )
            else:
                cfg = heuristic_config(
                    kind="gauss_rope", B=B, H=self.num_heads, T=used, W=self.latent_dim
                )
            ws = WORKSPACES.get(
                "gauss_rope", B, self.num_heads, self.latent_dim, cfg.splits, q.device
            )
            ext.gauss_rope_decode_append_out(
                q, c_now3, rho_now2, c_cache, rho_cache,
                ws.pm, ws.pl, ws.po, ws.out,
                1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits, pos,
                float(self.position.base), rope_width,
            )
            y = ws.out
            return self.out_proj(y.reshape(B, self.num_heads * self.latent_dim))

        if plain_fused:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="gauss", q=q, a=c_cache, b=rho_cache,
                    head_dim=self.head_dim, ext=ext, force=force_retune,
                    used_length=used if used != capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="gauss", B=B, H=self.num_heads, T=used, W=self.latent_dim
                )
            ws = WORKSPACES.get(
                "gauss", B, self.num_heads, self.latent_dim, cfg.splits, q.device
            )
            ext.gauss_decode_append_out(
                q, c_now3, rho_now2, c_cache, rho_cache,
                ws.pm, ws.pl, ws.po, ws.out,
                1.0 / math.sqrt(float(self.head_dim)), cfg.mode, cfg.splits, pos,
            )
            y = ws.out
            return self.out_proj(y.reshape(B, self.num_heads * self.latent_dim))

        if ext is not None and hasattr(ext, "gauss_append_cache"):
            ext.gauss_append_cache(c_now3, rho_now2, c_cache, rho_cache, pos)
        else:
            c_cache[:, :, pos:pos + 1, :].copy_(c_now3[:, :, None, :])
            rho_cache[:, :, pos:pos + 1].copy_(rho_now2[:, :, None])
        return self.decode_projected(
            q, c_cache, rho_cache, used_length=used, force_retune=force_retune
        )

    @torch.no_grad()
    def decode(
        self,
        x: torch.Tensor,
        c_cache: torch.Tensor,
        rho_cache: torch.Tensor,
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
        if c_cache.dim() != 4 or rho_cache.dim() != 3:
            raise ValueError("c_cache/rho_cache must be [B,H,T,R] and [B,H,T]")
        if (
            c_cache.size(0) != B
            or c_cache.size(1) != self.num_heads
            or c_cache.size(3) != self.latent_dim
            or rho_cache.shape != c_cache.shape[:3]
        ):
            raise ValueError("Gauss cache shape does not match this module")
        if c_cache.size(2) < 1:
            raise ValueError("decode cache must contain at least one token")
        if c_cache.device != x.device or rho_cache.device != x.device:
            raise ValueError("x and Gauss caches must be on the same device")
        cache_capacity = int(c_cache.size(2))
        used = cache_capacity if used_length is None else int(used_length)
        if used < 1 or used > cache_capacity:
            raise ValueError("used_length must be in [1, cache_capacity]")

        q = self.q_proj(x).view(B, self.num_heads, self.latent_dim).contiguous()
        q_pos = used if position is None else int(position)
        if self.position is not None:
            q = self.position(q[:, :, None, :], start_pos=q_pos)[:, :, 0, :].contiguous()
        native_eligible = (
            self.position is None
            and
            q.is_cuda
            and q.dtype == torch.float16
            and c_cache.dtype == torch.float16
            and rho_cache.dtype == torch.float16
            and self.latent_dim <= 64
            and self.head_dim <= 64
            and c_cache.is_contiguous()
            and rho_cache.is_contiguous()
        )

        ext = None
        if _planner_native_decode(
            self, "bolt_decode", q, self.backend, native_eligible,
            extra=(self.num_heads, used, self.latent_dim, self.head_dim, "decode"),
        ):
            ext = load_cuda_extension()
        native_used_api = bool(ext is not None and hasattr(ext, "gauss_decode_out_used"))
        if ext is not None and used != cache_capacity and not native_used_api:
            ext = None
        if self.backend == "native" and (not native_eligible or ext is None):
            raise RuntimeError(
                "backend='native' requested but the native FP16 CUDA Gauss decode path is unavailable"
            )

        if ext is None:
            c_used = c_cache[:, :, :used, :]
            rho_used = rho_cache[:, :, :used]
            key_cache = c_used
            if self.position is not None:
                key_cache = self.position(c_used, start_pos=0)
            scores = torch.matmul(q[:, :, None, :], key_cache.transpose(-2, -1))
            scores = scores * rho_used[:, :, None, :].float()
            scores = scores * (1.0 / math.sqrt(float(self.head_dim)))
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            y = torch.matmul(p, c_used)[:, :, 0, :]
        else:
            if self.autotune_kernels:
                cfg = autotune(
                    kind="gauss",
                    q=q,
                    a=c_cache,
                    b=rho_cache,
                    head_dim=self.head_dim,
                    ext=ext,
                    force=force_retune,
                    used_length=used if used != cache_capacity else None,
                )
            else:
                cfg = heuristic_config(
                    kind="gauss",
                    B=B,
                    H=self.num_heads,
                    T=used,
                    W=self.latent_dim,
                )
            ws = WORKSPACES.get(
                "gauss",
                B,
                self.num_heads,
                self.latent_dim,
                cfg.splits,
                q.device,
            )
            if used != cache_capacity:
                ext.gauss_decode_out_used(
                    q, c_cache, rho_cache,
                    ws.pm, ws.pl, ws.po, ws.out,
                    1.0 / math.sqrt(float(self.head_dim)),
                    cfg.mode, cfg.splits, used,
                )
            else:
                ext.gauss_decode_out(
                    q,
                    c_cache,
                    rho_cache,
                    ws.pm,
                    ws.pl,
                    ws.po,
                    ws.out,
                    1.0 / math.sqrt(float(self.head_dim)),
                    cfg.mode,
                    cfg.splits,
                )
            y = ws.out

        return self.out_proj(y.reshape(B, self.num_heads * self.latent_dim))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, latent_dim={self.latent_dim}, "
            f"cache_saving={self.cache_saving_percent:.2f}%, backend={self.backend!r}, "
            f"position={'rope' if self.position is not None else 'none'}"
        )


# Canonical public API for MLBricks v1.0.0.
BoltAttention = Bolt
