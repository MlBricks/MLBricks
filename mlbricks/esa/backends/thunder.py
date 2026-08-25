# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..precision import resolve_scan_dtype


def associative_chunk_scan(
    A_chunk: torch.Tensor,
    B_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive affine scan over chunk summaries.

    Composition rule for:

        E_t = A_t * E_{t-1} + B_t

    is:

        (A2, B2) ∘ (A1, B1) = (A2 * A1, A2 * B1 + B2)

    Args:
        A_chunk: Tensor of shape [B, G, H, D].
        B_chunk: Tensor of shape [B, G, H, D].

    Returns:
        Tuple (A_scan, B_scan), both of shape [B, G, H, D].
    """
    if A_chunk.shape != B_chunk.shape:
        raise ValueError(
            f"A_chunk and B_chunk must have same shape, "
            f"got {A_chunk.shape} and {B_chunk.shape}"
        )

    if A_chunk.dim() != 4:
        raise ValueError(
            f"expected A_chunk/B_chunk shape [B,G,H,D], got {A_chunk.shape}"
        )

    A = A_chunk
    B = B_chunk

    G = A.size(1)
    step = 1

    while step < G:
        A_prev = A
        B_prev = B

        A_next = A.clone()
        B_next = B.clone()

        A_next[:, step:] = A_prev[:, step:] * A_prev[:, :-step]
        B_next[:, step:] = (
            A_prev[:, step:] * B_prev[:, :-step] + B_prev[:, step:]
        )

        A = A_next
        B = B_next
        step *= 2

    return A, B


def _thunder_scan_auto(
    A: torch.Tensor,
    B_write: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Internal auto-planned Thunder scan used by the public ESA layer."""
    from ...runtime import normalize_backend
    policy = normalize_backend(backend, warn_legacy=True)
    if policy != "pytorch":
        try:
            import os
            from ..native import thunder_scan_planned as _native_planned_scan
            from ..native import available as _native_available
            native_disabled = os.getenv("MLBRICKS_DISABLE_NATIVE", "0") == "1"
            if A.is_cuda and not native_disabled and _native_available():
                return _native_planned_scan(A, B_write, "auto")
        except Exception:
            import os
            if policy == "native" or os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                raise
        if policy == "native":
            raise RuntimeError("ESA backend='native' requested but native scan is unavailable")

    # Native-unavailable/disabled fallback preserves historical C16 math.
    return thunder_scan(A, B_write, compass=16, backend="pytorch")


def thunder_scan(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int | None = None,
    c: int | None = None,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Thunder chunked ESA scan for an explicit integer compass.

    ``compass='auto'`` is intentionally not part of this low-level public
    helper. Use ``ESA(..., backend='thunder', compass='auto')`` for automatic
    execution planning.
    """
    if c is not None:
        compass = c
    if compass is None:
        raise ValueError("thunder_scan requires an explicit positive integer compass.")

    if isinstance(compass, str):
        raise ValueError(
            "Automatic compass planning is available through "
            "ESA(..., backend='thunder', compass='auto') only."
        )

    from ...runtime import normalize_backend
    policy = normalize_backend(backend, warn_legacy=True)
    if policy != "pytorch":
        try:
            from ..native import enabled_for as _native_enabled_for
            from ..native import thunder_scan as _native_thunder_scan
            from ..native import available as _native_available
            if _native_available() and _native_enabled_for(A):
                return _native_thunder_scan(A, B_write, int(compass))
        except Exception:
            import os
            if policy == "native" or os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                raise
        if policy == "native":
            raise RuntimeError("ESA backend='native' requested but native scan is unavailable for this call")

    if A.shape != B_write.shape:
        raise ValueError(
            f"A and B_write must have same shape, got {A.shape} and {B_write.shape}"
        )

    if A.dim() != 4:
        raise ValueError(f"expected A/B_write shape [B,T,H,D], got {A.shape}")

    if not isinstance(compass, int) or compass <= 0:
        raise ValueError(f"compass must be a positive integer, got {compass!r}")

    Bsz, T, H, D = A.shape
    pad = (-T) % compass

    if pad > 0:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        B_write = F.pad(B_write, (0, 0, 0, 0, 0, pad), value=0.0)

    Tp = A.size(1)
    G = Tp // compass

    A5 = A.reshape(Bsz, G, compass, H, D)
    B5 = B_write.reshape(Bsz, G, compass, H, D)

    state = B_write.new_zeros(Bsz, G, H, D)
    transition = A.new_ones(Bsz, G, H, D)

    local_states = []
    prefix_As = []

    for i in range(compass):
        A_i = A5[:, :, i]
        B_i = B5[:, :, i]
        state = A_i * state + B_i
        transition = A_i * transition
        local_states.append(state)
        prefix_As.append(transition)

    local_state = torch.stack(local_states, dim=2)
    prefix_A = torch.stack(prefix_As, dim=2)

    A_chunk = prefix_A[:, :, -1]
    B_chunk = local_state[:, :, -1]

    _, chunk_end_state = associative_chunk_scan(A_chunk, B_chunk)

    zero = chunk_end_state.new_zeros(Bsz, 1, H, D)
    chunk_init = torch.cat([zero, chunk_end_state[:, :-1]], dim=1)

    E = prefix_A * chunk_init.unsqueeze(2) + local_state
    E = E.reshape(Bsz, Tp, H, D)
    return E[:, :T]


class ThunderESA(nn.Module):
    """Thunder backend: optimized chunked ESA backend.

    Thunder is the only production ESA backend. It uses the internal ESA
    compass value for chunked associative state scanning.

    Compass planning is automatic by default. Advanced users may pass a positive integer manually.

    Clean public usage:

        ESA(embd=128, head=4, batch=16, block=1024, backend="thunder")

    Backward-compatible usage is also supported:

        ESA(n_embd=128, n_head=4, backend="thunder")

    This implementation follows the CF-ESA-c16-FP16Scan path with:

        compass="auto" by default (or a positive manual integer)
        precision="fp16"
        gate_min=0.80
        gate_max=0.995
        eps=1e-5
    """

    def __init__(
        self,

        # New clean names
        embd: int | None = None,
        head: int = 4,

        # Old names for compatibility
        n_embd: int | None = None,
        n_head: int | None = None,

        dropout: float = 0.0,

        # New internal name
        compass: int | str = "auto",

        # Old name for compatibility
        c: int | None = None,

        precision: str = "fp16",
        gate_min: float = 0.80,
        gate_max: float = 0.995,
        eps: float = 1e-5,
        strict_precision: bool = False,
        runtime_backend: str = "auto",
    ):
        super().__init__()

        # ----------------------------------------------------
        # Resolve new names and old names
        # ----------------------------------------------------
        if embd is None:
            embd = n_embd

        if n_head is not None:
            head = n_head

        if c is not None:
            compass = c

        if embd is None:
            raise ValueError(
                "ThunderESA requires embd. Example: ThunderESA(embd=128, head=4)"
            )

        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={embd}, head={head}"
            )

        if isinstance(compass, str):
            if compass.strip().lower() != "auto":
                raise ValueError("compass must be a positive integer or 'auto'")
            compass = "auto"
        elif not isinstance(compass, int) or compass <= 0:
            raise ValueError(
                f"compass must be a positive integer or 'auto', got {compass!r}"
            )

        # Clean public/internal attributes
        self.embd = embd
        self.head = head
        self.head_dim = embd // head
        self.compass = compass

        # Backward-compatible attributes
        self.n_embd = embd
        self.n_head = head
        self.c = compass

        self.precision = precision
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.eps = eps
        self.strict_precision = strict_precision
        from ...runtime import normalize_backend
        self.runtime_backend = normalize_backend(runtime_backend, warn_legacy=True)

        # Match the optimized benchmark: bias=False.
        self.qgv = nn.Linear(embd, 3 * embd, bias=False)
        self.out_proj = nn.Linear(embd, embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected input shape [B,T,C], got {x.shape}")

        B, T, C = x.shape

        if self.runtime_backend == "native":
            from ..native import available as _native_available
            if not _native_available():
                raise RuntimeError("ESA backend='native' requested but the MLBricks native extension is unavailable")
            if torch.is_grad_enabled():
                raise RuntimeError("ESA backend='native' is currently an eager inference path; use auto/pytorch for training")

        if C != self.embd:
            raise ValueError(
                f"expected embedding dim {self.embd}, got input dim {C}"
            )

        # Performance-aware Auto routing. Explicit native/pytorch requests are
        # never changed. Training keeps the existing auto/native-autograd scan
        # behavior; inference can choose the qualified PyTorch compiled path.
        effective_backend = self.runtime_backend
        if self.runtime_backend == "auto":
            try:
                from ..auto_backend import select_esa_auto_backend
                from ..native import available as _native_available
                from ..native import cuda_available as _native_cuda_available
                effective_backend = select_esa_auto_backend(
                    x,
                    workload="inference",
                    training=bool(self.training or torch.is_grad_enabled()),
                    compile_mode=getattr(self, "_mlbricks_compile_mode", None),
                    native_available=bool(_native_available()),
                    native_cuda_available=bool(_native_cuda_available()),
                )
            except Exception:
                # Auto must remain safe: an unavailable planner never blocks
                # the historical native-first/fallback execution path.
                effective_backend = "auto"

        # Experimental v4 projection orchestration path. It keeps the exact
        # public API and exact trained weights, but moves QGV -> hierarchical
        # ESA -> out_proj orchestration behind one C++ extension call. This is
        # opt-in with MLBRICKS_NATIVE_PROJECTIONS=1 until target-GPU profiling
        # confirms it is beneficial.
        try:
            from ..native import projection_fused_enabled_for as _proj_enabled_for
            from ..native import thunder_forward_hierarchical as _native_full_forward
            if (
                effective_backend != "pytorch"
                and self.compass != "auto"
                and _proj_enabled_for(x)
                and (not self.training or self.dropout.p == 0.0)
            ):
                return _native_full_forward(
                    x,
                    self.qgv.weight,
                    self.out_proj.weight,
                    self.gate_min,
                    self.gate_max,
                    self.eps,
                    self.compass,
                )
        except (ImportError, AttributeError, RuntimeError):
            import os
            if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                raise

        qgv = self.qgv(x)

        # Native CUDA v2 inference fast path. The public ThunderESA API,
        # parameters, checkpoint format, and training path are unchanged.
        # This path consumes the combined Q/G/V projection directly, fusing
        # gate/value transforms + recurrence and then RMS/readout into two
        # CUDA kernels before the existing output projection.
        try:
            from ..native import cached_compass_for_qgv as _cached_compass_for_qgv
            from ..native import should_use_fused_readout as _should_use_fused_readout
            from ..native import thunder_fused_readout as _native_fused_readout
            resolved_compass = self.compass
            if resolved_compass == "auto":
                resolved_compass = _cached_compass_for_qgv(
                    qgv, C, training=self.training
                )
            if (
                effective_backend != "pytorch"
                and resolved_compass is not None
                and _should_use_fused_readout(qgv, C, int(resolved_compass))
            ):
                y = _native_fused_readout(
                    qgv,
                    C,
                    self.gate_min,
                    self.gate_max,
                    self.eps,
                    int(resolved_compass),
                )
                y = self.out_proj(y.to(x.dtype))
                return self.dropout(y)
        except (ImportError, AttributeError, RuntimeError):
            import os
            if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                raise

        q, gate_raw, value_raw = qgv.split(C, dim=-1)

        q = q.reshape(B, T, self.head, self.head_dim)
        gate_raw = gate_raw.reshape(B, T, self.head, self.head_dim)
        value_raw = value_raw.reshape(B, T, self.head, self.head_dim)

        gate = torch.sigmoid(gate_raw)
        A = self.gate_min + (self.gate_max - self.gate_min) * gate

        V = torch.tanh(value_raw)
        B_write = (1.0 - A) * V

        scan_dtype = resolve_scan_dtype(
            self.precision,
            x.device,
            strict_precision=self.strict_precision,
        )

        # Match CF-ESA-c16-FP16Scan when precision="fp16".
        A_scan = A.to(scan_dtype).contiguous()
        B_scan = B_write.to(scan_dtype).contiguous()

        if self.compass == "auto":
            E = _thunder_scan_auto(A_scan, B_scan, backend=effective_backend)
        else:
            E = thunder_scan(
                A_scan,
                B_scan,
                compass=int(self.compass),
                backend=effective_backend,
            )

        E = E.reshape(B, T, C)
        q = q.reshape(B, T, C).to(E.dtype)

        # Benchmark-matching normalization epsilon: 1e-5.
        E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        y = torch.sigmoid(q) * E
        y = y.to(x.dtype)

        y = self.out_proj(y)
        y = self.dropout(y)

        return y