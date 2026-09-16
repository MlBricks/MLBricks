# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Internal native ESA operators.

This module is deliberately not part of the public MLBricks API. Existing ESA
constructors, checkpoints, and methods stay unchanged. When the compiled
extension is available, eligible CUDA inference and recurrent-training paths
can use native C++/CUDA operators.
"""
from __future__ import annotations

import os
import statistics

import torch

from .planner import AUTO_COMPASS_CANDIDATES, EXECUTION_PLANNER, ceil_div

try:
    from .. import _C  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - depends on local compiler/CUDA
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

_EXTENSION_MODULE = _C

# Loading the extension above registers the C++ TORCH_LIBRARY schemas. Attach
# FakeTensor/autograd formulas, then route runtime calls through torch.ops so
# torch.compile/AOTAutograd sees stable dispatcher operators instead of opaque
# pybind functions. The pybind module remains available for compatibility and
# for the non-Tensor ``has_cuda`` capability query.
if _C is not None:
    try:
        from ..custom_ops import register_native_custom_ops
        _CUSTOM_OPS_REGISTERED = bool(register_native_custom_ops())
    except Exception as exc:  # pragma: no cover - extension/torch-version specific
        _CUSTOM_OPS_REGISTERED = False
        _CUSTOM_OPS_ERROR = exc
    else:
        _CUSTOM_OPS_ERROR = None
else:
    _CUSTOM_OPS_REGISTERED = False
    _CUSTOM_OPS_ERROR = _IMPORT_ERROR

_OPS = torch.ops.mlbricks_native if _CUSTOM_OPS_REGISTERED else _C


def _ops_backend():
    # Keep source-only unit tests and legacy monkeypatching functional while
    # routing the real built extension through torch.ops. If a test/tool has
    # deliberately replaced ``_C`` after import, honor that replacement.
    if _C is not _EXTENSION_MODULE:
        return _C
    if _CUSTOM_OPS_REGISTERED:
        return torch.ops.mlbricks_native
    return _C


def custom_ops_registered() -> bool:
    """Whether native kernels are exposed through ``torch.ops``."""
    return bool(_CUSTOM_OPS_REGISTERED)


def custom_ops_error() -> Exception | None:
    return _CUSTOM_OPS_ERROR


def available() -> bool:
    return _C is not None


def cuda_available() -> bool:
    return bool(_C is not None and _C.has_cuda())


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def training_enabled_for(tensor: torch.Tensor) -> bool:
    """Return whether the native Thunder scan is safe for autograd training.

    Native ESA training is deliberately narrower than native inference.  The
    recurrent CUDA scan has an explicit registered autograd formula backed by
    ``thunder_scan_backward_chunked``.  Fused readout/full-forward kernels stay
    inference-only and are guarded separately by :func:`fused_enabled_for`.
    """
    if os.getenv("MLBRICKS_DISABLE_NATIVE", "0") == "1":
        return False
    if _C is None or not bool(getattr(tensor, "is_cuda", False)):
        return False
    try:
        if not bool(_C.has_cuda()):
            return False
    except Exception:
        return False

    # Explicit native training uses the dispatcher autograd registration. This
    # also keeps torch.compile/AOTAutograd on stable torch.ops operators rather
    # than exposing a raw pybind call with no autograd contract.
    return bool(_CUSTOM_OPS_REGISTERED)


def enabled_for(tensor: torch.Tensor) -> bool:
    if os.getenv("MLBRICKS_DISABLE_NATIVE", "0") == "1":
        return False
    if _C is None:
        return False

    # Native CUDA Thunder now supports recurrent training/backward.  CPU native
    # remains a correctness/benchmark opt-in only.
    if torch.is_grad_enabled():
        return training_enabled_for(tensor)
    if tensor.is_cuda:
        return True
    return os.getenv("MLBRICKS_NATIVE_CPU", "0") == "1"


def fused_enabled_for(tensor: torch.Tensor) -> bool:
    """Return True when the fused Thunder *inference-only* path may be used."""
    if os.getenv("MLBRICKS_DISABLE_FUSED", "0") == "1":
        return False
    # Do not let enabling native scan training expose fused kernels that do not
    # have the recurrent training autograd contract.
    if torch.is_grad_enabled():
        return False
    if not enabled_for(tensor):
        return False
    # v2 fused kernels are CUDA inference kernels and are optimized for the
    # FP16/BF16 path used by Thunder. Float32 keeps the existing implementation.
    return bool(tensor.is_cuda and tensor.dtype in (torch.float16, torch.bfloat16))


def should_use_fused_readout(qgv: torch.Tensor, embd: int, compass: int) -> bool:
    """Choose the fused Thunder prefill readout without regressing short prompts.

    ``MLBRICKS_FUSED_READOUT`` may be ``auto`` (default), ``always`` or
    ``never``.  The 0.1.1 T4 sweep showed that B1/T256 paid more launch/setup
    cost in the fused readout than the decomposed native scan.  In auto mode we
    therefore use a workload threshold instead of forcing fusion everywhere.
    The decision is bucket-cached and remains GPU-model agnostic.
    """
    if not fused_enabled_for(qgv):
        return False
    policy = os.getenv("MLBRICKS_FUSED_READOUT", "auto").strip().lower()
    if policy not in {"auto", "always", "never"}:
        raise ValueError("MLBRICKS_FUSED_READOUT must be auto, always, or never")
    if policy == "always":
        return True
    if policy == "never":
        return False

    batch, time, _ = qgv.shape
    key = (
        "fused_readout",
        qgv.device.index,
        str(qgv.dtype),
        int(batch),
        # Reuse the same route across nearby prompt lengths.
        int(1 << max(6, (int(time) - 1).bit_length())),
        int(embd),
        int(compass),
    )
    cached = EXECUTION_PLANNER.route_cache.get(key)
    if cached is not None:
        return bool(cached)

    # Short single-batch prefill is launch-bound; the decomposed native path was
    # ~10% faster on T4 at T=256.  Larger batch*sequence products amortize the
    # fused kernel and benefit from fewer intermediates/global-memory passes.
    selected = bool(int(batch) * int(time) >= 1024)
    EXECUTION_PLANNER.route_cache[key] = selected
    return selected


def projection_fused_enabled_for(tensor: torch.Tensor) -> bool:
    """Return True for the experimental C++-orchestrated projection path.

    This path preserves the exact Linear -> hierarchical ESA -> Linear math.
    It is opt-in until it has been benchmarked on each target GPU.
    """
    if os.getenv("MLBRICKS_NATIVE_PROJECTIONS", "0") != "1":
        return False
    return fused_enabled_for(tensor)


def _fused_mode(qgv: torch.Tensor, compass: int) -> str:
    """Select the native fused Thunder scan strategy.

    Modes are entirely native CUDA:
      - direct: one recurrent worker per (batch, channel), best for short T
      - hierarchical: chunk-parallel time scan, best for long T / low occupancy
      - auto: shape-based dispatch between the two
    """
    mode = os.getenv("MLBRICKS_FUSED_MODE", "auto").strip().lower()
    if mode not in {"auto", "direct", "hierarchical"}:
        raise ValueError(
            "MLBRICKS_FUSED_MODE must be one of: auto, direct, hierarchical"
        )
    if mode != "auto":
        return mode

    batch, time, qgv_width = qgv.shape
    channels = qgv_width // 3
    chunks = (time + int(compass) - 1) // int(compass)

    # Direct v2 is exceptional at short sequences because it uses only two
    # kernels. Hierarchical v3 is selected when direct recurrence becomes
    # serial-depth limited. Keep all dispatch inside native CUDA; there is no
    # PyTorch fallback here. The threshold is intentionally conservative until
    # the T4 sweep is collected.
    if chunks <= 1024 and time >= 2048 and batch * channels < 4096:
        return "hierarchical"
    return "direct"


def thunder_fused_readout(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """Fused Thunder transform + scan + normalized gated readout."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")

    qgv = qgv.contiguous()
    mode = _fused_mode(qgv, compass)
    if mode == "hierarchical":
        accum = os.getenv("MLBRICKS_SCAN_ACCUM", "v3").strip().lower()
        if accum not in {"v3", "mixed32", "full32"}:
            raise ValueError(
                "MLBRICKS_SCAN_ACCUM must be one of: v3, mixed32, full32"
            )
        if accum == "mixed32":
            return _ops_backend().thunder_fused_readout_hierarchical_mixed32(
                qgv, int(embd), float(gate_min), float(gate_max),
                float(eps), int(compass),
            )
        if accum == "full32":
            return _ops_backend().thunder_fused_readout_hierarchical_full32(
                qgv, int(embd), float(gate_min), float(gate_max),
                float(eps), int(compass),
            )
        return _ops_backend().thunder_fused_readout_hierarchical(
            qgv,
            int(embd),
            float(gate_min),
            float(gate_max),
            float(eps),
            int(compass),
        )

    return _ops_backend().thunder_fused_readout(
        qgv,
        int(embd),
        float(gate_min),
        float(gate_max),
        float(eps),
        int(compass),
    )


def thunder_forward_hierarchical(
    x: torch.Tensor,
    qgv_weight: torch.Tensor,
    out_weight: torch.Tensor,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """Run QGV projection + hierarchical ESA + output projection in C++.

    The GEMMs still use PyTorch/ATen's optimized CUDA BLAS backend; this entry
    point removes Python-side orchestration and is a stepping stone toward a
    cached cuBLASLt projection backend if profiling shows the GEMMs dominate.
    """
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    x = x.contiguous()
    qgv_weight = qgv_weight.contiguous()
    out_weight = out_weight.contiguous()

    gemm_accum = os.getenv("MLBRICKS_GEMM_ACCUM", "fp32").strip().lower()
    if gemm_accum not in {"fp32", "fp16", "mixed", "hybrid"}:
        raise ValueError(
            "MLBRICKS_GEMM_ACCUM must be one of: fp32, fp16, mixed, hybrid"
        )
    if gemm_accum in {"fp16", "mixed", "hybrid"} and x.dtype != torch.float16:
        raise RuntimeError(f"MLBRICKS_GEMM_ACCUM={gemm_accum} requires FP16 tensors")
    if gemm_accum == "fp16":
        return _ops_backend().thunder_forward_hierarchical_fp16gemm(
            x, qgv_weight, out_weight,
            float(gate_min), float(gate_max), float(eps), int(compass),
        )
    if gemm_accum == "mixed":
        return _ops_backend().thunder_forward_hierarchical_mixedgemm(
            x, qgv_weight, out_weight,
            float(gate_min), float(gate_max), float(eps), int(compass),
        )

    if gemm_accum == "hybrid":
        chunk = int(os.getenv("MLBRICKS_HYBRID_CHUNK", "32"))
        return _ops_backend().thunder_forward_hierarchical_hybridgemm(
            x, qgv_weight, out_weight,
            float(gate_min), float(gate_max), float(eps), int(compass),
            int(chunk),
        )

    return _ops_backend().thunder_forward_hierarchical(
        x, qgv_weight, out_weight,
        float(gate_min), float(gate_max), float(eps), int(compass),
    )


def thunder_fused_readout_hierarchical_mixed32(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """v8 mixed precision recurrence: FP32 chunk summaries/global prefix."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_fused_readout_hierarchical_mixed32(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max),
        float(eps), int(compass),
    )


def thunder_fused_readout_hierarchical_full32(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """v8 full precision recurrence: FP32 state/transition/prefix accumulation."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_fused_readout_hierarchical_full32(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max),
        float(eps), int(compass),
    )




def thunder_fused_readout_hierarchical_precise_gate(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """v10 fused path with PyTorch-faithful gate/value elementwise math."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_fused_readout_hierarchical_precise_gate(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max),
        float(eps), int(compass),
    )


def thunder_fused_readout_hierarchical_precise_readout(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """v10 fused path with PyTorch-faithful RMS/readout math."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_fused_readout_hierarchical_precise_readout(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max),
        float(eps), int(compass),
    )


def thunder_fused_readout_hierarchical_precise_both(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    """v10 fused path with precise gate/value and readout math."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_fused_readout_hierarchical_precise_both(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max),
        float(eps), int(compass),
    )


def thunder_prepare_ab_precise(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v10 debug operator for PyTorch-faithful gate/value preparation."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    A, B_write = _ops_backend().thunder_prepare_ab_precise(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max)
    )
    return A, B_write


def thunder_readout_precise(
    q: torch.Tensor,
    states: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """v10 debug operator for PyTorch-faithful RMS/sigmoid readout."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_readout_precise(
        q.contiguous(), states.contiguous(), float(eps)
    )


def thunder_prepare_ab(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v9 debug operator exposing native gate/value -> A/B preparation."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    A, B_write = _ops_backend().thunder_prepare_ab(
        qgv.contiguous(), int(embd), float(gate_min), float(gate_max)
    )
    return A, B_write


def thunder_readout(
    q: torch.Tensor,
    states: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """v9 debug operator exposing native RMS + sigmoid(q) gated readout."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().thunder_readout(
        q.contiguous(), states.contiguous(), float(eps)
    )


def _planner_supported(A: torch.Tensor, B_write: torch.Tensor) -> bool:
    return bool(
        _C is not None
        and A.is_cuda
        and B_write.is_cuda
        and A.dtype in (torch.float16, torch.bfloat16)
        and B_write.dtype == A.dtype
        and A.shape == B_write.shape
        and A.dim() == 4
    )


def _normalize_compass(compass: int | str) -> int | str:
    if isinstance(compass, str):
        value = compass.strip().lower()
        if value != "auto":
            raise ValueError("compass must be a positive integer or 'auto'")
        return "auto"
    value = int(compass)
    if value <= 0:
        raise ValueError(f"compass must be positive, got {value}")
    return value


def _benchmark_group_size(
    chunk_A: torch.Tensor,
    chunk_B: torch.Tensor,
    group_size: int,
) -> float:
    assert _C is not None
    group_size = int(group_size)
    for _ in range(2):
        pref_A, pref_B, parent_A, parent_B = _ops_backend().thunder_summary_scan(
            chunk_A, chunk_B, group_size
        )
        global_parent_A, global_parent_B = _ops_backend().thunder_group_prefix(
            parent_A, parent_B
        )
        _ops_backend().thunder_apply_group(
            pref_A, pref_B, global_parent_A, global_parent_B, group_size
        )
    torch.cuda.synchronize(chunk_A.device)

    samples: list[float] = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            pref_A, pref_B, parent_A, parent_B = _ops_backend().thunder_summary_scan(
                chunk_A, chunk_B, group_size
            )
            global_parent_A, global_parent_B = _ops_backend().thunder_group_prefix(
                parent_A, parent_B
            )
            _ops_backend().thunder_apply_group(
                pref_A, pref_B, global_parent_A, global_parent_B, group_size
            )
        end.record()
        torch.cuda.synchronize(chunk_A.device)
        samples.append(float(start.elapsed_time(end)) / 5.0)
    return float(statistics.median(samples))


def _choose_group_size(
    chunk_A: torch.Tensor,
    chunk_B: torch.Tensor,
    direct_budget: int,
) -> int:
    key = EXECUTION_PLANNER.group_key(chunk_A, direct_budget)
    cached = EXECUTION_PLANNER.group_cache.get(key)
    if cached is not None:
        return int(cached)

    candidates = EXECUTION_PLANNER.group_candidates(
        int(chunk_A.shape[1]), int(direct_budget), chunk_A.device
    )
    results: list[tuple[int, float, int]] = []
    for group_size in candidates:
        ms = _benchmark_group_size(chunk_A, chunk_B, int(group_size))
        parent_count = ceil_div(int(chunk_A.shape[1]), int(group_size))
        results.append((int(group_size), float(ms), int(parent_count)))

    selected = min(results, key=lambda item: item[1])[0]
    EXECUTION_PLANNER.group_cache[key] = int(selected)
    EXECUTION_PLANNER.group_benchmarks[key] = results
    return int(selected)


def _resolve_summary_prefix(
    summary_A: torch.Tensor,
    summary_B: torch.Tensor,
    direct_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    assert _C is not None
    summary_count = int(summary_A.shape[1])
    if summary_count <= int(direct_budget):
        out_A, out_B = _ops_backend().thunder_group_prefix(summary_A, summary_B)
        return out_A, out_B, ()

    group_size = _choose_group_size(summary_A, summary_B, direct_budget)
    pref_A, pref_B, parent_A, parent_B = _ops_backend().thunder_summary_scan(
        summary_A, summary_B, int(group_size)
    )
    global_parent_A, global_parent_B, parent_levels = _resolve_summary_prefix(
        parent_A, parent_B, direct_budget
    )
    out_A, out_B = _ops_backend().thunder_apply_group(
        pref_A,
        pref_B,
        global_parent_A,
        global_parent_B,
        int(group_size),
    )
    return out_A, out_B, (int(group_size),) + parent_levels


def _fixed_planned_forward(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int,
    direct_budget: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert _C is not None
    chunks = ceil_div(int(A.shape[1]), int(compass))
    if chunks <= int(direct_budget):
        return _ops_backend().thunder_scan_hierarchical(A, B_write, int(compass)), ()

    local_states, chunk_A, chunk_B = _ops_backend().thunder_scan_local(
        A, B_write, int(compass)
    )
    _, global_chunk_B, levels = _resolve_summary_prefix(
        chunk_A, chunk_B, int(direct_budget)
    )
    states = _ops_backend().thunder_apply_chunk_prefix(
        A, local_states, global_chunk_B, int(compass)
    )
    return states, levels


class _PlannedThunderScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_write: torch.Tensor,
        compass: int,
        direct_budget: int,
    ) -> torch.Tensor:
        states, _ = _fixed_planned_forward(
            A.contiguous(), B_write.contiguous(), int(compass), int(direct_budget)
        )
        ctx.compass = int(compass)
        ctx.save_for_backward(A, states)
        return states

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        assert _C is not None
        A, states = ctx.saved_tensors

        # The chunk-parallel reverse recurrence is valid regardless of whether
        # the forward scan used the direct or recursive summary path. Using it
        # for every training shape avoids materializing reverse_A, reverse_B,
        # and a second full sequence of reverse states for hierarchical cases.
        # This is both lower-memory and faster on long-context CUDA workloads.
        grad_A, grad_B = _ops_backend().thunder_scan_backward_chunked(
            A.contiguous(),
            states.contiguous(),
            grad_out.contiguous(),
            int(ctx.compass),
        )
        return grad_A, grad_B, None, None


def _run_fixed_planned_scan(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int,
) -> torch.Tensor:
    channels = int(A.shape[2] * A.shape[3])
    direct_budget = EXECUTION_PLANNER.direct_summary_budget(
        A.device, int(A.shape[0]), channels
    )
    chunks = ceil_div(int(A.shape[1]), int(compass))

    if torch.is_grad_enabled() and (A.requires_grad or B_write.requires_grad):
        # The overwhelmingly common case (including T=256/512/1024 training)
        # fits within the direct summary budget.  When torch.library custom ops
        # are registered, call the dispatcher op directly: its autograd formula
        # is registered in custom_ops.py and AOTAutograd can capture both the
        # forward scan and native chunked backward without a pybind graph break.
        if _CUSTOM_OPS_REGISTERED and chunks <= int(direct_budget):
            return _ops_backend().thunder_scan_hierarchical(
                A.contiguous(), B_write.contiguous(), int(compass)
            )

        # Very long contexts may recurse through multiple summary levels. Keep
        # the established custom Function for that composite planner path; each
        # native primitive inside it is nevertheless a registered torch op.
        return _PlannedThunderScanFn.apply(
            A.contiguous(), B_write.contiguous(), int(compass), int(direct_budget)
        )
    states, _ = _fixed_planned_forward(
        A.contiguous(), B_write.contiguous(), int(compass), int(direct_budget)
    )
    return states


def _benchmark_compass(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int,
    *,
    training: bool,
) -> float:
    compass = int(compass)
    if training:
        probe_A = A.detach().requires_grad_(True)
        probe_B = B_write.detach().requires_grad_(True)
        grad = torch.ones_like(A)
        for _ in range(1):
            y = _run_fixed_planned_scan(probe_A, probe_B, compass)
            torch.autograd.grad(y, (probe_A, probe_B), grad_outputs=grad)
        torch.cuda.synchronize(A.device)
        samples: list[float] = []
        for _ in range(2):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y = _run_fixed_planned_scan(probe_A, probe_B, compass)
            torch.autograd.grad(y, (probe_A, probe_B), grad_outputs=grad)
            end.record()
            torch.cuda.synchronize(A.device)
            samples.append(float(start.elapsed_time(end)))
        return float(statistics.median(samples))

    with torch.no_grad():
        for _ in range(2):
            _run_fixed_planned_scan(A, B_write, compass)
        torch.cuda.synchronize(A.device)
        samples = []
        for _ in range(3):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_fixed_planned_scan(A, B_write, compass)
            end.record()
            torch.cuda.synchronize(A.device)
            samples.append(float(start.elapsed_time(end)))
        return float(statistics.median(samples))


def cached_compass_for_qgv(
    qgv: torch.Tensor,
    embd: int,
    *,
    training: bool = False,
) -> int | None:
    """Return an already tuned Compass for a QGV workload bucket, if available."""
    if not qgv.is_cuda or not torch.cuda.is_available():
        return None
    key = EXECUTION_PLANNER.compass_key_values(
        qgv.device,
        batch=int(qgv.shape[0]),
        time=int(qgv.shape[1]),
        channels=int(embd),
        dtype=qgv.dtype,
        training=bool(training),
    )
    value = EXECUTION_PLANNER.compass_cache.get(key)
    return None if value is None else int(value)


def choose_compass(
    A: torch.Tensor,
    B_write: torch.Tensor,
) -> int:
    """Select and cache the fastest supported compass for this workload/device."""
    if not _planner_supported(A, B_write):
        return 16

    training = bool(
        torch.is_grad_enabled() and (A.requires_grad or B_write.requires_grad)
    )
    key = EXECUTION_PLANNER.compass_key(A, training=training)
    cached = EXECUTION_PLANNER.compass_cache.get(key)
    if cached is not None:
        return int(cached)

    results: list[tuple[int, float]] = []
    for candidate in AUTO_COMPASS_CANDIDATES:
        ms = _benchmark_compass(
            A, B_write, int(candidate), training=training
        )
        results.append((int(candidate), float(ms)))

    selected = min(results, key=lambda item: item[1])[0]
    EXECUTION_PLANNER.compass_cache[key] = int(selected)
    EXECUTION_PLANNER.compass_benchmarks[key] = results
    return int(selected)


def thunder_scan_planned(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int | str = "auto",
) -> torch.Tensor:
    """Run the resource-aware CUDA scan with automatic hierarchy/group planning."""
    normalized = _normalize_compass(compass)
    if not _planner_supported(A, B_write):
        if normalized == "auto":
            normalized = 16
        return thunder_scan(A, B_write, int(normalized), _allow_planner=False)

    resolved = choose_compass(A, B_write) if normalized == "auto" else int(normalized)
    return _run_fixed_planned_scan(A, B_write, resolved)


def clear_planner_cache() -> None:
    """Clear cached auto-compass and hierarchy microbenchmark decisions."""
    EXECUTION_PLANNER.clear()


def thunder_scan(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int | str,
    *,
    _allow_planner: bool = True,
) -> torch.Tensor:
    """Run a native scan when eligible, preserving CPU/direct compatibility."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")

    normalized = _normalize_compass(compass)
    if _allow_planner and _planner_supported(A, B_write):
        needs_backward = bool(
            torch.is_grad_enabled() and (A.requires_grad or B_write.requires_grad)
        )
        # Training no longer needs the old MLBRICKS_NATIVE_TRAINING opt-in.
        # Route differentiable CUDA scans through the planned autograd-capable
        # implementation automatically; inference keeps the same planner path.
        if normalized == "auto" or needs_backward or not torch.is_grad_enabled():
            return thunder_scan_planned(A, B_write, normalized)

    if normalized == "auto":
        normalized = 16
    return _ops_backend().thunder_scan(A.contiguous(), B_write.contiguous(), int(normalized))


def thunder_scan_hierarchical(
    A: torch.Tensor,
    B_write: torch.Tensor,
    compass: int,
) -> torch.Tensor:
    """Run the original one-level hierarchical CUDA scan (<=1024 summaries)."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    if not A.is_cuda or not B_write.is_cuda:
        raise RuntimeError("hierarchical Thunder scan requires CUDA tensors")
    return _ops_backend().thunder_scan_hierarchical(
        A.contiguous(),
        B_write.contiguous(),
        int(compass),
    )

def lightning_step(
    A: torch.Tensor,
    B_write: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    dtype = state.dtype
    A = A.to(dtype=dtype).contiguous()
    B_write = B_write.to(dtype=dtype).contiguous()
    state = state.contiguous()
    return _ops_backend().lightning_step(A, B_write, state)


def ffn_gelu_residual(
    normalized: torch.Tensor,
    residual: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Native inference orchestration for Linear -> GELU -> Linear -> residual."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    return _ops_backend().ffn_gelu_residual(
        normalized.contiguous(), residual.contiguous(),
        w1.contiguous(), b1.contiguous(), w2.contiguous(), b2.contiguous()
    )


def elastic_linear_packed(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Direct packed ElasticBit CUDA linear without materializing the full weight."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    if not x.is_cuda:
        raise RuntimeError("elastic_linear_packed requires CUDA")
    return _ops_backend().elastic_linear_packed(
        x.contiguous(), packed.contiguous(), scales.contiguous(), bias.contiguous(),
        int(bits), int(group_size), int(out_features), int(in_features)
    )


def residual_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + LayerNorm for inference.

    CUDA uses one native kernel; CPU uses the matching reference operator.
    Training intentionally stays on PyTorch so autograd remains fully explicit.
    """
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    residual, normalized = _ops_backend().residual_layer_norm(
        x.contiguous(), update.contiguous(), weight.contiguous(), bias.contiguous(), float(eps)
    )
    return residual, normalized


def lightning_fused_step(
    qgv: torch.Tensor,
    state: torch.Tensor,
    gate_min: float,
    gate_max: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused CUDA one-token gate/value transform, recurrence and readout."""
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    readout, new_state = _ops_backend().lightning_fused_step(
        qgv.contiguous(),
        state.contiguous(),
        float(gate_min),
        float(gate_max),
        float(eps),
    )
    return readout, new_state


def linear_fp16_accum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Experimental FP16-accumulation cuBLAS projection.

    This is intentionally internal/benchmark-only until numerical tolerance is
    validated on trained ESA checkpoints.
    """
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise RuntimeError("linear_fp16_accum requires FP16 tensors")
    return _ops_backend().linear_fp16_accum(x.contiguous(), weight.contiguous())


def linear_fp32_accum(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Native cuBLAS projection with FP32 accumulation and FP16 output.

    Internal/benchmark operator for the v6 mixed projection experiment.
    """
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise RuntimeError("linear_fp32_accum requires FP16 tensors")
    return _ops_backend().linear_fp32_accum(x.contiguous(), weight.contiguous())


def linear_hybrid_accum(
    x: torch.Tensor,
    weight: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Hybrid output GEMM: FP16 chunk partials, FP32 cross-chunk reduction.

    This is an internal v7.1 experiment. ``chunk`` is the preferred K dimension
    for each FP16-accumulation partial GEMM. Non-divisible K uses one smaller
    final tail GEMM before the FP32 cross-chunk reduction.
    """
    if _C is None:
        raise RuntimeError(f"MLBricks native extension is unavailable: {_IMPORT_ERROR}")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise RuntimeError("linear_hybrid_accum requires FP16 tensors")
    if int(chunk) <= 0:
        raise ValueError("chunk must be positive")
    return _ops_backend().linear_hybrid_accum(
        x.contiguous(), weight.contiguous(), int(chunk)
    )
