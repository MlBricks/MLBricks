"""Weight-independent affine scans for FAST (E[t] = A[t]*E[t-1] + U[t]).

Thunder is a self-contained pure-PyTorch adaptation of the MLBricks ESA
hierarchical scan; SWQR is optional native FP32 CUDA with an experimental
reverse-scan autograd implementation. No JIT compilation on package import.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from contextlib import nullcontext
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F


_SUPPORTED_CHUNKS = {16, 32, 64, 128, 256, 512}
_EXTENSIONS: dict[str, object] = {}
_FAILED_EXTENSIONS: set[tuple[str, str]] = set()


def _validate(A: Tensor, U: Tensor, state: Tensor | None) -> None:
    if A.ndim != 4 or A.shape != U.shape:
        raise ValueError("A and U must have matching [batch, seq, heads, head_dim] shapes")
    if A.size(1) < 1:
        raise ValueError("Sequence must have at least one token")
    if A.dtype != torch.float32 or U.dtype != torch.float32:
        raise TypeError("Scan requires FP32 A and U")
    if state is not None and (state.shape != A.shape[:1] + A.shape[2:] or
                              state.device != A.device):
        raise ValueError("Cache must have shape [batch, heads, head_dim] and match device")


def direct_scan(A: Tensor, U: Tensor, state: Tensor | None = None) -> Tensor:
    """Single-step direct recurrence; does not launch a parallel scan."""
    _validate(A, U, state)
    if A.shape[1] != 1:
        raise ValueError("direct_scan only accepts one token")
    return U if state is None else A * state.float().unsqueeze(1) + U


def thunder_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *, compass: int = 16) -> Tensor:
    """Differentiable hierarchical Thunder scan, no ESA installation required.

    Input cache is incorporated into the initial write and never detached,
    preserving gradients through chunk boundaries. Equivalent in real arithmetic.
    """
    _validate(A, U, state)
    if not isinstance(compass, int) or compass < 1:
        raise ValueError("compass must be a positive integer")
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if state is not None:
        U = torch.cat((U[:, :1] + A[:, :1] * state.float().unsqueeze(1), U[:, 1:]), dim=1)
    B, T, H, Dh = A.shape
    pad = (-T) % compass
    if pad:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        U = F.pad(U, (0, 0, 0, 0, 0, pad), value=0.0)
    groups = A.shape[1] // compass
    As = A.reshape(B, groups, compass, H, Dh)
    Us = U.reshape(B, groups, compass, H, Dh)
    st = U.new_zeros(B, groups, H, Dh)
    trans = A.new_ones(B, groups, H, Dh)
    local = []
    prefix = []
    for t in range(compass):
        st = As[:, :, t] * st + Us[:, :, t]
        trans = As[:, :, t] * trans
        local.append(st)
        prefix.append(trans)
    local = torch.stack(local, dim=2)
    prefix = torch.stack(prefix, dim=2)
    cp = prefix[:, :, -1]
    cs = local[:, :, -1]
    shift = 1
    while shift < groups:
        prev_p, prev_s = cp, cs
        cp = torch.cat((prev_p[:, :shift], prev_p[:, shift:] * prev_p[:, :-shift]), dim=1)
        cs = torch.cat((prev_s[:, :shift], prev_p[:, shift:] * prev_s[:, :-shift] + prev_s[:, shift:]), dim=1)
        shift *= 2
    carry = torch.cat((cs.new_zeros(B, 1, H, Dh), cs[:, :-1]), dim=1)
    return (prefix * carry.unsqueeze(2) + local).reshape(B, -1, H, Dh)[:, :T]


def _extension():
    """Lazy native build on the current CUDA device; PyTorch cache reused."""
    if not torch.cuda.is_available():
        raise RuntimeError("SWQR native FastScan needs a CUDA-enabled PyTorch installation")
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / "cuda"
    if not all((root / file).is_file() for file in ("scan_bindings.cpp", "scan.cu")):
        raise RuntimeError("SWQR CUDA source files are missing from the installation")
    name = "mlbricks_esa_scan_v039"
    if name not in _EXTENSIONS:
        os.environ.setdefault("MAX_JOBS", "2")
        # Standard torch extension cache avoids writing into read-only site-packages.
        _EXTENSIONS[name] = load(
            name=name,
            sources=[str(root / "scan_bindings.cpp"), str(root / "scan.cu")],
            extra_cflags=["-O2"], extra_cuda_cflags=["-O3"],
            verbose=os.getenv("ESA_CUDA_VERBOSE", os.getenv("FASTESA_CUDA_VERBOSE", "0")) == "1",
        )
    return _EXTENSIONS[name]


class _SWQR(Function):
    @staticmethod
    def forward(ctx, A: Tensor, U: Tensor, state: Tensor, chunk: int):
        with (torch.cuda.device(A.device) if A.is_cuda else nullcontext()):
            E = _extension().scan(A.contiguous(), U.contiguous(), state.contiguous(), chunk)
        ctx.save_for_backward(A, E, state)
        ctx.chunk = chunk
        return E

    @staticmethod
    def backward(ctx, gradE: Tensor):
        A, E, state = ctx.saved_tensors
        # Reverse recurrence: r_t = gradE_t + A_{t+1} * r_{t+1}.
        Ar = torch.cat((torch.zeros_like(A[:, :1]), A[:, 1:].flip(1)), dim=1).contiguous()
        Ur = gradE.float().flip(1).contiguous()
        empty = A.new_empty(0)
        with (torch.cuda.device(A.device) if A.is_cuda else nullcontext()):
            r = _extension().scan(Ar, Ur, empty, ctx.chunk).flip(1)
        initial = state.unsqueeze(1) if state.numel() else torch.zeros_like(E[:, :1])
        previous = torch.cat((initial, E[:, :-1]), dim=1)
        dA = r * previous
        dU = r
        dState = (A[:, 0] * r[:, 0]).contiguous() if state.numel() else None
        return dA, dU, dState, None


def swqr_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *, chunk: int = 128) -> Tensor:
    """Native FP32 SWQR scan; CUDA required for sequences longer than one token."""
    _validate(A, U, state)
    if chunk not in _SUPPORTED_CHUNKS:
        raise ValueError(f"SWQR chunk must be one of {sorted(_SUPPORTED_CHUNKS)}")
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if not A.is_cuda:
        raise RuntimeError("scan='swqr' requires CUDA; use scan='auto' for portable fallback")
    initial = state.float().contiguous() if state is not None else A.new_empty(0)
    return _SWQR.apply(A.contiguous(), U.contiguous(), initial, chunk)


def select_swqr_backend(batch: int, seq: int, device: torch.device | str,
                        *, device_name: str | None = None,
                        capability: tuple[int, int] | None = None) -> tuple[str, int | None]:
    """Pure shape/device dispatch policy, no kernels launched or model weights changed.

    Returns (backend, warps) where backend is direct, thunder, swqr, or
    swqr_short. Current short preferences are based ONLY on Tesla T4 / sm75
    benchmark measurements (not assumed transferable to other GPUs).
    Supply device_name/capability for offline policy tests.
    """
    if not isinstance(batch, int) or batch < 1 or not isinstance(seq, int) or seq < 1:
        raise ValueError("batch and sequence lengths must be positive integers")
    if seq == 1:
        return "direct", None
    target = torch.device(device)
    if target.type != "cuda":
        return "thunder", None
    if device_name is None or capability is None:
        try:
            props = torch.cuda.get_device_properties(target)
        except (RuntimeError, AssertionError):
            # Unknown CUDA device: prefer the already-validated long SWQR path.
            return "swqr", None
        if device_name is None:
            device_name = props.name
        if capability is None:
            capability = (props.major, props.minor)
    is_t4 = (capability == (7, 5) and
             device_name.strip().lower() in {"tesla t4", "nvidia t4"})
    if is_t4:
        if batch == 1 and 128 <= seq <= 512:
            return "swqr_short", 16
        if batch == 2 and 128 <= seq <= 256:
            return "swqr_short", 8
    return "swqr", None


def _scan_selected(A: Tensor, U: Tensor, state: Tensor | None,
                   backend: str, warps: int | None, chunk: int) -> Tensor:
    if backend == "swqr_short":
        assert warps is not None
        return swqr_short_scan(A, U, state, warps=warps)
    if backend == "swqr":
        return swqr_scan(A, U, state, chunk=chunk)
    if backend == "direct":
        return direct_scan(A, U, state)
    raise RuntimeError(f"Unsupported native scan backend {backend!r}")


def auto_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *,
              compass: int = 16, chunk: int = 128) -> Tensor:
    """T4-aware fastEsa v0.3.2 dispatcher; safe Thunder fallback.

    Failed short-extension compilation falls back to original SWQR first.
    If SWQR is also unavailable, falls back to PyTorch Thunder. Explicit
    scan='swqr_adaptive' and 'swqr_short' never silently use another backend.
    """
    _validate(A, U, state)
    if A.shape[1] == 1:
        return direct_scan(A, U, state)
    if not A.is_cuda or (os.getenv("ESA_DISABLE_NATIVE", "0") == "1" or os.getenv("FASTESA_DISABLE_SWQR", "0") == "1"):
        return thunder_scan(A, U, state, compass=compass)
    backend, warps = select_swqr_backend(A.shape[0], A.shape[1], A.device)
    key = (str(A.device), backend)
    if backend == "swqr_short" and key not in _FAILED_EXTENSIONS:
        try:
            return swqr_short_scan(A, U, state, warps=warps)
        except (RuntimeError, OSError, ImportError) as exc:
            _FAILED_EXTENSIONS.add(key)
            warnings.warn(f"SWQR-Short unavailable ({type(exc).__name__}: {exc}); "
                          "falling back to original SWQR", RuntimeWarning, stacklevel=2)
    key = (str(A.device), "swqr")
    if key not in _FAILED_EXTENSIONS:
        try:
            return swqr_scan(A, U, state, chunk=chunk)
        except (RuntimeError, OSError, ImportError) as exc:
            _FAILED_EXTENSIONS.add(key)
            warnings.warn(f"SWQR native unavailable ({type(exc).__name__}: {exc}); "
                          "falling back to PyTorch Thunder", RuntimeWarning, stacklevel=2)
    return thunder_scan(A, U, state, compass=compass)


def prepare_swqr() -> None:
    """Precompile native SWQR (optional, e.g. before timing CUDA benchmarks)."""
    _extension()


def _short_extension():
    """Lazy load of the independent SWQR-Short native CUDA extension."""
    if not torch.cuda.is_available():
        raise RuntimeError("SWQR-Short requires CUDA")
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent / "cuda"
    sources = [root / "short_scan_bindings.cpp", root / "short_scan.cu"]
    if not all(p.is_file() for p in sources):
        raise RuntimeError("SWQR-Short native CUDA source files missing")
    name = "mlbricks_esa_short_scan_v039"
    if name not in _EXTENSIONS:
        os.environ.setdefault("MAX_JOBS", "2")
        _EXTENSIONS[name] = load(
            name=name, sources=[str(p) for p in sources],
            extra_cflags=["-O2"], extra_cuda_cflags=["-O3"],
            verbose=os.getenv("ESA_CUDA_VERBOSE", os.getenv("FASTESA_CUDA_VERBOSE", "0")) == "1",
        )
    return _EXTENSIONS[name]


class _SWQRShort(Function):
    @staticmethod
    def forward(ctx, A: Tensor, U: Tensor, state: Tensor, warps: int):
        with (torch.cuda.device(A.device) if A.is_cuda else nullcontext()):
            E = _short_extension().scan(A.contiguous(), U.contiguous(), state.contiguous(), warps)
        ctx.save_for_backward(A, E, state)
        ctx.warps = warps
        return E

    @staticmethod
    def backward(ctx, gradE: Tensor):
        A, E, state = ctx.saved_tensors
        # Reverse-time affine recurrence. Exact gradient of E[t]=A[t]E[t-1]+U[t].
        Ar = torch.cat((torch.zeros_like(A[:, :1]), A[:, 1:].flip(1)), dim=1).contiguous()
        Ur = gradE.float().flip(1).contiguous()
        with (torch.cuda.device(A.device) if A.is_cuda else nullcontext()):
            r = _short_extension().scan(Ar, Ur, A.new_empty(0), ctx.warps).flip(1)
        previous = torch.cat((state.unsqueeze(1) if state.numel() else
                              torch.zeros_like(E[:, :1]), E[:, :-1]), dim=1)
        dA = r * previous
        dU = r
        dState = (A[:, 0] * r[:, 0]).contiguous() if state.numel() else None
        return dA, dU, dState, None


def swqr_short_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *,
                    warps: int = 8) -> Tensor:
    """Experimental single-launch FP32 CUDA scan for 2..2048 tokens.

    Deliberately opt-in: the existing SWQR implementation remains unchanged.
    No silent fallback: a missing compiler or a native failure raises.
    """
    _validate(A, U, state)
    if warps not in (4, 8, 16):
        raise ValueError("warps must be 4, 8 or 16")
    if A.size(1) == 1:
        return direct_scan(A, U, state)
    if A.size(1) > 2048:
        raise ValueError("SWQR-Short supports at most 2048 tokens; use 'swqr_adaptive'")
    if not A.is_cuda:
        raise RuntimeError("scan='swqr_short' requires CUDA")
    initial = state.float().contiguous() if state is not None else A.new_empty(0)
    return _SWQRShort.apply(A.contiguous(), U.contiguous(), initial, warps)


def swqr_adaptive_scan(A: Tensor, U: Tensor, state: Tensor | None = None, *,
                       warps: int = 8, chunk: int = 128,
                       cutoff: int | None = None) -> Tensor:
    """Strict native batch-aware dispatch, without any hidden fallback.

    No cutoff (default): T4 batch/length policy, original SWQR elsewhere.
    Explicit cutoff: backwards-compatible v0.3.1 fixed-length policy using
    the caller-supplied warps. This also serves as a manual policy override.
    """
    _validate(A, U, state)
    if warps not in (4, 8, 16):
        raise ValueError("warps must be 4, 8, or 16")
    if chunk not in _SUPPORTED_CHUNKS:
        raise ValueError("unsupported SWQR chunk size")
    if cutoff is not None and (not isinstance(cutoff, int) or not 2 <= cutoff <= 2048):
        raise ValueError("cutoff must be in [2,2048]")
    if A.size(1) == 1:
        return direct_scan(A, U, state)
    if not A.is_cuda:
        raise RuntimeError("scan='swqr_adaptive' requires CUDA; use 'auto' on CPU")
    if cutoff is not None:
        backend, selected_warps = (("swqr_short", warps) if A.size(1) <= cutoff
                                   else ("swqr", None))
    else:
        backend, selected_warps = select_swqr_backend(A.size(0), A.size(1), A.device)
    return _scan_selected(A, U, state, backend, selected_warps, chunk)


def prepare_swqr_short() -> None:
    """Optionally precompile the SWQR-Short extension before GPU benchmarking."""
    _short_extension()


# Clean execution-engine names for the MLBricks ESA integration.
sequential_scan = direct_scan
hierarchical_scan = thunder_scan
parallel_scan = swqr_scan
short_scan = swqr_short_scan
adaptive_scan = swqr_adaptive_scan
scan = auto_scan
prepare_scan = prepare_swqr
prepare_short_scan = prepare_swqr_short

def select_scan_backend(*args, **kwargs):
    """Select a human-readable execution backend, preserving prior policy."""
    mode, warps = select_swqr_backend(*args, **kwargs)
    return {"direct":"sequential", "thunder":"hierarchical",
            "swqr":"parallel", "swqr_short":"short"}[mode], warps
