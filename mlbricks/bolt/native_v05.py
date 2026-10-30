"""Optional BOLT v0.5 inference-only CUDA U/G and AVQ fusions.

The original trainable Parameters, PyTorch training path and checkpoints stay
unchanged.  For development, set ``MLBRICKS_BOLT_V05_ALLOW_JIT=1`` *before*
launching Python to permit an opt-in CUDA extension build.  Production users
should prebuild ``mlbricks._bolt_v05_cuda`` and leave JIT disabled.

The extension must be loaded before CUDA Graph capture.  Operations write to
persistent caller-owned outputs to avoid graph allocator address changes.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

_EXTENSION = None
_ATTEMPTED = False
_ERROR = None


def get_extension(*, required: bool = False):
    """Return the native extension or raise with an actionable diagnostic."""
    global _EXTENSION, _ATTEMPTED, _ERROR
    if not _ATTEMPTED:
        _ATTEMPTED = True
        if not torch.cuda.is_available():
            _ERROR = "a CUDA GPU is required"
        else:
            try:
                _EXTENSION = importlib.import_module("mlbricks._bolt_v05_cuda")
            except ImportError as exc:
                _ERROR = str(exc)
            if _EXTENSION is None and os.environ.get(
                "MLBRICKS_BOLT_V05_ALLOW_JIT", "0"
            ).strip().lower() in ("1", "yes", "true", "on"):
                try:
                    from torch.utils.cpp_extension import load
                    root = Path(__file__).resolve().parent / "native_v05"
                    _EXTENSION = load(
                        name="mlbricks_bolt_v05h_cuda_dev",
                        sources=[str(root / "bindings.cpp"), str(root / "kernels.cu")],
                        extra_cflags=["-O2"],
                        extra_cuda_cflags=["-O3", "-fmad=false"],
                        with_cuda=True,
                        verbose=False,
                    )
                    _ERROR = None
                except Exception as exc:
                    _ERROR = f"CUDA extension build failed: {exc}"
    if required and _EXTENSION is None:
        raise RuntimeError(
            "BOLT v0.5 native fusions unavailable (" + str(_ERROR) + "). "
            "On Kaggle with nvcc, set MLBRICKS_BOLT_V05_ALLOW_JIT=1 "
            "before importing MLBricks and call model.prepare_bolt_v05_native() "
            "before CUDA Graph capture."
        )
    return _EXTENSION


def ensure_supported(tensor: torch.Tensor):
    if not tensor.is_cuda:
        raise RuntimeError("native BOLT v0.5 operations require CUDA")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native BOLT v0.5 supports FP16, BF16 and FP32")
    return get_extension(required=True)


def gate_out(packed: torch.Tensor, out: torch.Tensor):
    """Write U*(1+tanh(G)) from contiguous packed Q/U/G [B,1,3W]."""
    return ensure_supported(packed).gate_out(packed, out)


def avq_out(raw: torch.Tensor, state: torch.Tensor, features: torch.Tensor, eps: float):
    """Update caller-owned FP32 state and write gated FP32 AVQ features."""
    return ensure_supported(raw).avq_out(raw.contiguous(), state, features, float(eps))


def avq_state_only(raw: torch.Tensor, state: torch.Tensor):
    # Update caller-owned FP32 AVQ state; preserve PyTorch readout.
    return ensure_supported(raw).avq_state_only(raw.contiguous(), state)


def avq_readout_pt(state, r, inv, features):
    return ensure_supported(state).avq_readout_pt(state, r.contiguous(), inv.contiguous(), features)


def avq_readout_raw(state, raw, inv, features):
    return ensure_supported(state).avq_readout_raw(state, raw.contiguous(), inv.contiguous(), features)


__all__ = ["get_extension", "gate_out", "avq_out", "avq_state_only", "avq_readout_pt", "avq_readout_raw", "ensure_supported"]
