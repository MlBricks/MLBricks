"""Optional ElasticBit 0.2 native 4-32 bit CUDA runtime.

This module intentionally does not fail package import when CUDA/nvcc support
was not built. MLBricks can still use its PyTorch ElasticLinear compatibility
path and the execution planner can select it for ``backend='auto'``.
"""
from __future__ import annotations

try:
    from . import _C  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - build/environment dependent
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def available() -> bool:
    return _C is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


class _UnavailableRuntime:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "ElasticBit native 4-32 bit CUDA runtime is unavailable. Build MLBricks "
            "on Linux with an NVIDIA CUDA toolkit/nvcc, or use the PyTorch "
            "ElasticLinear compatibility runtime."
        ) from _IMPORT_ERROR

    @classmethod
    def from_auto(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        return cls(*args, **kwargs)


if _C is not None:
    RuntimeMatrix = _C.RuntimeMatrix
    NativeFP16Matrix = _C.NativeFP16Matrix
    bitsAnaliser = _C.bitsAnaliser
else:
    RuntimeMatrix = _UnavailableRuntime
    NativeFP16Matrix = _UnavailableRuntime

    def bitsAnaliser(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "ElasticBit.bitsAnaliser requires the ElasticBit native CUDA runtime."
        ) from _IMPORT_ERROR


__all__ = [
    "RuntimeMatrix", "NativeFP16Matrix", "bitsAnaliser", "available", "import_error"
]
