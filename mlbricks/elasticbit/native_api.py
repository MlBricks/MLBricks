"""Thin loader for the optional ElasticBit CUDA runtime.

ElasticBit's public API lives in :mod:`mlbricks.elasticbit.core`.  This module
only owns extension discovery and intentionally exposes no compatibility aliases.
"""
from __future__ import annotations

import importlib

try:
    _C = importlib.import_module(f"{__package__}._C")
except Exception as exc:  # pragma: no cover - build/environment dependent
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def available() -> bool:
    return _C is not None


def importError() -> Exception | None:
    return _IMPORT_ERROR


def requireNative():
    if _C is None:
        raise RuntimeError(
            "ElasticBit requires its CUDA runtime. Build MLBricks with "
            "MLBRICKS_BUILD_ELASTICBIT_NATIVE=1 on a CUDA-enabled system."
        ) from _IMPORT_ERROR
    return _C


__all__ = ["available", "importError", "requireNative"]
