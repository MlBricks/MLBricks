# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Performance-aware routing for ``ESA(..., backend='auto')``.

The public backend contract stays simple:

``native``
    Require the MLBricks native implementation.
``pytorch``
    Require the PyTorch implementation.
``auto``
    Use a conservative performance-qualified route when one is known for the
    active hardware/workload, otherwise preserve the historical native-first
    behavior.

The first qualified profile is NVIDIA SM 7.5 (Tesla T4-class), measured with
three seeds and both GPU placements.  A route is changed away from the
historical native default only when the measured alternative is at least 5%
better.  Smaller differences are treated as noise/hysteresis and keep native.
"""
from __future__ import annotations

import os
from typing import Final

import torch

# Minimum relative throughput advantage required to change the historical
# native-first auto route.  Example: +0.0638 means PyTorch was 6.38% faster.
ESA_AUTO_SWITCH_MARGIN: Final[float] = 0.05

# Qualified T4 / SM 7.5 benchmark deltas: PyTorch throughput relative to native.
# Positive => PyTorch faster. Negative => native faster.
#
# These are deliberately kept local to the exact qualified architecture rather
# than extrapolated to unbenchmarked GPU generations.
_SM75_PYTORCH_GAIN: Final[dict[tuple[str, str], float]] = {
    ("inference", "eager"): -0.7604,
    ("inference", "compile-default"): +0.0638,
    ("inference", "compile-reduce-overhead"): +1.0204,
    ("decode", "eager"): +0.0271,
    ("decode", "compile-default"): +0.2329,
    ("decode", "compile-reduce-overhead"): -0.0446,
}


def _compiler_is_active() -> bool:
    """Best-effort, version-tolerant torch.compile detection."""
    try:
        compiler = getattr(torch, "compiler", None)
        fn = getattr(compiler, "is_compiling", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    try:
        import torch._dynamo
        return bool(torch._dynamo.is_compiling())
    except Exception:
        return False


def _normalize_compile_mode(mode: str | None, *, compiling: bool) -> str:
    if mode is not None:
        value = str(mode).strip().lower().replace("_", "-")
        if value in {"default", "compile-default"}:
            return "compile-default"
        if value in {"reduce-overhead", "compile-reduce-overhead"}:
            return "compile-reduce-overhead"
        if value in {"eager", "none", "off"}:
            return "eager"
    return "compiled-external" if compiling else "eager"


def _sm75_route(workload: str, mode: str, *, margin: float) -> str:
    """Return native/pytorch for the qualified SM 7.5 profile."""
    workload = str(workload).strip().lower()

    if mode == "compiled-external":
        # External torch.compile does not expose its mode to the module.
        # Inference: PyTorch beat native by >5% in *both* tested compile modes.
        if workload == "inference":
            gains = (
                _SM75_PYTORCH_GAIN[("inference", "compile-default")],
                _SM75_PYTORCH_GAIN[("inference", "compile-reduce-overhead")],
            )
            return "pytorch" if min(gains) >= margin else "native"

        # Decode: PyTorch won compile-default by 23.29%, while native won
        # reduce-overhead by only 4.46% (inside the 5% hysteresis band).  Use
        # PyTorch for an unknown external compile mode because the meaningful
        # win is on PyTorch and the known loss remains inside the noise margin.
        if workload == "decode":
            default_gain = _SM75_PYTORCH_GAIN[("decode", "compile-default")]
            reduce_gain = _SM75_PYTORCH_GAIN[("decode", "compile-reduce-overhead")]
            if default_gain >= margin and reduce_gain > -margin:
                return "pytorch"
            return "native"

        return "native"

    gain = _SM75_PYTORCH_GAIN.get((workload, mode))
    if gain is None:
        return "native"
    return "pytorch" if float(gain) >= float(margin) else "native"


def select_esa_auto_backend(
    tensor: torch.Tensor,
    *,
    workload: str,
    training: bool,
    compile_mode: str | None = None,
    native_available: bool,
    native_cuda_available: bool = True,
    switch_margin: float = ESA_AUTO_SWITCH_MARGIN,
) -> str:
    """Select the effective ESA route for ``backend='auto'``.

    Returns ``'native'``, ``'pytorch'`` or ``'auto'``.  ``'auto'`` is used for
    training because the existing ESA training path can use registered native
    scan operators with autograd without turning the whole layer into the
    inference-only explicit native path.
    """
    if bool(training):
        return "auto"

    if os.getenv("MLBRICKS_DISABLE_NATIVE", "0") == "1":
        return "pytorch"
    if not native_available:
        return "pytorch"

    # A compiled CUDA extension being importable does not make the native ESA
    # runtime valid for CPU tensors.  Normal CPU execution stays on PyTorch;
    # the native CPU path is an explicit test/benchmark opt-in that mirrors
    # ``mlbricks.esa.native.enabled_for``.  This prevents a CUDA-capable host
    # from freezing a CPU ESA/decode element to an unusable native route.
    if not tensor.is_cuda:
        return "native" if os.getenv("MLBRICKS_NATIVE_CPU", "0") == "1" else "pytorch"

    if not native_cuda_available:
        return "pytorch"

    compiling = _compiler_is_active()
    mode = _normalize_compile_mode(compile_mode, compiling=compiling)

    try:
        major, minor = torch.cuda.get_device_capability(tensor.device)
    except Exception:
        # Unknown/unqualified CUDA hardware: preserve the old native-first auto.
        return "native"

    if (int(major), int(minor)) == (7, 5):
        return _sm75_route(workload, mode, margin=float(switch_margin))

    # No qualification data for this architecture yet.  Preserve behavior.
    return "native"


__all__ = [
    "ESA_AUTO_SWITCH_MARGIN",
    "select_esa_auto_backend",
]
