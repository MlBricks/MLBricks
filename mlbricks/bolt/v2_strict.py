"""Opt-in BOLT V2 strict decode postprocessing.

This is an *inference-only* alternate implementation of the modern matched-width
AVQ one-token step. Default behavior is unchanged. Enable deliberately with
MLBRICKS_BOLT_V2_STRICT_DECODE=1 and install the optional extension with
MLBRICKS_BUILD_BOLT_V2_STRICT=1. No JIT or Triton dependency.

The native kernel takes the *original torch.nn.Linear projection* of attention
output O; it does not use the separately tested V3 custom GEMV. The CUDA
kernel updates a contiguous FP32 state using separate, correctly rounded
multiply and add operations. Supported shapes: batch 1, one token,
identity return map, AVQ width 192 or 384, CUDA FP16/FP32 attention output.

This path requires qualification on each target GPU/PyTorch version before
being enabled in production. Its numerical behavior is not promised to be
bitwise identical to the PyTorch reference on all GPUs.
"""
from __future__ import annotations

import importlib
import os

import torch
from torch import nn

_EXTENSION = None
_ATTEMPTED = False


def enabled() -> bool:
    return os.environ.get("MLBRICKS_BOLT_V2_STRICT_DECODE", "0").lower() in (
        "1", "true", "yes", "on"
    )


def get_extension():
    global _EXTENSION, _ATTEMPTED
    if not _ATTEMPTED:
        _ATTEMPTED = True
        if torch.cuda.is_available():
            try:
                _EXTENSION = importlib.import_module("mlbricks._bolt_avq_v2_strict")
            except ImportError:
                _EXTENSION = None
    return _EXTENSION


def eligible(avq, o, state):
    # Never accelerate training or change any unsupported state/return format.
    return (
        enabled() and avq.use_native and not avq.training
        and not torch.is_grad_enabled()
        and o.is_cuda and state.is_cuda and o.device == state.device
        and o.dtype in (torch.float16, torch.float32) and state.dtype == torch.float32
        and o.is_contiguous() and state.is_contiguous()
        and o.ndim == 3 and o.shape[:2] == (1, 1)
        and int(o.shape[-1]) in (192, 384)
        and tuple(state.shape) == (1, o.shape[-1])
        and avq.width == o.shape[-1]
        and isinstance(avq.back, nn.Identity)
    )


def maybe_step(avq, o, state, *, inplace_cache=False):
    """Return (output, state) or None when the caller should use PyTorch.

    Standalone step() preserves input state and output by cloning. Cached decode
    may pass inplace_cache=True when both input tensors are temporary/owned by
    that decoder, avoiding a redundant state copy.
    """
    if not eligible(avq, o, state):
        return None
    ext = get_extension()
    if ext is None:
        # Auto mode safely uses the existing PyTorch/current-native AVQ path.
        if avq.backend == "native":
            raise RuntimeError(
                "BOLT V2 strict decode was enabled but mlbricks._bolt_avq_v2_strict "
                "is not installed (set MLBRICKS_BUILD_BOLT_V2_STRICT=1 at build time)"
            )
        return None
    # Preserve the validated projector and its dtype/autocast behavior.
    z = avq.project(o).contiguous()
    if z.dtype != torch.float16 or z.shape != (1, 1, 3 * avq.width):
        # No silent conversion of the original projection's rounding semantics.
        return None
    # Avoid a functional-contract change for BoltStateMemory.step() callers.
    target_o = o if inplace_cache else o.clone()
    target_state = state if inplace_cache else state.clone()
    norm = avq.norm.weight.float().contiguous()
    gamma = avq.gamma.float().reshape(1).contiguous()
    ext.inplace_strict(target_o, z, target_state, norm, gamma, float(avq.norm.eps))
    return target_o, target_state
