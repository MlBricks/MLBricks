"""Lazy native half-life Quality ESA CUDA decode (inference only).

The source comes from the validated legacy fused half-life kernel used in the
v0.3.7.1 TinyStories notebook. No approximate exp2 path is enabled by default.
"""
from __future__ import annotations

import os
from pathlib import Path
import torch

_EXTENSION = None


def prepare_quality_decode():
    """Compile/load fused decode once, before graph capture or benchmarking.

    Requires CUDA PyTorch, a compatible CUDA toolkit/nvcc, and C++ build tools.
    The original PyTorch/Thunder quality path works without these tools.
    """
    global _EXTENSION
    if _EXTENSION is None:
        if not torch.cuda.is_available():
            raise RuntimeError('Quality fused CUDA decoding requires a CUDA-enabled PyTorch build')
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent / 'cuda'
        paths = [root / 'decode_bindings.cpp', root / 'decode.cu']
        if not all(p.is_file() for p in paths):
            raise RuntimeError('Native quality decode sources missing from installed package')
        os.environ.setdefault('MAX_JOBS', '2')
        _EXTENSION = load(
            name='mlbricks_esa_decode_v039',
            sources=[str(p) for p in paths],
            extra_cflags=['-O2'],
            extra_cuda_cflags=['-O3'],
            verbose=os.getenv('ESA_CUDA_VERBOSE', os.getenv('FASTESA_CUDA_VERBOSE', '0')) == '1',
        )
    return _EXTENSION


def decode_quality(qgv, x, log_half_life, raw_decay_bias, norm_weight, state, heads):
    """Return (readout[B,1,D], new_state[B,H,D/H]) with native CUDA fusion."""
    if torch.is_grad_enabled():
        raise RuntimeError('Fused Quality ESA decode is inference-only; use torch.no_grad()')
    if not qgv.is_cuda or not x.is_cuda:
        raise ValueError('Fused Quality ESA decode requires CUDA')
    init = (state.float().contiguous() if state is not None
            else torch.empty(0, device=x.device, dtype=torch.float32))
    with torch.cuda.device(x.device):
        return prepare_quality_decode().forward(
            qgv.contiguous(), x, log_half_life.contiguous(),
            raw_decay_bias.contiguous(), norm_weight.contiguous(), init,
            int(heads))


# Canonical public entry points.
prepare_decode = prepare_quality_decode
decode = decode_quality
