"""Lazy fused CUDA BaseESA one-token decode (inference only).

Shares the Half-Life decoder's launch architecture, while retaining the
learned-bounds BaseESA recurrence exactly. No changes to Fast-EsaQ's six
original CUDA/C++ source files. Training and prefill use the old scan path.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXTENSION = None


def prepare_base_decode():
    """Compile once before measurement or CUDA Graph capture (not in graph)."""
    global _EXTENSION
    if _EXTENSION is None:
        if not torch.cuda.is_available():
            raise RuntimeError("BaseESA fused decode requires CUDA")
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent / "cuda"
        sources = [root / "base_decode_bindings.cpp", root / "base_decode.cu"]
        if not all(p.is_file() for p in sources):
            raise RuntimeError("BaseESA native decode source files are missing")
        os.environ.setdefault("MAX_JOBS", "2")
        _EXTENSION = load(
            name="mlbricks_esa_base_decode_v1",
            sources=[str(p) for p in sources],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O3"],
            verbose=os.getenv("ESA_CUDA_VERBOSE", "0") == "1",
        )
    return _EXTENSION


def decode_base(qgv, x, raw_alpha, raw_beta, state, heads):
    """Return (read[B,1,D], state[B,H,D/H]); one native kernel after qgv."""
    if torch.is_grad_enabled():
        raise RuntimeError("Fused BaseESA decode is inference-only")
    if not qgv.is_cuda or not x.is_cuda:
        raise ValueError("Fused BaseESA decode requires CUDA")
    previous = (state.float().contiguous() if state is not None else
                torch.empty(0, device=x.device, dtype=torch.float32))
    with torch.cuda.device(x.device):
        return prepare_base_decode().forward(
            qgv.contiguous(), x, raw_alpha.float().contiguous(),
            raw_beta.float().contiguous(), previous, int(heads))
