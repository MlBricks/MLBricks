"""Compiler-aware dispatch signature for BOLT dynamic-position native CUDA read.

This operator advertises all in-place buffer writes to torch.compile/export's
custom-op system. It is an inference-only primitive and does not implement a
backward. A model-level compiler still needs graph-safe recurrent position and
per-mixer state management to capture an entire hybrid decode step correctly.
"""
from __future__ import annotations

import importlib
import torch


@torch.library.custom_op(
    'mlbricks::bolt_dynamic_decode',
    mutates_args=('cache', 'scratch'),
    device_types='cuda',
)
def bolt_dynamic_decode(
    q: torch.Tensor,
    current: torch.Tensor,
    cache: torch.Tensor,
    scratch: torch.Tensor,
    position: torch.Tensor,
    scale: float,
    chunk_tokens: int,
    max_chunks: int,
    use_split: bool,
) -> torch.Tensor:
    """Read BOLT shared-latent cache with GPU-resident mutable position.

    `cache` and `scratch` are explicitly marked as mutable, unlike pybind-only
    operators which can surprise compiler alias/mutation analysis.
    """
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, current, cache)):
        raise RuntimeError('BOLT dynamic CUDA decode is inference only')
    ext = importlib.import_module('mlbricks._bolt_dynamic_cuda')
    return ext.decode(q, current, cache, scratch, position, float(scale),
                      int(chunk_tokens), int(max_chunks), bool(use_split))


@bolt_dynamic_decode.register_fake
def _fake_bolt_dynamic_decode(q, current, cache, scratch, position,
                              scale, chunk_tokens, max_chunks, use_split):
    return q.new_empty(q.shape)


__all__ = ['bolt_dynamic_decode']
