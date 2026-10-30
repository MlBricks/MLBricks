"""Small CUDA-first placement helper used by public BOLT/QKV constructors.

Never modifies PyTorch global defaults; explicit .to(...) still works afterwards.
"""
from __future__ import annotations

import os
import torch


def initial_device():
    """Automatically choose CUDA only when enabled and available."""
    if os.environ.get('MLBRICKS_AUTO_CUDA', '1').strip().lower() in {'0', 'false', 'off', 'no'}:
        return None
    return torch.device('cuda') if torch.cuda.is_available() else None


def place_new_module(module):
    target = initial_device()
    if target is not None:
        module.to(device=target)
    return module


def input_on_module(x, module):
    """Align a tensor with the actual (possibly user-overridden) module device."""
    if not isinstance(x, torch.Tensor):
        return x
    p = next(module.parameters(), None)
    if p is None or x.device == p.device:
        return x
    return x.to(device=p.device, non_blocking=True)
