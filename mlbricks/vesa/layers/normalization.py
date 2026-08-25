# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""PerspectiveNorm: independent normalization over channel groups."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...runtime import normalize_backend


class PerspectiveNorm(nn.Module):
    def __init__(self, dim: int, groups: int = 4, *, backend: str = "auto"):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if groups <= 0:
            raise ValueError("groups must be positive")
        if dim % groups:
            raise ValueError("dim must be divisible by groups")
        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups
        self.backend = normalize_backend(backend, warn_legacy=True)
        # Keep historical state_dict layout/checkpoint compatibility.
        self.norms = nn.ModuleList(nn.LayerNorm(self.group_dim) for _ in range(groups))

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = normalize_backend(backend, warn_legacy=True)
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        try:
            from ...vision_native import available
            if available():
                return "native"
        except Exception:
            pass
        return "native-required" if self.backend == "native" else "pytorch"

    def forward(self, x: Tensor) -> Tensor:
        # Stack the historical per-group affine parameters into [G, GD].  The
        # stack itself is a native ATen op; the compiled C++ operator owns the
        # grouped normalization math when available.
        weight = torch.stack([norm.weight for norm in self.norms], dim=0)
        bias = torch.stack([norm.bias for norm in self.norms], dim=0)
        if self.backend != "pytorch":
            try:
                from ...vision_native import perspective_norm
                native = perspective_norm(
                    x, weight, bias, groups=self.groups,
                    eps=float(self.norms[0].eps), backend=self.backend,
                )
            except RuntimeError:
                if self.backend == "native":
                    raise
                native = None
            if native is not None:
                return native

        # Vectorized fallback: no Python split/cat loop in the tensor path.
        shape = (*x.shape[:-1], self.groups, self.group_dim)
        y = x.reshape(shape)
        yf = y.float()
        mean = yf.mean(dim=-1, keepdim=True)
        var = (yf - mean).square().mean(dim=-1, keepdim=True)
        y = ((yf - mean) * torch.rsqrt(var + self.norms[0].eps)).to(x.dtype)
        y = y * weight + bias
        return y.reshape_as(x)
