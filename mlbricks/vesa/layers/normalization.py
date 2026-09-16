# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""PerspectiveNorm: independent normalization over channel groups."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...runtime import normalize_backend
from ...planner import EXECUTION_PLANNER


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
        EXECUTION_PLANNER.clear_owner_routes(self)
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        frozen = EXECUTION_PLANNER.owner_routes(self)
        routes = sorted(set(frozen.values()))
        if len(routes) == 1:
            return routes[0]
        try:
            from ...vision_native import available
            if available():
                return "planner(auto)"
        except Exception:
            pass
        return "pytorch"

    def _python_forward(self, x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        shape = (*x.shape[:-1], self.groups, self.group_dim)
        y = x.reshape(shape)
        yf = y.float()
        mean = yf.mean(dim=-1, keepdim=True)
        var = (yf - mean).square().mean(dim=-1, keepdim=True)
        y = ((yf - mean) * torch.rsqrt(var + self.norms[0].eps)).to(x.dtype)
        y = y * weight + bias
        return y.reshape_as(x)

    def forward(self, x: Tensor) -> Tensor:
        # Stack the historical per-group affine parameters into [G, GD].  The
        # stack itself is a native ATen op; the compiled C++ operator owns the
        # grouped normalization math when available.
        weight = torch.stack([norm.weight for norm in self.norms], dim=0)
        bias = torch.stack([norm.bias for norm in self.norms], dim=0)
        if self.backend != "pytorch":
            frozen = None
            perspective_norm = None
            native_ok = False
            if self.backend == "auto":
                frozen = EXECUTION_PLANNER.owner_routes(self).get(
                    ("perspective_norm", bool(torch.is_grad_enabled()))
                )
                if frozen == "pytorch":
                    return self._python_forward(x, weight, bias)
                if frozen == "native":
                    native_ok = True
            if frozen != "native":
                try:
                    from ...vision_native import perspective_norm, available, cuda_built
                    native_ok = bool(available())
                    if x.is_cuda:
                        native_ok = native_ok and bool(cuda_built())
                except Exception:
                    native_ok = False
                    perspective_norm = None

            if self.backend == "auto" and frozen is None and native_ok and perspective_norm is not None and not torch.is_grad_enabled():
                route = EXECUTION_PLANNER.qualify_operator_once(
                    self,
                    "perspective_norm",
                    x,
                    {
                        "native": lambda: perspective_norm(
                            x, weight, bias, groups=self.groups,
                            eps=float(self.norms[0].eps), backend="native", owner=self,
                        ),
                        "pytorch": lambda: self._python_forward(x, weight, bias),
                    },
                    requested_backend="auto",
                    native_available=True,
                    native_supports_training=True,
                    training=False,
                    extra=(int(self.groups), int(self.group_dim)),
                    default_auto="native",
                )
                if route == "pytorch":
                    return self._python_forward(x, weight, bias)

            try:
                if perspective_norm is None and frozen == "native":
                    from ...vision_native import perspective_norm
                if perspective_norm is not None:
                    native = perspective_norm(
                        x, weight, bias, groups=self.groups,
                        eps=float(self.norms[0].eps), backend=self.backend, owner=self,
                    )
                else:
                    native = None
            except RuntimeError:
                if self.backend == "native":
                    raise
                native = None
            if native is not None:
                return native

        return self._python_forward(x, weight, bias)
