# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared feed-forward and adaptive-normalization components."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        if dim <= 0 or mult <= 0:
            raise ValueError("dim and mult must be positive")
        hidden = dim * mult
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class AdaLNCondition(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, condition: Tensor) -> tuple[Tensor, ...]:
        return self.net(condition).chunk(6, dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
