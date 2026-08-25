from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Simple dtype-preserving RMS normalization."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x.float() * scale).to(x.dtype) * self.weight
