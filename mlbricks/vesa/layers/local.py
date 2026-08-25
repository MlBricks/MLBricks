# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Depthwise local spatial mixing for flattened image tokens."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class LocalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(self, tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        batch, count, dim = tokens.shape
        height, width = grid
        if count != height * width:
            raise ValueError("token count must equal grid height times width")
        image = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        image = self.conv(image)
        return image.flatten(2).transpose(1, 2)
