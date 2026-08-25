# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Generic denoising-loop helper for execution benchmarks."""

from __future__ import annotations

import torch
from torch import Tensor


def benchmark_denoise(model, noise: Tensor, steps: int) -> Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = noise
    for step in range(steps, 0, -1):
        timestep = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        x = x - model(x, timestep) / steps
    return x
