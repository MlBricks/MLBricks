# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Sinusoidal sequence-position and diffusion-timestep embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sinusoidal_positions(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if length < 0 or dim <= 0:
        raise ValueError("length must be non-negative and dim must be positive")
    half = dim // 2
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    ).unsqueeze(0)
    angles = positions * frequencies
    encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if encoding.shape[1] < dim:
        encoding = F.pad(encoding, (0, dim - encoding.shape[1]))
    return encoding.to(dtype=dtype)


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=1)
    if embedding.shape[1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding
