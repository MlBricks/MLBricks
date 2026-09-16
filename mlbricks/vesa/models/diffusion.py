# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Matched ESA and bidirectional-SDPA diffusion denoisers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import DiffusionConfig, ESAConfig
from ..layers.attention import AttentionMixer
from ..layers.mixer import ESAMixer
from ..layers.positional import sinusoidal_positions, timestep_embedding
from .common import MLP, AdaLNCondition, modulate


class ESADiffusionBlock(nn.Module):
    def __init__(self, config: DiffusionConfig, *, reverse: bool):
        super().__init__()
        self.reverse = reverse
        self.norm1 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.mixer = ESAMixer(
            ESAConfig(dim=config.dim, heads=config.heads, backend=config.backend, chunk_size=config.chunk_size)
        )
        self.norm2 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.mlp = MLP(config.dim, config.mlp_mult)
        self.condition = AdaLNCondition(config.dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.condition(condition)
        mixed, _ = self.mixer(
            modulate(self.norm1(x), shift1, scale1),
            reverse=self.reverse,
        )
        x = x + gate1.unsqueeze(1) * mixed
        feed_forward = self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x + gate2.unsqueeze(1) * feed_forward


class AttentionDiffusionBlock(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.mixer = AttentionMixer(config.dim, config.heads)
        self.norm2 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.mlp = MLP(config.dim, config.mlp_mult)
        self.condition = AdaLNCondition(config.dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.condition(condition)
        mixed, _, _ = self.mixer(modulate(self.norm1(x), shift1, scale1), causal=False)
        x = x + gate1.unsqueeze(1) * mixed
        feed_forward = self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x + gate2.unsqueeze(1) * feed_forward


class ESADiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig | None = None):
        super().__init__()
        config = config or DiffusionConfig()
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.SiLU(),
            nn.Linear(config.dim * 4, config.dim),
        )
        self.blocks = nn.ModuleList(
            ESADiffusionBlock(
                config,
                reverse=config.alternating_reverse and index % 2 == 1,
            )
            for index in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.dim)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, tokens, dim]")
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"x final dimension must be {self.config.dim}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        positions = sinusoidal_positions(
            x.shape[1],
            self.config.dim,
            device=x.device,
            dtype=x.dtype,
        )
        hidden = x + positions.unsqueeze(0)
        condition = self.time_mlp(timestep_embedding(timesteps, self.config.dim).to(x.dtype))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(self.final_norm(hidden))

    @torch.no_grad()
    def benchmark_sample_loop(self, noise: Tensor, denoising_steps: int) -> Tensor:
        """Simple deterministic loop used only for throughput comparisons."""
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        x = noise
        for step in range(denoising_steps, 0, -1):
            timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x = x - self(x, timesteps) / denoising_steps
        return x


class AttentionDiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig | None = None):
        super().__init__()
        config = config or DiffusionConfig()
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.SiLU(),
            nn.Linear(config.dim * 4, config.dim),
        )
        self.blocks = nn.ModuleList(
            AttentionDiffusionBlock(config) for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.dim)

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, tokens, dim]")
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"x final dimension must be {self.config.dim}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        positions = sinusoidal_positions(
            x.shape[1],
            self.config.dim,
            device=x.device,
            dtype=x.dtype,
        )
        hidden = x + positions.unsqueeze(0)
        condition = self.time_mlp(timestep_embedding(timesteps, self.config.dim).to(x.dtype))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(self.final_norm(hidden))

    @torch.no_grad()
    def benchmark_sample_loop(self, noise: Tensor, denoising_steps: int) -> Tensor:
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        x = noise
        for step in range(denoising_steps, 0, -1):
            timesteps = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x = x - self(x, timesteps) / denoising_steps
        return x
