# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Matched PyTorch SDPA attention mixer used as the baseline."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionMixer(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim <= 0 or heads <= 0:
            raise ValueError("dim and heads must be positive")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = self._split(query)
        key = self._split(key)
        value = self._split(value)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        output = output.transpose(1, 2).contiguous().reshape(x.shape[0], x.shape[1], self.dim)
        return self.out_proj(output), key, value

    def step(
        self,
        x: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        position: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError("step input must have shape [batch, dim]")
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        batch = x.shape[0]
        query = query.reshape(batch, self.heads, 1, self.head_dim)
        key = key.reshape(batch, self.heads, 1, self.head_dim)
        value = value.reshape(batch, self.heads, 1, self.head_dim)
        key_cache[:, :, position : position + 1, :] = key
        value_cache[:, :, position : position + 1, :] = value
        output = F.scaled_dot_product_attention(
            query,
            key_cache[:, :, : position + 1, :],
            value_cache[:, :, : position + 1, :],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.out_proj(output.reshape(batch, self.dim)), key_cache, value_cache
