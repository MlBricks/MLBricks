"""Positional components for MLBricks model builders."""
from __future__ import annotations

import math
import torch
from torch import nn


class LearnedPosition(nn.Module):
    """Learned additive positional embedding."""

    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        if dim <= 0 or max_seq_len <= 0:
            raise ValueError("dim and max_seq_len must be positive")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.embedding = nn.Embedding(self.max_seq_len, self.dim)

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [B,T,{self.dim}] input")
        end = int(start_pos) + x.size(1)
        if start_pos < 0 or end > self.max_seq_len:
            raise ValueError("position range exceeds max_seq_len")
        pos = torch.arange(start_pos, end, device=x.device)
        return x + self.embedding(pos).to(dtype=x.dtype).unsqueeze(0)


class SinusoidalPosition(nn.Module):
    """Deterministic additive sinusoidal positional encoding."""

    def __init__(self, dim: int, max_seq_len: int = 65536, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or max_seq_len <= 0 or base <= 0:
            raise ValueError("dim, max_seq_len, and base must be positive")
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)[:, None]
        even = torch.arange(0, self.dim, 2, dtype=torch.float32)
        inv = torch.exp(-math.log(self.base) * even / max(self.dim, 1))
        phase = positions * inv[None, :]
        table = torch.zeros(self.max_seq_len, self.dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(phase)
        if self.dim > 1:
            table[:, 1::2] = torch.cos(phase[:, : table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"expected [B,T,{self.dim}] input")
        end = int(start_pos) + x.size(1)
        if start_pos < 0 or end > self.max_seq_len:
            raise ValueError("position range exceeds max_seq_len")
        return x + self.table[start_pos:end].to(device=x.device, dtype=x.dtype).unsqueeze(0)


class RoPE(nn.Module):
    """Rotary position transform for Q/K-like tensors ``[B,H,T,D]``."""

    def __init__(self, dim: int | None = None, *, base: float = 10000.0) -> None:
        super().__init__()
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive")
        if base <= 0:
            raise ValueError("base must be positive")
        self.dim = None if dim is None else int(dim)
        self.base = float(base)

    def forward(self, x: torch.Tensor, *, start_pos: int = 0) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("RoPE expects [B,H,T,D]")
        width = x.size(-1) if self.dim is None else min(self.dim, x.size(-1))
        width -= width % 2
        if width <= 0:
            return x
        dtype = x.dtype
        xf = x[..., :width].float()
        positions = torch.arange(
            start_pos, start_pos + x.size(-2), device=x.device, dtype=torch.float32
        )
        inv = self.base ** (
            -torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width
        )
        phase = positions[:, None] * inv[None, :]
        cos = torch.cos(phase)[None, None, :, :]
        sin = torch.sin(phase)[None, None, :, :]
        even = xf[..., 0::2]
        odd = xf[..., 1::2]
        rotated = torch.empty_like(xf)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        if width == x.size(-1):
            return rotated.to(dtype)
        return torch.cat([rotated.to(dtype), x[..., width:]], dim=-1)


def make_position(position, *, dim: int, max_seq_len: int):
    """Resolve a positional spec to ``(additive_position, rotary_position)``."""
    if position is None:
        return None, None
    if isinstance(position, str):
        name = position.strip().lower().replace("-", "_")
        if name in {"none", "off", "disabled"}:
            return None, None
        if name in {"learned", "absolute", "learned_absolute"}:
            return LearnedPosition(dim, max_seq_len), None
        if name in {"sin", "sine", "sinusoidal"}:
            return SinusoidalPosition(dim, max_seq_len), None
        if name in {"rope", "rotary"}:
            return None, RoPE()
        raise ValueError("position must be none, learned, sinusoidal, rope, or an nn.Module")
    if isinstance(position, RoPE):
        return None, position
    if isinstance(position, nn.Module):
        return position, None
    raise TypeError("position must be a string, nn.Module, or None")
