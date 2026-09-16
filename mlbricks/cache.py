"""Fixed-capacity inference caches used by MLBricks generation.

The buffers are allocated once and updated in-place.  This avoids the O(T)
allocation/copy cost of ``torch.cat`` at every autoregressive token.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class GaussCache:
    c: torch.Tensor
    rho: torch.Tensor
    length: int = 0

    @property
    def capacity(self) -> int:
        return int(self.c.size(2))

    @classmethod
    def allocate(cls, batch: int, heads: int, capacity: int, latent_dim: int, *, device, dtype):
        if batch <= 0 or heads <= 0 or capacity <= 0 or latent_dim <= 0:
            raise ValueError("GaussCache dimensions must be positive")
        c = torch.empty(batch, heads, capacity, latent_dim, device=device, dtype=dtype)
        rho = torch.empty(batch, heads, capacity, device=device, dtype=dtype)
        return cls(c=c, rho=rho, length=0)

    def reset(self) -> "GaussCache":
        self.length = 0
        return self

    def load_prefix(self, c: torch.Tensor, rho: torch.Tensor) -> "GaussCache":
        n = int(c.size(2))
        if c.shape[:2] != self.c.shape[:2] or c.size(3) != self.c.size(3):
            raise ValueError("Gauss cache prefix shape mismatch")
        if rho.shape != c.shape[:3]:
            raise ValueError("rho prefix shape mismatch")
        if n > self.capacity:
            raise ValueError("Gauss cache prefix exceeds capacity")
        self.c[:, :, :n, :].copy_(c)
        self.rho[:, :, :n].copy_(rho)
        self.length = n
        return self

    def append(self, c: torch.Tensor, rho: torch.Tensor, *, position: int | None = None) -> int:
        if c.size(2) != 1 or rho.size(2) != 1:
            raise ValueError("GaussCache.append expects one token")
        pos = self.length if position is None else int(position)
        if pos < 0 or pos >= self.capacity:
            raise ValueError("Gauss cache position exceeds capacity")
        self.c[:, :, pos:pos + 1, :].copy_(c)
        self.rho[:, :, pos:pos + 1].copy_(rho)
        self.length = max(self.length, pos + 1)
        return self.length


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor
    length: int = 0

    @property
    def capacity(self) -> int:
        return int(self.k.size(2))

    @classmethod
    def allocate(cls, batch: int, heads: int, capacity: int, head_dim: int, *, device, dtype):
        if batch <= 0 or heads <= 0 or capacity <= 0 or head_dim <= 0:
            raise ValueError("KVCache dimensions must be positive")
        k = torch.empty(batch, heads, capacity, head_dim, device=device, dtype=dtype)
        v = torch.empty(batch, heads, capacity, head_dim, device=device, dtype=dtype)
        return cls(k=k, v=v, length=0)

    def reset(self) -> "KVCache":
        self.length = 0
        return self

    def load_prefix(self, k: torch.Tensor, v: torch.Tensor) -> "KVCache":
        n = int(k.size(2))
        if k.shape != v.shape:
            raise ValueError("K/V prefix shape mismatch")
        if k.shape[:2] != self.k.shape[:2] or k.size(3) != self.k.size(3):
            raise ValueError("KV cache prefix shape mismatch")
        if n > self.capacity:
            raise ValueError("KV cache prefix exceeds capacity")
        self.k[:, :, :n, :].copy_(k)
        self.v[:, :, :n, :].copy_(v)
        self.length = n
        return self

    def append(self, k: torch.Tensor, v: torch.Tensor, *, position: int | None = None) -> int:
        if k.size(2) != 1 or v.size(2) != 1:
            raise ValueError("KVCache.append expects one token")
        pos = self.length if position is None else int(position)
        if pos < 0 or pos >= self.capacity:
            raise ValueError("KV cache position exceeds capacity")
        self.k[:, :, pos:pos + 1, :].copy_(k)
        self.v[:, :, pos:pos + 1, :].copy_(v)
        self.length = max(self.length, pos + 1)
        return self.length


__all__ = ["GaussCache", "KVCache"]
