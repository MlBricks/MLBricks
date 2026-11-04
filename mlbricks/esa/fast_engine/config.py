"""Serializable configuration for the MLBricks ESA public API."""
from __future__ import annotations

from dataclasses import dataclass
from .core import normalize_variant
from .public_api import _check_backend


@dataclass(frozen=True)
class ESAConfig:
    embd: int = 384
    head: int = 6
    variant: str = "current"
    backend: str = "auto"
    chunk_size: int = 32
    cumpass: int = 16
    parallel_chunk: int = 128
    cuda_warps: int = 8

    def __post_init__(self) -> None:
        if self.embd < 1 or self.head < 1 or self.embd % self.head:
            raise ValueError("embd must be positive and divisible by head")
        _check_backend(self.backend)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not isinstance(self.cumpass, int) or self.cumpass < 1:
            raise ValueError("cumpass must be a positive integer")
        if self.parallel_chunk not in {16, 32, 64, 128, 256, 512}:
            raise ValueError("parallel_chunk must be 16, 32, 64, 128, 256 or 512")
        if self.cuda_warps not in {4, 8, 16}:
            raise ValueError("cuda_warps must be 4, 8, or 16")
        if normalize_variant(self.variant) == "base" and self.chunk_size != 32:
            raise ValueError("Historical Base ESA uses chunk_size=32")

    def build(self, *, device=None):
        from .public_api import ESA
        return ESA(embd=self.embd, head=self.head, variant=self.variant,
                   backend=self.backend, chunk_size=self.chunk_size,
                   cumpass=self.cumpass, parallel_chunk=self.parallel_chunk,
                   cuda_warps=self.cuda_warps, device=device) if normalize_variant(self.variant) == "current" else ESA(
                   embd=self.embd, head=self.head, variant=self.variant,
                   backend=self.backend, cumpass=self.cumpass,
                   parallel_chunk=self.parallel_chunk, cuda_warps=self.cuda_warps,
                   device=device)
