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

    def __post_init__(self) -> None:
        if self.embd < 1 or self.head < 1 or self.embd % self.head:
            raise ValueError("embd must be positive and divisible by head")
        if normalize_variant(self.variant) == "current" and self.head != 6:
            raise ValueError("Current half-life ESA currently requires head=6")
        _check_backend(self.backend)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if normalize_variant(self.variant) == "base" and self.chunk_size != 32:
            raise ValueError("Historical Base ESA uses chunk_size=32")

    def build(self, *, device=None):
        from .public_api import ESA
        return ESA(embd=self.embd, head=self.head, variant=self.variant,
                   backend=self.backend, chunk_size=self.chunk_size,
                   device=device) if normalize_variant(self.variant) == "current" else ESA(
                   embd=self.embd, head=self.head, variant=self.variant,
                   backend=self.backend, device=device)
