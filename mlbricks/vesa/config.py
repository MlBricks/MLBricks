# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Validated configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..runtime import normalize_backend
from ..vision_common import canonical_engine, canonical_position, canonical_scan

FullBackend = Literal["auto", "native", "pytorch"]
Backend = Literal["auto", "native", "pytorch"]

FULL_BACKENDS = {"auto", "native", "pytorch"}
BACKENDS = FULL_BACKENDS


def _canonical_backend(value: str) -> str:
    return normalize_backend(value, warn_legacy=True)



@dataclass(frozen=True)
class ESAConfig:
    dim: int = 192
    backend: Backend = "auto"
    chunk_size: int = 64
    gate_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        object.__setattr__(self, "backend", _canonical_backend(self.backend))


@dataclass(frozen=True)
class AutoregressiveConfig:
    vocab_size: int = 8192
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_mult: int = 4
    prefill_backend: FullBackend = "auto"
    chunk_size: int = 64
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("dim, depth, and heads must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.mlp_mult <= 0 or self.chunk_size <= 0:
            raise ValueError("mlp_mult and chunk_size must be positive")
        object.__setattr__(self, "prefill_backend", _canonical_backend(self.prefill_backend))


@dataclass(frozen=True)
class DiffusionConfig:
    dim: int = 192
    depth: int = 6
    heads: int = 6
    mlp_mult: int = 4
    backend: FullBackend = "auto"
    chunk_size: int = 64
    alternating_reverse: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.depth <= 0 or self.heads <= 0:
            raise ValueError("dim, depth, and heads must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.mlp_mult <= 0 or self.chunk_size <= 0:
            raise ValueError("mlp_mult and chunk_size must be positive")
        object.__setattr__(self, "backend", _canonical_backend(self.backend))


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 192
    depth: int = 6
    mlp_mult: int = 4
    backend: FullBackend = "auto"
    chunk_size: int = 64
    perspective_groups: int = 4
    local_kernel_size: int = 3
    serpentine: bool = True  # legacy compatibility flag; engine supersedes it
    alternating_reverse: bool = False

    # Unified image-engine policy.
    engine: str = "Serpentine"
    position: str | None = "auto"
    scan: str = "cross"

    # Fields used by AR/Bolt-compatible visual engines. They are harmless for
    # the classic VESA classifier and keep one compact configuration surface.
    vocab_size: int = 8192
    heads: int = 6
    latent_dim: int = 32
    tie_embeddings: bool = True

    # Selectable MLBricks components.
    ffn: str = "standard"
    residual: str = "standard"
    ffn_state_dim: int = 256
    ffn_depth_embedding_dim: int = 64
    ffn_virtual_refinements: int = 2
    ffn_virtual_hidden_dim: int = 128
    ffn_micro_hidden_dim: int = 64
    ffn_micro_refinements: int = 1
    residual_update_ratio: float = 0.18
    residual_stream_ratio: float = 1.08
    residual_update_softness: float = 8.0
    residual_stream_softness: float = 8.0

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.in_channels <= 0 or self.num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")
        if self.dim <= 0 or self.depth <= 0 or self.mlp_mult <= 0:
            raise ValueError("dim, depth, and mlp_mult must be positive")
        object.__setattr__(self, "backend", _canonical_backend(self.backend))
        engine = canonical_engine(self.engine)
        scan = canonical_scan(self.scan)
        position = canonical_position(self.position, engine=engine)
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "scan", scan)
        object.__setattr__(self, "position", position)
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if self.heads <= 0 or self.latent_dim <= 0:
            raise ValueError("heads and latent_dim must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.perspective_groups <= 0:
            raise ValueError("perspective_groups must be positive")
        if self.dim % self.perspective_groups:
            raise ValueError("dim must be divisible by perspective_groups")
        if self.local_kernel_size <= 0:
            raise ValueError("local_kernel_size must be positive")
        if self.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd")

        ffn_name = str(self.ffn).strip().lower().replace("-", "_")
        ffn_aliases = {
            "standard": "standard",
            "default": "standard",
            "ffn": "standard",
            "ffnbrick": "ffnbrick",
            "state_aware": "ffnbrick",
            "stateaware": "ffnbrick",
            "virtual_ffnbrick": "virtual_ffnbrick",
            "virtual_state_aware": "virtual_ffnbrick",
            "virtualstateaware": "virtual_ffnbrick",
            "micro_ffnbrick": "micro_ffnbrick",
            "micro_virtual": "micro_ffnbrick",
            "microvirtual": "micro_ffnbrick",
        }
        if ffn_name not in ffn_aliases:
            raise ValueError(
                "ffn must be one of: standard, ffnbrick, virtual_ffnbrick, "
                "micro_ffnbrick."
            )
        object.__setattr__(self, "ffn", ffn_aliases[ffn_name])

        residual_name = str(self.residual).strip().lower().replace("-", "_")
        residual_aliases = {
            "standard": "standard",
            "default": "standard",
            "residual": "standard",
            "rescontroller": "rescontroller",
            "res_controller": "rescontroller",
            "controller": "rescontroller",
        }
        if residual_name not in residual_aliases:
            raise ValueError("residual must be one of: standard, rescontroller.")
        object.__setattr__(self, "residual", residual_aliases[residual_name])

        positive_ints = {
            "ffn_state_dim": self.ffn_state_dim,
            "ffn_depth_embedding_dim": self.ffn_depth_embedding_dim,
            "ffn_virtual_refinements": self.ffn_virtual_refinements,
            "ffn_virtual_hidden_dim": self.ffn_virtual_hidden_dim,
            "ffn_micro_hidden_dim": self.ffn_micro_hidden_dim,
            "ffn_micro_refinements": self.ffn_micro_refinements,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "residual_update_ratio",
            "residual_stream_ratio",
            "residual_update_softness",
            "residual_stream_softness",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
