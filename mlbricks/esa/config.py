# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..runtime import normalize_backend

Backend = Literal["auto", "native", "pytorch"]
Precision = Literal["fp8", "fp16", "bf16", "fp32", "fp64"]


@dataclass
class ESAConfig:
    embd: int
    head: int = 4
    batch: int | None = None
    block: int | None = None
    dropout: float = 0.0

    # Uniform MLBricks backend policy. Automatic Compass planning is default.
    backend: Backend = "auto"
    compass: int | str | None = "auto"

    precision: Precision = "fp16"
    gate_min: float = 0.8
    gate_max: float = 0.995
    eps: float = 1e-5
    strict_precision: bool = False
    strict_backend: bool = False

    def __post_init__(self) -> None:
        if self.embd <= 0:
            raise ValueError(f"embd must be positive, got {self.embd}.")
        if self.head <= 0:
            raise ValueError(f"head must be positive, got {self.head}.")
        if self.embd % self.head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={self.embd}, head={self.head}."
            )
        self.backend = normalize_backend(self.backend, warn_legacy=True)

        # None is accepted for old checkpoints/configs and upgraded to auto.
        if self.compass is None:
            self.compass = "auto"
        elif isinstance(self.compass, str):
            if self.compass.strip().lower() != "auto":
                raise ValueError("compass must be a positive integer or 'auto'.")
            self.compass = "auto"
        else:
            self.compass = int(self.compass)
            if self.compass <= 0:
                raise ValueError(f"compass must be positive, got {self.compass}.")
