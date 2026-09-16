# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Backend dispatch kept in one place to avoid hidden automatic selection."""

from __future__ import annotations

from torch import Tensor

from ..config import FullBackend
from ...runtime import normalize_backend
from ..core.reference import reference_scan
from .flare import flare_scan
from .native import native_scan
from .pulse import pulse_scan
from .thunder import thunder_scan


def full_scan(
    backend: FullBackend | str,
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    chunk_size: int = 64,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    policy = normalize_backend(backend, warn_legacy=True)
    if policy != "pytorch":
        try:
            from .native import native_available
            if native_available():
                return native_scan(gates, values, initial_state, reverse=reverse)
        except Exception:
            if policy == "native":
                raise
        if policy == "native":
            raise RuntimeError("VESA backend='native' requested but native extension is unavailable")

    # Thunder is retained internally as VESA's optimized PyTorch scan.
    return thunder_scan(
        gates, values, initial_state,
        chunk_size=chunk_size, reverse=reverse,
    )
