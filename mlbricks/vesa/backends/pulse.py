# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Pulse: explicit reference-compatible full-sequence backend."""

from __future__ import annotations

from torch import Tensor

from ..core.reference import reference_scan


def pulse_scan(
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    return reference_scan(gates, values, initial_state, reverse=reverse)
