# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Lightning: one-token recurrent ESA state update."""

from __future__ import annotations

from torch import Tensor


def lightning_step(gate: Tensor, value: Tensor, state: Tensor) -> Tensor:
    if gate.shape != value.shape or gate.shape != state.shape:
        raise ValueError("gate, value, and state must have identical [batch, dim] shapes")
    if gate.ndim != 2:
        raise ValueError("gate, value, and state must have shape [batch, dim]")
    return gate * state + (1.0 - gate) * value
