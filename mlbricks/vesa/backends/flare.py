# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Flare: recurrent full-sequence implementation suitable for torch.compile."""

from __future__ import annotations

import torch
from torch import Tensor


def flare_scan(
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    if gates.shape != values.shape or gates.ndim != 3:
        raise ValueError("gates and values must have matching [batch, tokens, dim] shapes")
    batch, _, dim = gates.shape
    state = (
        torch.zeros(batch, dim, device=gates.device, dtype=gates.dtype)
        if initial_state is None
        else initial_state
    )
    if state.shape != (batch, dim):
        raise ValueError(f"initial_state must have shape {(batch, dim)}")

    if reverse:
        gates = gates.flip(1)
        values = values.flip(1)

    outputs: list[Tensor] = []
    for gate, value in zip(gates.unbind(1), values.unbind(1), strict=True):
        state = gate * state + (1.0 - gate) * value
        outputs.append(state)

    states = torch.stack(outputs, dim=1) if outputs else gates.new_empty(gates.shape)
    if reverse:
        states = states.flip(1)
    return states, state
