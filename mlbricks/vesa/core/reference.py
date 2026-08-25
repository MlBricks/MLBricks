# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Mathematical source of truth for the controlled ESA recurrence."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(gates: Tensor, values: Tensor, initial_state: Tensor | None) -> Tensor:
    if gates.ndim != 3 or values.ndim != 3:
        raise ValueError("gates and values must have shape [batch, tokens, dim]")
    if gates.shape != values.shape:
        raise ValueError("gates and values must have identical shapes")
    batch, _, dim = gates.shape
    if initial_state is None:
        return torch.zeros(batch, dim, device=gates.device, dtype=gates.dtype)
    if initial_state.shape != (batch, dim):
        raise ValueError(f"initial_state must have shape {(batch, dim)}")
    return initial_state


def reference_scan(
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Evaluate ``s_t = g_t*s_(t-1) + (1-g_t)*v_t`` token by token."""

    state = _validate(gates, values, initial_state)
    if reverse:
        gates = gates.flip(1)
        values = values.flip(1)

    outputs: list[Tensor] = []
    for token_index in range(gates.shape[1]):
        gate = gates[:, token_index, :]
        value = values[:, token_index, :]
        state = gate * state + (1.0 - gate) * value
        outputs.append(state)

    if outputs:
        states = torch.stack(outputs, dim=1)
    else:
        states = gates.new_empty(gates.shape)

    if reverse:
        states = states.flip(1)
    return states, state
