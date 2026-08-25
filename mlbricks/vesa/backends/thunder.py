# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Thunder: exact chunked full-sequence ESA scan."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def thunder_scan(
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    chunk_size: int = 64,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Run an exact chunked scan without changing the recurrence or receptive field."""

    if gates.ndim != 3 or gates.shape != values.shape:
        raise ValueError("gates and values must have matching [batch, tokens, dim] shapes")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch, original_tokens, dim = gates.shape
    if initial_state is None:
        initial_state = torch.zeros(batch, dim, device=gates.device, dtype=gates.dtype)
    if initial_state.shape != (batch, dim):
        raise ValueError(f"initial_state must have shape {(batch, dim)}")

    if reverse:
        gates = gates.flip(1)
        values = values.flip(1)

    if original_tokens == 0:
        return gates.new_empty(gates.shape), initial_state

    pad = (-original_tokens) % chunk_size
    if pad:
        gates = F.pad(gates, (0, 0, 0, pad), value=1.0)
        values = F.pad(values, (0, 0, 0, pad), value=0.0)

    padded_tokens = gates.shape[1]
    chunks = padded_tokens // chunk_size
    chunk_gates = gates.reshape(batch, chunks, chunk_size, dim)
    chunk_values = values.reshape(batch, chunks, chunk_size, dim)

    attenuation_prefix = torch.cumprod(chunk_gates, dim=2)

    local_outputs: list[Tensor] = []
    local_state = torch.zeros(batch, chunks, dim, device=gates.device, dtype=gates.dtype)
    for index in range(chunk_size):
        gate = chunk_gates[:, :, index, :]
        value = chunk_values[:, :, index, :]
        local_state = gate * local_state + (1.0 - gate) * value
        local_outputs.append(local_state)
    local_prefix = torch.stack(local_outputs, dim=2)

    chunk_attenuation = attenuation_prefix[:, :, -1, :]
    chunk_offset = local_prefix[:, :, -1, :]

    incoming_states: list[Tensor] = []
    state = initial_state
    for chunk_index in range(chunks):
        incoming_states.append(state)
        state = (
            chunk_attenuation[:, chunk_index, :] * state
            + chunk_offset[:, chunk_index, :]
        )
    incoming = torch.stack(incoming_states, dim=1)

    states = local_prefix + attenuation_prefix * incoming.unsqueeze(2)
    states = states.reshape(batch, padded_tokens, dim)[:, :original_tokens, :]
    if reverse:
        states = states.flip(1)
    return states, state
