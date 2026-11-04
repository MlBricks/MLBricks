# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import torch
import torch.nn.functional as F



def associative_compass_scan(
    A_compass: torch.Tensor,
    B_compass: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive affine scan over compass summaries."""
    if A_compass.shape != B_compass.shape:
        raise ValueError("A_compass and B_compass must have the same shape.")
    if A_compass.dim() != 4:
        raise ValueError(
            f"expected [B,G,H,D], got {tuple(A_compass.shape)}"
        )

    A = A_compass
    B = B_compass
    step = 1

    while step < A.size(1):
        A_prev, B_prev = A, B
        A_next, B_next = A.clone(), B.clone()

        A_next[:, step:] = A_prev[:, step:] * A_prev[:, :-step]
        B_next[:, step:] = (
            A_prev[:, step:] * B_prev[:, :-step]
            + B_prev[:, step:]
        )

        A, B = A_next, B_next
        step *= 2

    return A, B


def lightning_scan(
    A: torch.Tensor,
    B_write: torch.Tensor,
    *,
    compass: int = 4,
) -> torch.Tensor:
    """
    Exact chunked affine ESA scan used by the Lightning backend.

    Recurrence:
        E_t = A_t * E_{t-1} + B_t

    ``compass=4`` is the Lightning default.
    """
    # Native C++/CUDA inference fast path. The native kernel computes the same
    # causal recurrence directly in one launch; training keeps the original
    # PyTorch chunked implementation for autograd.
    try:
        from ..native import enabled_for as _native_enabled_for
        from ..native import thunder_scan as _native_scan
        if _native_enabled_for(A):
            return _native_scan(A, B_write, compass)
    except Exception:
        import os
        if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
            raise

    if A.shape != B_write.shape:
        raise ValueError(
            f"A and B_write must match, got {A.shape} and {B_write.shape}"
        )
    if A.dim() != 4:
        raise ValueError(f"expected [B,T,H,D], got {tuple(A.shape)}")
    if not isinstance(compass, int) or compass <= 0:
        raise ValueError(
            f"compass must be a positive integer, got {compass!r}"
        )

    Bsz, T, H, D = A.shape
    pad = (-T) % compass

    if pad:
        A = F.pad(A, (0, 0, 0, 0, 0, pad), value=1.0)
        B_write = F.pad(
            B_write,
            (0, 0, 0, 0, 0, pad),
            value=0.0,
        )

    Tp = A.size(1)
    groups = Tp // compass

    A5 = A.reshape(Bsz, groups, compass, H, D)
    B5 = B_write.reshape(Bsz, groups, compass, H, D)

    state = B_write.new_zeros(Bsz, groups, H, D)
    transition = A.new_ones(Bsz, groups, H, D)

    local_states = []
    prefix_as = []

    for i in range(compass):
        A_i = A5[:, :, i]
        B_i = B5[:, :, i]

        state = A_i * state + B_i
        transition = A_i * transition

        local_states.append(state)
        prefix_as.append(transition)

    local_state = torch.stack(local_states, dim=2)
    prefix_A = torch.stack(prefix_as, dim=2)

    A_compass = prefix_A[:, :, -1]
    B_compass = local_state[:, :, -1]

    _, compass_end_state = associative_compass_scan(
        A_compass,
        B_compass,
    )

    zero = compass_end_state.new_zeros(Bsz, 1, H, D)
    compass_init = torch.cat(
        [zero, compass_end_state[:, :-1]],
        dim=1,
    )

    E = prefix_A * compass_init.unsqueeze(2) + local_state
    E = E.reshape(Bsz, Tp, H, D)

    if pad:
        E = E[:, :T]

    return E
