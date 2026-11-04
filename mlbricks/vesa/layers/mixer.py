# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""VESA adapter around the canonical MLBricks ESA engine.

VESA deliberately does not carry a second ESA implementation. Spatial scan,
local vision processing, normalization, residuals, and model plumbing remain
VESA responsibilities; sequence mixing is delegated to :class:`mlbricks.esa.ESA`.
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
from torch import Tensor

from ...esa import ESA
from ...runtime import normalize_backend
from ..config import ESAConfig, FullBackend


class ESAMixer(nn.Module):
    """Thin compatibility adapter that uses :class:`mlbricks.esa.ESA` as core.

    The historical VESA ``ESAMixer`` API is preserved: ``forward`` returns
    ``(outputs, final_state)`` and ``step`` performs one recurrent update. The
    recurrent state is exposed in flattened ``[B, dim]`` layout for backwards
    compatibility, while the canonical ESA engine internally uses headed state.
    """

    def __init__(self, config: ESAConfig):
        super().__init__()
        self.config = config
        # Early VESA exposed a ``heads`` field that did not constrain its ESA
        # mixer, so configs such as dim=16, heads=6 were valid. Canonical ESA
        # requires divisibility. Preserve source compatibility by using the
        # requested partition when possible and a mathematically equivalent
        # single-head partition otherwise (ESA recurrence/readout are elementwise).
        self.core_heads = config.heads if config.dim % config.heads == 0 else 1
        self.engine = ESA(
            embd=config.dim,
            head=self.core_heads,
            backend=config.backend,
            device=None,
        )

    @property
    def backend(self) -> str:
        return str(self.engine.resolved_backend())

    @property
    def core_engine(self) -> ESA:
        """Return the canonical ESA instance used by this VESA mixer."""
        return self.engine

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        value = normalize_backend(backend, warn_legacy=True)
        self.config = replace(self.config, backend=value)
        self.engine.set_backend(value)
        return self

    def resolved_backend(self) -> str:
        return self.engine.resolved_backend()

    def project(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return canonical ESA affine recurrence terms ``A`` and ``B_write``.

        This method is retained for compatibility with code that inspected the
        old VESA projection. Its semantics now intentionally match canonical ESA
        rather than the removed two-way gate/value projection.
        """
        import math
        import torch.nn.functional as F
        core = self.engine
        B, T, D = x.shape
        _, raw_gate, raw_value = core.qgv(x).chunk(3, dim=-1)
        gate = raw_gate.reshape(B, T, self.core_heads, D // self.core_heads).float()
        value = torch.tanh(raw_value.reshape(B, T, self.core_heads, D // self.core_heads).float())
        if self.engine.esa_variant == "current":
            half = core.log_half_life.exp()
            decay = F.softplus(gate + core.raw_decay_bias[None, None, :, None])
            A = torch.exp(-math.log(2.0) * decay / half[None, None, :, None])
        else:
            alpha, beta = core.bounds()
            A = alpha[None, None, :, None] + (beta - alpha)[None, None, :, None] * torch.sigmoid(gate)
        B_write = (1.0 - A) * value
        return A.flatten(2), B_write.flatten(2)

    def _headed_state(self, state: Tensor | None) -> Tensor | None:
        if state is None:
            return None
        if state.ndim == 2:
            return state.reshape(state.shape[0], self.core_heads,
                                 self.config.dim // self.core_heads)
        return state

    def forward(self, x: Tensor, initial_state: Tensor | None = None, *,
                reverse: bool = False, backend: FullBackend | None = None):
        if backend is not None:
            self.engine.set_backend(normalize_backend(backend, warn_legacy=True))
        # Reverse is a VESA sequence-layout feature, outside native ESA.
        tokens = x.flip(1) if reverse else x
        y, state = self.engine(tokens, self._headed_state(initial_state))
        return (y.flip(1) if reverse else y), state.flatten(1)

    @torch.no_grad()
    def prefill(self, x: Tensor, initial_state: Tensor | None = None, *,
                backend: FullBackend | None = None):
        if backend is not None:
            self.engine.set_backend(normalize_backend(backend, warn_legacy=True))
        output, state = self.engine.prefill(x, self._headed_state(initial_state))
        return output, state.flatten(1)

    def step(self, x: Tensor, state: Tensor):
        if x.ndim != 2:
            raise ValueError("step input must have shape [batch, dim]")
        output, new_state = self.engine.decode_step(x, self._headed_state(state))
        # VESA's spatial step contract is [B,D]. ESA itself remains [B,1,D].
        return output[:, 0], new_state.flatten(1)

    decode_step = step
    lightning_prefill = prefill
    lightning_step = step
