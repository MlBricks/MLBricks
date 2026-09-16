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
            precision=config.precision,
            compass=config.compass,
            gate_min=config.gate_min,
            gate_max=config.gate_max,
            eps=config.eps,
            device=None,
            auto_move_input=False,
        )

    @property
    def backend(self) -> str:
        return str(self.engine.backend)

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
        from ...esa.generation import _project_affine_terms

        _, A, B_write = _project_affine_terms(self.engine, x)
        return A.flatten(2), B_write.flatten(2)

    def forward(
        self,
        x: Tensor,
        initial_state: Tensor | None = None,
        *,
        reverse: bool = False,
        backend: FullBackend | None = None,
    ) -> tuple[Tensor, Tensor]:
        selected = None if backend is None else normalize_backend(backend, warn_legacy=True)
        outputs, final_state = self.engine.forward_with_state(
            x,
            state=initial_state,
            backend=selected,
            reverse=bool(reverse),
        )
        return outputs, final_state.reshape(final_state.shape[0], self.config.dim)

    @torch.no_grad()
    def prefill(
        self,
        x: Tensor,
        initial_state: Tensor | None = None,
        *,
        backend: FullBackend | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run canonical ESA prefill and return a flattened recurrent state."""
        selected = None if backend is None else normalize_backend(backend, warn_legacy=True)
        outputs, final_state = self.engine.prefill(
            x,
            state=initial_state,
            backend=selected,
        )
        return outputs, final_state.reshape(final_state.shape[0], self.config.dim)

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError("step input must have shape [batch, dim]")
        if state.ndim not in {2, 3}:
            raise ValueError("state must have shape [batch, dim] or [batch, heads, head_dim]")
        output, new_state = self.engine.decode_step(x, state)
        return output, new_state.reshape(new_state.shape[0], self.config.dim)

    decode_step = step
    lightning_prefill = prefill
    lightning_step = step
