# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Projected ESA mixer shared by all model families."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..backends.lightning import lightning_step
from ..backends.native import native_lightning_step
from ..backends.registry import full_scan
from ..config import ESAConfig, FullBackend
from ...runtime import normalize_backend


class ESAMixer(nn.Module):
    def __init__(self, config: ESAConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.dim, 2 * config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)
        if config.gate_bias:
            with torch.no_grad():
                self.in_proj.bias[: config.dim].fill_(config.gate_bias)

    @property
    def backend(self) -> str:
        return str(self.config.backend)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        from dataclasses import replace
        self.config = replace(self.config, backend=normalize_backend(backend, warn_legacy=True))
        return self

    def resolved_backend(self) -> str:
        policy = normalize_backend(self.config.backend)
        if policy == "pytorch":
            return "pytorch"
        if policy == "native":
            return "native-required"
        try:
            from ..backends.native import native_available
            return "native" if native_available() else "pytorch"
        except Exception:
            return "pytorch"

    def project(self, x: Tensor) -> tuple[Tensor, Tensor]:
        gate_logits, value_logits = self.in_proj(x).chunk(2, dim=-1)
        return torch.sigmoid(gate_logits), torch.tanh(value_logits)

    def forward(
        self,
        x: Tensor,
        initial_state: Tensor | None = None,
        *,
        reverse: bool = False,
        backend: FullBackend | None = None,
    ) -> tuple[Tensor, Tensor]:
        selected = normalize_backend(backend or self.config.backend, warn_legacy=True)
        gates, values = self.project(x)
        states, final_state = full_scan(
            selected,
            gates,
            values,
            initial_state,
            chunk_size=self.config.chunk_size,
            reverse=reverse,
        )
        return self.out_proj(states), final_state

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2:
            raise ValueError("step input must have shape [batch, dim]")
        gate, value = self.project(x)
        selected = normalize_backend(self.config.backend, warn_legacy=True)
        if selected != "pytorch":
            try:
                from ..backends.native import native_available
                if native_available():
                    new_state = native_lightning_step(gate, value, state)
                elif selected == "native":
                    raise RuntimeError("VESA backend='native' requested but native extension is unavailable")
                else:
                    new_state = lightning_step(gate, value, state)
            except RuntimeError:
                if selected == "native":
                    raise
                new_state = lightning_step(gate, value, state)
        else:
            new_state = lightning_step(gate, value, state)
        return self.out_proj(new_state), new_state
