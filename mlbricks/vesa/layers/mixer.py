# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Projected ESA mixer shared by all model families."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..backends.lightning import lightning_step
from ..backends.native import native_lightning_step, native_scan
from ..backends.thunder import thunder_scan
from ..backends.registry import full_scan
from ..config import ESAConfig, FullBackend
from ...runtime import normalize_backend
from ...planner import EXECUTION_PLANNER


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
        EXECUTION_PLANNER.clear_owner_routes(self)
        return self

    def resolved_backend(self) -> str:
        policy = normalize_backend(self.config.backend)
        if policy == "pytorch":
            return "pytorch"
        if policy == "native":
            return "native-required"
        frozen = EXECUTION_PLANNER.owner_routes(self)
        routes = sorted(set(frozen.values()))
        if len(routes) == 1:
            return routes[0]
        try:
            from ..backends.native import native_available
            return "planner(auto)" if native_available() else "pytorch"
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
        if selected == "auto":
            is_training = bool(torch.is_grad_enabled())
            frozen = EXECUTION_PLANNER.owner_routes(self).get(("vesa_scan", is_training))
            if frozen in {"native", "pytorch"}:
                selected = frozen
                native_ok = frozen == "native"
            else:
                try:
                    from ..backends.native import native_available, native_cuda_built
                    native_ok = bool(native_available())
                    if gates.is_cuda:
                        native_ok = native_ok and bool(native_cuda_built())
                except Exception:
                    native_ok = False
            extra = (int(self.config.chunk_size), bool(reverse), int(self.config.dim))
            if frozen not in {"native", "pytorch"} and not torch.is_grad_enabled() and native_ok:
                selected = EXECUTION_PLANNER.qualify_operator_once(
                    self,
                    "vesa_scan",
                    gates,
                    {
                        "native": lambda: native_scan(
                            gates, values, initial_state, reverse=reverse
                        ),
                        "pytorch": lambda: thunder_scan(
                            gates, values, initial_state,
                            chunk_size=self.config.chunk_size, reverse=reverse,
                        ),
                    },
                    requested_backend="auto",
                    native_available=True,
                    native_supports_training=True,
                    training=False,
                    extra=extra,
                    default_auto="native",
                )
            elif frozen not in {"native", "pytorch"}:
                selected = EXECUTION_PLANNER.select_operator_once(
                    self,
                    "vesa_scan",
                    gates,
                    requested_backend="auto",
                    native_available=native_ok,
                    native_supports_training=True,
                    training=bool(torch.is_grad_enabled()),
                    extra=extra,
                    default_auto="native",
                )
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
        native_ok = False
        frozen = None
        if selected == "auto":
            frozen = EXECUTION_PLANNER.owner_routes(self).get(
                ("vesa_decode", bool(torch.is_grad_enabled()))
            )
            if frozen in {"native", "pytorch"}:
                selected = frozen
                native_ok = frozen == "native"
        if selected != "pytorch" and frozen not in {"native", "pytorch"}:
            try:
                from ..backends.native import native_available, native_cuda_built
                native_ok = bool(native_available())
                if gate.is_cuda:
                    native_ok = native_ok and bool(native_cuda_built())
            except Exception:
                native_ok = False
        if selected == "auto":
            extra = (int(self.config.dim),)
            if not torch.is_grad_enabled() and native_ok:
                selected = EXECUTION_PLANNER.qualify_operator_once(
                    self,
                    "vesa_decode",
                    gate,
                    {
                        "native": lambda: native_lightning_step(gate, value, state),
                        "pytorch": lambda: lightning_step(gate, value, state),
                    },
                    requested_backend="auto",
                    native_available=True,
                    native_supports_training=True,
                    training=False,
                    extra=extra,
                    default_auto="native",
                )
            else:
                selected = EXECUTION_PLANNER.select_operator_once(
                    self,
                    "vesa_decode",
                    gate,
                    requested_backend="auto",
                    native_available=native_ok,
                    native_supports_training=True,
                    training=bool(torch.is_grad_enabled()),
                    extra=extra,
                    default_auto="native",
                )
        if selected == "native":
            if not native_ok:
                raise RuntimeError("VESA backend='native' requested but native extension is unavailable")
            new_state = native_lightning_step(gate, value, state)
        else:
            new_state = lightning_step(gate, value, state)
        return self.out_proj(new_state), new_state
