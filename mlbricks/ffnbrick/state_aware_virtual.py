from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .norm import RMSNorm
from .state_aware import StateAwareFFN
from . import native as _native


class _VirtualStateRefiner(nn.Module):
    def __init__(self, d_model: int, state_dim: int, hidden_dim: int, refinements: int) -> None:
        super().__init__()
        self.refinements = int(refinements)
        self.norm = RMSNorm(state_dim)
        self.state_up = nn.Linear(state_dim, hidden_dim)
        self.x_condition = nn.Linear(d_model, hidden_dim, bias=False)
        self.esa_condition = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, state_dim)
        self.pass_embedding = nn.Parameter(torch.empty(refinements, hidden_dim))
        self.gate_logit = nn.Parameter(torch.full((refinements, state_dim), -2.0))
        nn.init.normal_(self.pass_embedding, mean=0.0, std=0.02)
        self.reset_identity()

    def reset_identity(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    def forward(
        self, state: torch.Tensor, x_condition: torch.Tensor,
        esa_condition: torch.Tensor, refinement_index: int,
    ) -> torch.Tensor:
        hidden = F.silu(
            self.state_up(self.norm(state)) + x_condition + esa_condition
            + self.pass_embedding[refinement_index]
        )
        gate = torch.sigmoid(self.gate_logit[refinement_index])
        return state + gate * self.down(hidden)


class VirtualStateAwareFFN(StateAwareFFN):
    """ESA-State-Aware FFN with shared condition-aware virtual state passes.

    Model-width hidden and ESA conditioning are computed once, then reused by
    all virtual refinements. The virtual output projection is zero-initialized,
    making this module functionally identical to :class:`StateAwareFFN` at
    initialization when shared weights are copied.
    """

    def __init__(
        self, d_model: int, state_dim: int = 256, depth_embedding_dim: int = 64,
        layer_index: int = 0, total_layers: int = 1,
        virtual_refinements: int = 2, virtual_hidden_dim: int = 128,
        use_native: bool | None = None, fused_cuda: bool = True, backend: str = "auto",
    ) -> None:
        if virtual_refinements <= 0 or virtual_hidden_dim <= 0:
            raise ValueError("virtual_refinements and virtual_hidden_dim must be positive")
        super().__init__(
            d_model, state_dim, depth_embedding_dim, layer_index, total_layers,
            use_native=use_native, fused_cuda=fused_cuda, backend=backend,
        )
        self.virtual_refiner = _VirtualStateRefiner(
            d_model, state_dim, virtual_hidden_dim, virtual_refinements
        )

    @property
    def virtual_gate_mean(self) -> torch.Tensor:
        return torch.sigmoid(self.virtual_refiner.gate_logit).mean()

    def reset_virtual_identity(self) -> None:
        self.virtual_refiner.reset_identity()

    def forward_python(
        self, x: torch.Tensor, esa_update: torch.Tensor,
        previous_esa: torch.Tensor, previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x, esa_update, previous_esa, previous_state)
        next_state, scaled_delta, depth_value = self._state_update(
            x, esa_update, previous_esa, previous_state
        )
        x_condition = self.virtual_refiner.x_condition(x)
        esa_condition = self.virtual_refiner.esa_condition(esa_update)
        for index in range(self.virtual_refiner.refinements):
            next_state = self.virtual_refiner(next_state, x_condition, esa_condition, index)
        return self._read(x, next_state, scaled_delta, depth_value), next_state

    def forward(
        self, x: torch.Tensor, esa_update: torch.Tensor,
        previous_esa: torch.Tensor, previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x, esa_update, previous_esa, previous_state)
        self._require_native_for(x)
        if _native.inference_native_allowed(self, x):
            return _native.virtual_state_aware_forward(
                self, x, esa_update, previous_esa, previous_state
            )
        return self.forward_python(x, esa_update, previous_esa, previous_state)
