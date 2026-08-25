from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import native as _native
from ..runtime import normalize_backend


class StateAwareFFN(nn.Module):
    """ESA-conditioned recurrent feature network across physical depth.

    The state is token-wise with shape ``[..., state_dim]``. The forward pass
    uses the normalized hidden stream, current ESA output, previous ESA output,
    previous FFN state, and a learned physical-depth embedding.

    If the native extension is available, orchestration is performed in C++.
    On CUDA, GEMMs dispatch to cuBLAS through ATen and inference can fuse the
    recurrent gate/state and read-mix elementwise stages using custom kernels.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 256,
        depth_embedding_dim: int = 64,
        layer_index: int = 0,
        total_layers: int = 1,
        use_native: bool | None = None,
        fused_cuda: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if min(d_model, state_dim, depth_embedding_dim, total_layers) <= 0:
            raise ValueError("dimensions and total_layers must be positive")
        if not 0 <= layer_index < total_layers:
            raise ValueError("layer_index must be in [0, total_layers)")
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.backend = normalize_backend(backend, warn_legacy=True)
        if use_native is not None and self.backend == "auto":
            self.backend = "auto" if bool(use_native) else "pytorch"
        self.use_native = self.backend != "pytorch"
        self.fused_cuda = bool(fused_cuda)
        self._ffnbrick_native_cache = None

        self.x_candidate = nn.Linear(d_model, state_dim)
        self.esa_candidate = nn.Linear(d_model, state_dim, bias=False)
        self.state_candidate = nn.Linear(state_dim, state_dim, bias=False)
        self.x_write = nn.Linear(d_model, state_dim)
        self.esa_write = nn.Linear(d_model, state_dim, bias=False)
        self.state_write = nn.Linear(state_dim, state_dim, bias=False)
        self.value = nn.Linear(d_model, state_dim)
        self.output = nn.Linear(state_dim, d_model)

        self.depth_embedding = nn.Parameter(torch.empty(depth_embedding_dim))
        self.depth_to_candidate = nn.Linear(depth_embedding_dim, state_dim, bias=False)
        self.depth_to_write = nn.Linear(depth_embedding_dim, state_dim, bias=False)
        self.depth_to_value = nn.Linear(depth_embedding_dim, state_dim, bias=False)

        depth = layer_index / max(total_layers - 1, 1)
        self.retain_logit = nn.Parameter(torch.full((state_dim,), 1.4 - 0.5 * depth))
        self.read_logit = nn.Parameter(torch.full((state_dim,), -0.2 + 0.4 * depth))
        nn.init.normal_(self.depth_embedding, mean=0.0, std=0.02)

        self.candidate_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.write_transition_logit = nn.Parameter(torch.tensor(-2.0))
        self.retain_delta_scale = nn.Parameter(torch.full((state_dim,), 0.10))
        self.read_delta_scale = nn.Parameter(torch.full((state_dim,), 0.10))
        self.delta_magnitude_log_scale = nn.Parameter(torch.tensor(-1.0))

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = normalize_backend(backend, warn_legacy=True)
        self.use_native = self.backend != "pytorch"
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        return "planner(auto)" if _native.is_available() else "pytorch"

    def _require_native_for(self, x: torch.Tensor) -> None:
        if self.backend == "native" and not _native.inference_native_allowed(self, x):
            raise RuntimeError(
                "StateAwareFFN backend='native' requested but native eager inference "
                "is unavailable for this call"
            )

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], self.state_dim)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        esa_update: torch.Tensor,
        previous_esa: torch.Tensor,
        previous_state: torch.Tensor,
    ) -> None:
        if x.shape[-1] != self.d_model or esa_update.shape != x.shape or previous_esa.shape != x.shape:
            raise ValueError("x, esa_update, and previous_esa must share [..., d_model] shape")
        if previous_state.shape[:-1] != x.shape[:-1] or previous_state.shape[-1] != self.state_dim:
            raise ValueError("previous_state must have shape [..., state_dim]")

    def _state_update(
        self, x: torch.Tensor, esa_update: torch.Tensor,
        previous_esa: torch.Tensor, previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(x, esa_update, previous_esa, previous_state)

        depth_candidate = self.depth_to_candidate(self.depth_embedding)
        depth_write = self.depth_to_write(self.depth_embedding)
        depth_value = self.depth_to_value(self.depth_embedding)
        esa_delta = esa_update - previous_esa
        delta_magnitude = torch.sqrt(
            esa_delta.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(esa_update.dtype)
        scaled_delta = torch.exp(self.delta_magnitude_log_scale) * delta_magnitude

        candidate_esa = esa_update + torch.sigmoid(self.candidate_transition_logit) * esa_delta
        write_esa = esa_update + torch.sigmoid(self.write_transition_logit) * esa_delta
        candidate = torch.tanh(
            self.x_candidate(x) + self.esa_candidate(candidate_esa)
            + self.state_candidate(previous_state) + depth_candidate
        )
        write_gate = torch.sigmoid(
            self.x_write(x) + self.esa_write(write_esa)
            + self.state_write(previous_state) + depth_write
        )
        retain_gate = torch.sigmoid(
            self.retain_logit - scaled_delta * self.retain_delta_scale
        )
        next_state = (1.0 - write_gate) * (retain_gate * previous_state) + write_gate * candidate
        return next_state, scaled_delta, depth_value

    def _read(
        self, x: torch.Tensor, next_state: torch.Tensor,
        scaled_delta: torch.Tensor, depth_value: torch.Tensor,
    ) -> torch.Tensor:
        value = F.silu(self.value(x) + depth_value)
        read_gate = torch.sigmoid(self.read_logit + scaled_delta * self.read_delta_scale)
        return self.output(next_state * value * read_gate)

    def forward_python(
        self, x: torch.Tensor, esa_update: torch.Tensor,
        previous_esa: torch.Tensor, previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference PyTorch implementation, useful for parity/benchmarking."""
        next_state, scaled_delta, depth_value = self._state_update(
            x, esa_update, previous_esa, previous_state
        )
        return self._read(x, next_state, scaled_delta, depth_value), next_state

    def forward(
        self, x: torch.Tensor, esa_update: torch.Tensor,
        previous_esa: torch.Tensor, previous_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(x, esa_update, previous_esa, previous_state)
        # Keep the exact original PyTorch equations visible to autograd and
        # torch.compile.  The native path is an eager no-grad inference
        # specialization rather than an opaque replacement for the graph.
        self._require_native_for(x)
        if _native.inference_native_allowed(self, x):
            return _native.state_aware_forward(
                self, x, esa_update, previous_esa, previous_state
            )
        return self.forward_python(x, esa_update, previous_esa, previous_state)
