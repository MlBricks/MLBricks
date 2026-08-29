# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Image classifiers built around ESA with selectable spatial engines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import ESAConfig, VisionConfig
from ..layers.local import LocalDepthwiseConv
from ..layers.mixer import ESAMixer
from ..layers.normalization import PerspectiveNorm
from .common import MLP
from ...ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from ...residualbrick import ResController
from ...vision_common import (
    scan_indices,
    sinusoidal_2d_positions,
    apply_scan_native_or_pytorch,
    restore_scan_native_or_pytorch,
    add_sinusoidal_2d_native_or_pytorch,
)


def serpentine_indices(height: int, width: int, device: torch.device) -> Tensor:
    """Backward-compatible horizontal serpentine traversal."""
    return scan_indices(height, width, scan="horizontal", layer_index=0, device=device)


class VisionESABlock(nn.Module):
    """One ESA vision block.

    Tokens enter and leave in canonical row-major order. Directional scan
    permutations are applied only around the ESA mixer, so local convolution
    always operates on the true 2-D grid.
    """

    def __init__(self, config: VisionConfig, *, reverse: bool = False, layer_index: int):
        super().__init__()
        self.reverse = bool(reverse)  # legacy field; scan order now carries direction
        self.layer_index = int(layer_index)
        self.engine = config.engine
        self.ffn_kind = config.ffn
        self.residual_kind = config.residual
        self.use_local = config.engine in {"serpentine", "cnn"}
        self.scan = config.scan
        self.backend = config.backend

        self.local_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        self.local = LocalDepthwiseConv(config.dim, config.local_kernel_size)
        self.cnn_pointwise = (
            nn.Sequential(nn.Linear(config.dim, config.dim), nn.GELU())
            if config.engine == "cnn"
            else nn.Identity()
        )

        self.state_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        self.mixer = ESAMixer(
            ESAConfig(dim=config.dim, backend=config.backend, chunk_size=config.chunk_size)
        )
        self.mlp_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)

        if self.ffn_kind == "standard":
            self.mlp = MLP(config.dim, config.mlp_mult)
        elif self.ffn_kind == "ffnbrick":
            self.mlp = StateAwareFFN(
                d_model=config.dim,
                state_dim=config.ffn_state_dim,
                depth_embedding_dim=config.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=config.depth,
                backend=config.backend,
            )
        elif self.ffn_kind == "virtual_ffnbrick":
            self.mlp = VirtualStateAwareFFN(
                d_model=config.dim,
                state_dim=config.ffn_state_dim,
                depth_embedding_dim=config.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=config.depth,
                virtual_refinements=config.ffn_virtual_refinements,
                virtual_hidden_dim=config.ffn_virtual_hidden_dim,
                backend=config.backend,
            )
        elif self.ffn_kind == "micro_ffnbrick":
            self.mlp = MicroVirtualFFN(
                d_model=config.dim,
                hidden_dim=config.ffn_micro_hidden_dim,
                refinements=config.ffn_micro_refinements,
                backend=config.backend,
            )
        else:  # pragma: no cover - validated by config
            raise RuntimeError(f"Unhandled VESA FFN component: {self.ffn_kind}")

        if self.residual_kind == "rescontroller":
            controller_kwargs = dict(
                update_ratio=config.residual_update_ratio,
                stream_ratio=config.residual_stream_ratio,
                update_softness=config.residual_update_softness,
                stream_softness=config.residual_stream_softness,
                backend=config.backend,
            )
            self.local_residual = ResController(**controller_kwargs)
            self.esa_residual = ResController(**controller_kwargs)
            self.ffn_residual = ResController(**controller_kwargs)
        else:
            self.local_residual = None
            self.esa_residual = None
            self.ffn_residual = None

    @staticmethod
    def _combine(controller, residual: Tensor, update: Tensor) -> Tensor:
        return residual + update if controller is None else controller(residual, update)

    def _ffn_update(
        self,
        normalized: Tensor,
        esa_update: Tensor,
        previous_esa: Tensor | None,
        previous_ffn_state: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.ffn_kind == "standard":
            return self.mlp(normalized), previous_ffn_state
        if self.ffn_kind == "micro_ffnbrick":
            hidden = normalized
            total_update = torch.zeros_like(normalized)
            for refinement_index in range(self.mlp.refinements):
                update = self.mlp(hidden, refinement_index)
                total_update = total_update + update
                hidden = hidden + update
            return total_update, previous_ffn_state
        if previous_esa is None:
            previous_esa = torch.zeros_like(esa_update)
        if previous_ffn_state is None:
            previous_ffn_state = self.mlp.initial_state(normalized)
        return self.mlp(normalized, esa_update, previous_esa, previous_ffn_state)

    def forward(
        self,
        x: Tensor,
        grid: tuple[int, int],
        order: Tensor | None = None,
        inverse_order: Tensor | None = None,
        previous_esa: Tensor | None = None,
        previous_ffn_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        # Local spatial work always sees canonical row-major tokens.
        if self.use_local:
            local_output = self.local(self.local_norm(x), grid)
            local_output = self.cnn_pointwise(local_output)
            x = self._combine(self.local_residual, x, local_output)

        state_input = self.state_norm(x)
        use_scan = self.engine in {"serpentine", "cnn"}
        if use_scan:
            state_input = apply_scan_native_or_pytorch(
                state_input, *grid, scan=self.scan, layer_index=self.layer_index,
                backend=self.backend, owner=self,
            )
        elif order is not None:
            if inverse_order is None:
                inverse_order = torch.argsort(order)
            state_input = state_input.index_select(1, order)

        mixed, _ = self.mixer(state_input, reverse=False)
        if use_scan:
            mixed = restore_scan_native_or_pytorch(
                mixed, *grid, scan=self.scan, layer_index=self.layer_index,
                backend=self.backend, owner=self,
            )
        elif order is not None:
            mixed = mixed.index_select(1, inverse_order)
        x = self._combine(self.esa_residual, x, mixed)

        ffn_update, next_ffn_state = self._ffn_update(
            self.mlp_norm(x), mixed, previous_esa, previous_ffn_state
        )
        x = self._combine(self.ffn_residual, x, ffn_update)
        return x, mixed, next_ffn_state


class VisionESAClassifier(nn.Module):
    """ESA image classifier supporting Serpentine, ViT and CNN engines."""

    def __init__(self, config: VisionConfig | None = None):
        super().__init__()
        config = config or VisionConfig()
        if config.engine not in {"serpentine", "vit", "cnn"}:
            raise ValueError(
                "VisionESAClassifier supports classifier engines: Serpentine, ViT, CNN"
            )
        self.config = config
        self.engine = config.engine
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        grid_size = config.image_size // config.patch_size
        self.grid_size = (grid_size, grid_size)
        token_count = grid_size * grid_size
        if config.position == "learned":
            self.learned_position = nn.Parameter(torch.zeros(1, token_count, config.dim))
            nn.init.trunc_normal_(self.learned_position, std=0.02)
        else:
            self.register_parameter("learned_position", None)

        self.blocks = nn.ModuleList(
            VisionESABlock(
                config,
                reverse=config.alternating_reverse and index % 2 == 1,
                layer_index=index,
            )
            for index in range(config.depth)
        )
        self.final_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        self.head = nn.Linear(config.dim, config.num_classes)

    def _add_position(self, hidden: Tensor, grid: tuple[int, int]) -> Tensor:
        if self.config.position is None:
            return hidden
        if self.config.position == "learned":
            return hidden + self.learned_position[:, : hidden.shape[1], :].to(hidden.dtype)
        if self.config.position == "2d_sincos":
            return add_sinusoidal_2d_native_or_pytorch(
                hidden, *grid, backend=self.config.backend, owner=self
            )
        raise RuntimeError(f"Unhandled position policy: {self.config.position}")

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(f"images must have {self.config.in_channels} channels")
        if images.shape[2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(
                f"images spatial size must be {(self.config.image_size, self.config.image_size)}"
            )

        hidden = self.patch_embed(images)
        grid = (hidden.shape[2], hidden.shape[3])
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = self._add_position(hidden, grid)

        previous_esa = None
        ffn_state = None
        for index, block in enumerate(self.blocks):
            hidden, previous_esa, ffn_state = block(
                hidden,
                grid,
                None,
                None,
                previous_esa=previous_esa,
                previous_ffn_state=ffn_state,
            )
        return self.head(self.final_norm(hidden).mean(dim=1))
