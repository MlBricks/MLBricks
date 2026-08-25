"""Shared Diffusion and AR vision-engine implementations.

These modules are intentionally mixer-parametric so the public Vesa and
VisionBolt classes expose the same engine contract without duplicating model
plumbing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .bolt import Bolt
from .vesa.config import ESAConfig, VisionConfig
from .vesa.layers.local import LocalDepthwiseConv
from .vesa.layers.mixer import ESAMixer
from .vesa.layers.positional import timestep_embedding
from .vesa.models.common import AdaLNCondition, MLP, modulate
from .vision_common import (
    scan_indices,
    sinusoidal_2d_positions,
    apply_scan_native_or_pytorch,
    restore_scan_native_or_pytorch,
    add_sinusoidal_2d_native_or_pytorch,
)


def _make_mixer(family: str, config: VisionConfig, *, causal: bool):
    if family == "esa":
        return ESAMixer(
            ESAConfig(dim=config.dim, backend=config.backend, chunk_size=config.chunk_size)
        )
    if family == "bolt":
        if config.dim % config.heads:
            raise ValueError("VisionBolt requires dim divisible by heads")
        return Bolt(
            config.dim,
            config.heads,
            latent_dim=config.latent_dim,
            causal=causal,
            position=None,
            backend=config.backend,
            native_full_sequence=True,
        )
    raise ValueError("family must be 'esa' or 'bolt'")


def _run_mixer(mixer, family: str, x: Tensor) -> Tensor:
    if family == "esa":
        return mixer(x, reverse=False)[0]
    return mixer(x)


class _SpatialConditionedBlock(nn.Module):
    def __init__(self, family: str, config: VisionConfig, layer_index: int):
        super().__init__()
        self.family = family
        self.layer_index = int(layer_index)
        self.config = config
        self.local_norm = nn.LayerNorm(config.dim)
        self.local = LocalDepthwiseConv(config.dim, config.local_kernel_size)
        self.norm1 = nn.LayerNorm(config.dim, elementwise_affine=False)
        # Directional/casual Bolt makes scan order computationally meaningful.
        self.mixer = _make_mixer(family, config, causal=(family == "bolt"))
        self.norm2 = nn.LayerNorm(config.dim, elementwise_affine=False)
        self.mlp = MLP(config.dim, config.mlp_mult)
        self.condition = AdaLNCondition(config.dim)

    def forward(self, x: Tensor, condition: Tensor, grid: tuple[int, int]) -> Tensor:
        x = x + self.local(self.local_norm(x), grid)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.condition(condition)
        normalized = modulate(self.norm1(x), shift1, scale1)
        normalized = apply_scan_native_or_pytorch(
            normalized, *grid, scan=self.config.scan, layer_index=self.layer_index,
            backend=self.config.backend,
        )
        mixed = _run_mixer(self.mixer, self.family, normalized)
        mixed = restore_scan_native_or_pytorch(
            mixed, *grid, scan=self.config.scan, layer_index=self.layer_index,
            backend=self.config.backend,
        )
        x = x + gate1.unsqueeze(1) * mixed
        ffn = self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x + gate2.unsqueeze(1) * ffn


class ImageDiffusionEngine(nn.Module):
    """Patch-based image denoiser used by both Vesa and VisionBolt.

    Input/output shapes are ``[B, C, H, W]``. Spatial position is derived from
    directional scan order by default; an explicit 2-D position can still be
    requested for ablation experiments.
    """

    def __init__(self, family: str, config: VisionConfig):
        super().__init__()
        self.family = family
        self.config = config
        self.engine = "diffusion"
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        grid = config.image_size // config.patch_size
        self.grid = (grid, grid)
        tokens = grid * grid
        self.learned_position = (
            nn.Parameter(torch.zeros(1, tokens, config.dim))
            if config.position == "learned"
            else None
        )
        if self.learned_position is not None:
            nn.init.trunc_normal_(self.learned_position, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.SiLU(),
            nn.Linear(config.dim * 4, config.dim),
        )
        self.blocks = nn.ModuleList(
            _SpatialConditionedBlock(family, config, i) for i in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.dim)
        patch_values = config.patch_size * config.patch_size * config.in_channels
        self.output = nn.Linear(config.dim, patch_values)

    def _position(self, x: Tensor) -> Tensor:
        if self.config.position is None:
            return x
        if self.config.position == "learned":
            return x + self.learned_position.to(x.dtype)
        if self.config.position == "2d_sincos":
            return add_sinusoidal_2d_native_or_pytorch(
                x, *self.grid, backend=self.config.backend
            )
        raise RuntimeError(f"Unhandled position policy: {self.config.position}")

    def _unpatchify(self, patches: Tensor) -> Tensor:
        b, n, _ = patches.shape
        gh, gw = self.grid
        p = self.config.patch_size
        c = self.config.in_channels
        if n != gh * gw:
            raise ValueError("unexpected patch count")
        try:
            from .vision_native import unpatchify as native_unpatchify
            native = native_unpatchify(
                patches, gh, gw, p, c, backend=self.config.backend
            )
        except RuntimeError:
            if str(self.config.backend).strip().lower() == "native":
                raise
            native = None
        if native is not None:
            return native
        x = patches.view(b, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, gh * p, gw * p)

    def forward(self, images: Tensor, timesteps: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("diffusion images must have shape [B,C,H,W]")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(f"images must have {self.config.in_channels} channels")
        if images.shape[2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image size does not match configuration")
        if timesteps.shape != (images.shape[0],):
            raise ValueError("timesteps must have shape [batch]")

        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = self._position(x)
        condition = self.time_mlp(
            timestep_embedding(timesteps, self.config.dim).to(device=x.device, dtype=x.dtype)
        )
        for block in self.blocks:
            x = block(x, condition, self.grid)
        return self._unpatchify(self.output(self.final_norm(x)))

    @torch.no_grad()
    def benchmark_sample_loop(self, noise: Tensor, denoising_steps: int) -> Tensor:
        if denoising_steps <= 0:
            raise ValueError("denoising_steps must be positive")
        x = noise
        for step in range(denoising_steps, 0, -1):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            x = x - self(x, t) / float(denoising_steps)
        return x


class _ARBlock(nn.Module):
    def __init__(self, family: str, config: VisionConfig):
        super().__init__()
        self.family = family
        self.norm1 = nn.LayerNorm(config.dim)
        self.mixer = _make_mixer(family, config, causal=True)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_mult)

    def forward(self, x: Tensor) -> Tensor:
        mixed = _run_mixer(self.mixer, self.family, self.norm1(x))
        x = x + mixed
        return x + self.mlp(self.norm2(x))


class VisualAREngine(nn.Module):
    """Autoregressive visual-token model with scan-order-defined sequence.

    Token IDs are interpreted in the selected visual scan order. By default no
    positional embedding is added; the recurrent ESA or causal Bolt dependency
    makes sequence position meaningful.
    """

    def __init__(self, family: str, config: VisionConfig):
        super().__init__()
        self.family = family
        self.config = config
        self.engine = "ar"
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(_ARBlock(family, config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        grid = config.image_size // config.patch_size
        self.grid = (grid, grid)
        full_tokens = grid * grid
        if config.position == "learned":
            self.learned_position = nn.Parameter(torch.zeros(1, full_tokens, config.dim))
            nn.init.trunc_normal_(self.learned_position, std=0.02)
        else:
            self.learned_position = None

    def _positions(self, length: int, device, dtype) -> Tensor | None:
        if self.config.position is None:
            return None
        max_tokens = self.grid[0] * self.grid[1]
        if length > max_tokens:
            raise ValueError("visual AR sequence exceeds configured image token grid")
        if self.config.position == "learned":
            return self.learned_position[:, :length, :].to(dtype=dtype)
        if self.config.position == "2d_sincos":
            reference = torch.empty(1, 1, self.config.dim, device=device, dtype=dtype)
            try:
                from .vision_native import sincos2d as native_sincos
                pos = native_sincos(reference, *self.grid, backend=self.config.backend)
            except RuntimeError:
                if str(self.config.backend).strip().lower() == "native":
                    raise
                pos = None
            if pos is None:
                pos = sinusoidal_2d_positions(
                    *self.grid, self.config.dim, device=device, dtype=dtype
                )
            pos3 = apply_scan_native_or_pytorch(
                pos.unsqueeze(0), *self.grid, scan=self.config.scan,
                layer_index=0, backend=self.config.backend,
            )
            return pos3[:, :length, :]
        raise RuntimeError(f"Unhandled position policy: {self.config.position}")

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have non-empty shape [B,T]")
        x = self.embedding(input_ids)
        pos = self._positions(x.shape[1], x.device, x.dtype)
        if pos is not None:
            x = x + pos
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, generated_tokens: int) -> Tensor:
        if generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have non-empty shape [B,T]")
        out = prompt_ids
        generated = []
        for _ in range(generated_tokens):
            logits = self(out)
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(token)
            out = torch.cat([out, token], dim=1)
        if not generated:
            return prompt_ids.new_empty((prompt_ids.shape[0], 0))
        return torch.cat(generated, dim=1)


__all__ = ["ImageDiffusionEngine", "VisualAREngine"]
