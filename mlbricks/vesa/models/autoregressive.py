# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Matched ESA and causal-SDPA autoregressive token models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import AutoregressiveConfig, ESAConfig
from ..layers.attention import AttentionMixer
from ..layers.mixer import ESAMixer
from ..layers.positional import sinusoidal_positions
from .common import MLP


class ESAARBlock(nn.Module):
    def __init__(self, config: AutoregressiveConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.mixer = ESAMixer(
            ESAConfig(
                dim=config.dim,
                backend=config.prefill_backend,
                chunk_size=config.chunk_size,
            )
        )
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_mult)

    def prefill(self, x: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        mixed, state = self.mixer(self.norm1(x), state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), state

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        mixed, state = self.mixer.step(self.norm1(x), state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), state


class AttentionARBlock(nn.Module):
    def __init__(self, config: AutoregressiveConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.mixer = AttentionMixer(config.dim, config.heads)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config.dim, config.mlp_mult)

    def prefill(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mixed, key, value = self.mixer(self.norm1(x), causal=True)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), key, value

    def step(
        self,
        x: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        position: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mixed, key_cache, value_cache = self.mixer.step(
            self.norm1(x), key_cache, value_cache, position
        )
        x = x + mixed
        return x + self.mlp(self.norm2(x)), key_cache, value_cache


class ESAARModel(nn.Module):
    def __init__(self, config: AutoregressiveConfig | None = None):
        super().__init__()
        config = config or AutoregressiveConfig()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(ESAARBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return causal next-token logits for every input position.

        The output shape is ``[batch, tokens, vocab_size]`` and is suitable for
        standard teacher-forced autoregressive training.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have non-empty shape [batch, tokens]")
        positions = sinusoidal_positions(
            input_ids.shape[1],
            self.config.dim,
            device=input_ids.device,
            dtype=self.embedding.weight.dtype,
        )
        x = self.embedding(input_ids) + positions.unsqueeze(0)
        for block in self.blocks:
            x, _ = block.prefill(x, None)
        return self.lm_head(self.final_norm(x))

    def prefill(self, prompt_ids: Tensor) -> tuple[Tensor, list[Tensor]]:
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have non-empty shape [batch, tokens]")
        positions = sinusoidal_positions(
            prompt_ids.shape[1],
            self.config.dim,
            device=prompt_ids.device,
            dtype=self.embedding.weight.dtype,
        )
        x = self.embedding(prompt_ids) + positions.unsqueeze(0)
        states: list[Tensor] = []
        for block in self.blocks:
            x, state = block.prefill(x, None)
            states.append(state)
        logits = self.lm_head(self.final_norm(x[:, -1, :]))
        return logits, states

    def decode_step(
        self,
        token_ids: Tensor,
        position: int,
        states: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must have shape [batch]")
        if len(states) != len(self.blocks):
            raise ValueError("one recurrent state is required for every block")
        position_embedding = sinusoidal_positions(
            position + 1,
            self.config.dim,
            device=token_ids.device,
            dtype=self.embedding.weight.dtype,
        )[-1]
        x = self.embedding(token_ids) + position_embedding.unsqueeze(0)
        new_states: list[Tensor] = []
        for block, state in zip(self.blocks, states, strict=True):
            x, state = block.step(x, state)
            new_states.append(state)
        return self.lm_head(self.final_norm(x)), new_states

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, generated_tokens: int) -> Tensor:
        if generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have non-empty shape [batch, tokens]")
        if generated_tokens == 0:
            return prompt_ids.new_empty((prompt_ids.shape[0], 0))
        logits, states = self.prefill(prompt_ids)
        token = logits.argmax(dim=-1)
        generated = [token]
        for offset in range(generated_tokens - 1):
            token, states = self.decode_step(token, prompt_ids.shape[1] + offset, states)
            token = token.argmax(dim=-1)
            generated.append(token)
        return torch.stack(generated, dim=1)


class AttentionARModel(nn.Module):
    def __init__(self, config: AutoregressiveConfig | None = None):
        super().__init__()
        config = config or AutoregressiveConfig()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList(AttentionARBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return causal next-token logits for every input position.

        The output shape is ``[batch, tokens, vocab_size]`` and mirrors
        :class:`ESAARModel` for matched training comparisons.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have non-empty shape [batch, tokens]")
        positions = sinusoidal_positions(
            input_ids.shape[1],
            self.config.dim,
            device=input_ids.device,
            dtype=self.embedding.weight.dtype,
        )
        x = self.embedding(input_ids) + positions.unsqueeze(0)
        for block in self.blocks:
            x, _, _ = block.prefill(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, generated_tokens: int) -> Tensor:
        if generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
            raise ValueError("prompt_ids must have non-empty shape [batch, tokens]")
        if generated_tokens == 0:
            return prompt_ids.new_empty((prompt_ids.shape[0], 0))

        batch, prompt_length = prompt_ids.shape
        cache_length = prompt_length + generated_tokens - 1
        positions = sinusoidal_positions(
            prompt_length + generated_tokens,
            self.config.dim,
            device=prompt_ids.device,
            dtype=self.embedding.weight.dtype,
        )
        x = self.embedding(prompt_ids) + positions[:prompt_length].unsqueeze(0)
        key_caches: list[Tensor] = []
        value_caches: list[Tensor] = []

        for block in self.blocks:
            x, prompt_key, prompt_value = block.prefill(x)
            key_cache = x.new_zeros(
                batch,
                self.config.heads,
                cache_length,
                self.config.dim // self.config.heads,
            )
            value_cache = torch.zeros_like(key_cache)
            key_cache[:, :, :prompt_length, :] = prompt_key
            value_cache[:, :, :prompt_length, :] = prompt_value
            key_caches.append(key_cache)
            value_caches.append(value_cache)

        token = self.lm_head(self.final_norm(x[:, -1, :])).argmax(dim=-1)
        generated = [token]
        for offset in range(generated_tokens - 1):
            position = prompt_length + offset
            hidden = self.embedding(token) + positions[position].unsqueeze(0)
            new_keys: list[Tensor] = []
            new_values: list[Tensor] = []
            for block, key_cache, value_cache in zip(
                self.blocks, key_caches, value_caches, strict=True
            ):
                hidden, key_cache, value_cache = block.step(
                    hidden, key_cache, value_cache, position
                )
                new_keys.append(key_cache)
                new_values.append(value_cache)
            key_caches, value_caches = new_keys, new_values
            token = self.lm_head(self.final_norm(hidden)).argmax(dim=-1)
            generated.append(token)
        return torch.stack(generated, dim=1)
