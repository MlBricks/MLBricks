# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""PyTorch-native building blocks exposed through the MLBricks API.

These classes intentionally delegate standard neural-network computation to
PyTorch. They provide a compact MLBricks import surface without maintaining
separate native kernels for operations already optimized by PyTorch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationLike = str | Callable[[torch.Tensor], torch.Tensor]


def _resolve_activation(activation: ActivationLike) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(activation):
        return activation

    name = str(activation).strip().lower().replace("-", "_")
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": F.gelu,
        "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
        "relu": F.relu,
        "silu": F.silu,
        "swish": F.silu,
        "tanh": torch.tanh,
    }
    try:
        return activations[name]
    except KeyError as exc:
        supported = ", ".join(sorted(activations))
        raise ValueError(
            f"Unsupported activation {activation!r}. Choose one of: {supported}, "
            "or pass a callable."
        ) from exc


class Linear(nn.Linear):
    """A thin, fully PyTorch-backed linear layer.

    This class exists to provide ``from mlbricks import Linear`` while keeping
    the exact execution, autograd, serialization, device, and dtype behavior of
    :class:`torch.nn.Linear`.
    """


class Embedding(nn.Embedding):
    """A PyTorch embedding table available from the MLBricks namespace.

    Parameters use model-oriented names while all additional keyword arguments
    are forwarded to :class:`torch.nn.Embedding`.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        **kwargs: Any,
    ) -> None:
        if int(vocab_size) <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if int(embedding_dim) <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}."
            )
        super().__init__(
            num_embeddings=int(vocab_size),
            embedding_dim=int(embedding_dim),
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self.num_embeddings

    @property
    def hidden_size(self) -> int:
        return self.embedding_dim


class LMHead(nn.Linear):
    """PyTorch linear output projection for language-model logits.

    ``tie_to`` may be an :class:`Embedding`, a standard PyTorch embedding, or
    any module exposing a compatible ``weight`` parameter.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        bias: bool = False,
        tie_to: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if int(hidden_size) <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if int(vocab_size) <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

        super().__init__(
            in_features=int(hidden_size),
            out_features=int(vocab_size),
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if tie_to is not None:
            self.tie_weights(tie_to)

    @property
    def hidden_size(self) -> int:
        return self.in_features

    @property
    def vocab_size(self) -> int:
        return self.out_features

    def tie_weights(self, embedding: nn.Module) -> "LMHead":
        weight = getattr(embedding, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise TypeError("tie_weights expects a module with a Parameter named 'weight'.")
        expected = (self.vocab_size, self.hidden_size)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Cannot tie weight with shape {tuple(weight.shape)}; expected {expected}."
            )
        self.weight = weight
        return self


class FFN(nn.Module):
    """PyTorch-native feed-forward network.

    The default path is the familiar ``Linear -> GELU -> Linear`` MLP used by
    the ESA model. Set ``gated=True`` with ``activation='silu'`` for a SwiGLU-
    style feed-forward block. No custom C++/CUDA FFN kernel is used.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        *,
        activation: ActivationLike = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        hidden_size = int(hidden_size)
        intermediate_size = (
            4 * hidden_size if intermediate_size is None else int(intermediate_size)
        )
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {intermediate_size}."
            )
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_name = (
            activation if isinstance(activation, str) else "callable"
        )
        self.activation = _resolve_activation(activation)
        self.gated = bool(gated)

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.gated:
            self.gate_proj = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=bias,
                **factory_kwargs,
            )
            self.up_proj = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=bias,
                **factory_kwargs,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                bias=bias,
                **factory_kwargs,
            )
        else:
            # Keep these names compatible with the existing ESAModel checkpoints.
            self.fc = nn.Linear(
                hidden_size,
                intermediate_size,
                bias=bias,
                **factory_kwargs,
            )
            self.proj = nn.Linear(
                intermediate_size,
                hidden_size,
                bias=bias,
                **factory_kwargs,
            )

        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            hidden = self.activation(self.gate_proj(x)) * self.up_proj(x)
            return self.drop(self.down_proj(hidden))
        return self.drop(self.proj(self.activation(self.fc(x))))


class LayerNorm(nn.Module):
    """PyTorch functional LayerNorm with optional bias across Torch versions."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(int(value) for value in normalized_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"normalized_shape must be positive, got {shape}.")

        self.normalized_shape = shape
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)

        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(shape, device=device, dtype=dtype)
            )
            self.bias = (
                nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
                if bias
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )


class RMSNorm(nn.Module):
    """Portable PyTorch RMS normalization with optional affine weight."""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            shape = (normalized_shape,)
        else:
            shape = tuple(int(value) for value in normalized_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"normalized_shape must be positive, got {shape}.")

        self.normalized_shape = shape
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        variance = x.to(torch.float32).pow(2).mean(dim=dims, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        output = x * inv_rms
        if self.weight is not None:
            output = output * self.weight
        return output


class Residual(nn.Module):
    """Add a residual update, optionally applying PyTorch dropout first."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        if x.shape != update.shape:
            raise ValueError(
                f"Residual tensors must have identical shapes, got "
                f"{tuple(x.shape)} and {tuple(update.shape)}."
            )
        return x + self.drop(update)


# Friendly aliases requested for compact imports. Class-style names are the
# canonical public API; lowercase aliases remain convenient and explicit.
ffn = FFN
embedding = Embedding
embeddings = Embedding
lmhead = LMHead
linear = Linear
layernorm = LayerNorm
rmsnorm = RMSNorm
residual = Residual


__all__ = [
    "FFN",
    "Embedding",
    "LMHead",
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "Residual",
    "ffn",
    "embedding",
    "embeddings",
    "lmhead",
    "linear",
    "layernorm",
    "rmsnorm",
    "residual",
]
