# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from ..runtime import normalize_backend


def _public_backend(backend: str) -> str:
    """Expose 'cuda' as an ESA backend without changing other MLBricks modules."""
    if isinstance(backend, str) and backend.strip().lower() == "cuda":
        return "cuda"
    return normalize_backend(backend, warn_legacy=True)


def _engine_backend(backend: str) -> str:
    """Fast-EsaQ names its strict CUDA kernel execution policy 'native'."""
    return "native" if backend == "cuda" else backend


class ESA(nn.Module):
    """Public Entangled State Attention sequence-mixing layer.

    Uses Fast-EsaQ's half-life gated recurrence exclusively. The MLBricks
    adapter returns a tensor from forward() and exposes FP32 recurrent state
    through prefill(), forward_with_state(), and decode_step().
    """

    def __init__(
        self,
        embd: int,
        head: int = 4,
        batch: int | None = None,
        block: int | None = None,
        backend: str = "auto",
        precision: str = "fp16",
        *,
        compass: int | str | None = "auto",
        dropout: float = 0.0,
        eps: float = 1e-5,
        device: str | torch.device | None = "cuda",
        auto_compile: bool = False,
        compile_mode: str = "default",
        auto_move_input: bool = True,
        strict_checks: bool = False,
        esa_variant: str = "current",
        chunk_size: int = 32,
        cumpass: int | None = None,
        parallel_chunk: int = 128,
        cuda_warps: int = 8,
    ):
        super().__init__()

        if embd <= 0:
            raise ValueError(f"embd must be positive, got {embd}.")
        if head <= 0:
            raise ValueError(f"head must be positive, got {head}.")
        if embd % head != 0:
            raise ValueError(
                f"embd must be divisible by head, got embd={embd}, head={head}."
            )
        if batch is not None and batch <= 0:
            raise ValueError(f"batch must be positive when set, got {batch}.")
        if block is not None and block <= 0:
            raise ValueError(f"block must be positive when set, got {block}.")

        if esa_variant not in {"current", "base"}:
            raise ValueError("esa_variant must be current or base")
        self.esa_variant = esa_variant
        self.chunk_size = int(chunk_size)
        self.embd = int(embd)
        self.head = int(head)
        self.batch = None if batch is None else int(batch)
        self.block = None if block is None else int(block)
        self.backend = _public_backend(backend)
        self.precision = str(precision).lower()
        self.dropout = float(dropout)
        self.eps = float(eps)
        self.auto_move_input = bool(auto_move_input)
        self.strict_checks = bool(strict_checks)
        self.compiled = False
        self.compile_mode: str | None = None

        if compass is None:
            self.compass: int | str = "auto"
        elif isinstance(compass, str):
            if compass.strip().lower() != "auto":
                raise ValueError("compass must be a positive integer or 'auto'.")
            self.compass = "auto"
        else:
            self.compass = int(compass)
            if self.compass <= 0:
                raise ValueError(f"compass must be positive, got {self.compass}.")

        # Fast-EsaQ execution controls. Preserve MLBricks' existing ``compass``
        # API; explicit ``cumpass`` takes priority when provided.
        self.cumpass = (self.compass if isinstance(self.compass, int) else 16) if cumpass is None else cumpass
        self.parallel_chunk = parallel_chunk
        self.cuda_warps = cuda_warps

        # Fast-EsaQ v0.4.0. It returns (y, FP32 recurrent state).
        from .fast_engine.public_api import ESA as FastESA
        self.layer = FastESA(
            embd=self.embd,
            head=self.head,
            variant=self.esa_variant,
            backend=_engine_backend(self.backend),
            cumpass=self.cumpass,
            parallel_chunk=self.parallel_chunk,
            cuda_warps=self.cuda_warps,
            **({"chunk_size": self.chunk_size} if self.esa_variant == "current" else {}),
        )
        self.layer.compass = self.compass

        self.register_buffer("_esa_device_ref", torch.empty(0), persistent=False)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device is not None:
            requested_device = torch.device(device)
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "ESA defaults to device='cuda', but CUDA is not available. "
                    "For CPU use, specify backend='pytorch', device='cpu'."
                )
            self.to(device)
        if auto_compile:
            self.compile(mode=compile_mode)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = _public_backend(backend)
        target = getattr(self.layer, "_orig_mod", self.layer)
        target.set_backend(_engine_backend(self.backend))
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, Mapping):
            raise TypeError("Expected checkpoint mapping")

        keys = set(state_dict)
        old = any(k.endswith("out_proj.weight") for k in keys)
        half = "log_half_life" in keys or "layer.log_half_life" in keys
        base = "raw_alpha" in keys or "layer.raw_alpha" in keys

        if old and not (half or base):
            raise ValueError(
                "Old Thunder ESA weights are incompatible with Fast-EsaQ"
            )

        bare = any(
            k in keys for k in
            ("qgv.weight", "log_half_life", "raw_alpha")
        )

        if bare:
            if any(k.startswith("layer.") for k in keys):
                raise ValueError("Cannot mix bare and prefixed keys")

            from .fast_engine.public_api import detect_esa_variant
            variant = detect_esa_variant(state_dict)

            if variant != self.esa_variant:
                raise ValueError(
                    f"Fast-EsaQ checkpoint variant {variant!r} "
                    f"does not match {self.esa_variant!r}"
                )

            state_dict = OrderedDict(
                ("layer." + key, value)
                for key, value in state_dict.items()
            )
        elif half and self.esa_variant != "current":
            raise ValueError("Half-life checkpoint requires current ESA")
        elif base and self.esa_variant != "base":
            raise ValueError("Base checkpoint requires BaseESA")

        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    def load_fast_esaq_checkpoint(self, path: str | Path, *, strict=True):
        checkpoint = torch.load(
            str(path), map_location="cpu", weights_only=True
        )

        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must be a mapping")

        state = checkpoint.get(
            "state_dict", checkpoint.get("model", checkpoint)
        )
        if not isinstance(state, Mapping):
            raise ValueError("Missing Fast-EsaQ state dictionary")

        label = checkpoint.get(
            "architecture", checkpoint.get("variant")
        )
        if label is not None:
            from .fast_engine.core import normalize_variant
            if normalize_variant(label) != self.esa_variant:
                raise ValueError("Fast-EsaQ checkpoint variant mismatch")

        cfg = checkpoint.get("config", {})
        if cfg is not None:
            if not isinstance(cfg, Mapping):
                raise ValueError("Invalid checkpoint config")
            for key, val in (
                ("embd", self.embd), ("head", self.head)
            ):
                if key in cfg and int(cfg[key]) != val:
                    raise ValueError(
                        f"Fast-EsaQ checkpoint {key} mismatch"
                    )

        return self.load_state_dict(state, strict=strict)

    def resolved_backend(self) -> str:
        if self.backend == "cuda":
            return "cuda"  # Requested strict CUDA policy, not a kernel benchmark.
        return getattr(self.layer, "_orig_mod", self.layer).resolved_backend()

    def compile(self, mode: str = "default"):
        """Compile this ESA layer with torch.compile and return the module.

        The requested compile mode is recorded on the underlying Fast-EsaQ module.
        """
        self.compile_mode = str(mode)
        target = getattr(self.layer, "_orig_mod", self.layer)
        try:
            setattr(target, "_mlbricks_compile_mode", self.compile_mode)
        except Exception:
            pass
        try:
            self.layer = torch.compile(self.layer, mode=mode, fullgraph=False)
            self.compiled = True
        except Exception:
            self.compiled = False
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_move_input:
            target_device = self._esa_device_ref.device
            if x.device != target_device:
                x = x.to(target_device, non_blocking=True)
        if x.dim() != 3:
            raise ValueError(f"ESA input must be [batch, seq, embd], got {tuple(x.shape)}.")
        B, T, C = x.shape
        if C != self.embd:
            raise ValueError(f"ESA expected embd={self.embd}, got {C}.")
        if self.strict_checks:
            if self.batch is not None and B != self.batch:
                raise ValueError(f"ESA expected batch={self.batch}, got {B}.")
            if self.block is not None and T > self.block:
                raise ValueError(f"ESA block limit is {self.block}, got {T}.")
        return x

    def _fast_input_state(self, state, batch, device):
        if state is None:
            return None, False
        flat = state.ndim == 2
        expected = (batch, self.head, self.embd // self.head)
        if flat:
            if tuple(state.shape) != (batch, self.embd):
                raise ValueError(f"Expected flat ESA state {(batch, self.embd)}")
            state = state.reshape(expected)
        elif tuple(state.shape) != expected:
            raise ValueError(f"Expected ESA state {expected}")
        if state.device != device:
            raise ValueError("ESA state and input must be on the same device")
        return state.float(), flat

    def _fast_run(self, x, state=None, *, reverse=False, backend=None):
        x = self._prepare_input(x)
        state, flat = self._fast_input_state(state, x.shape[0], x.device)
        if backend is None or backend == "lightning":
            y, last = self.layer(x.flip(1) if reverse else x, state)
        else:
            selected_engine = _engine_backend(
                _public_backend(backend)
            )
            engine = getattr(
                self.layer, "_orig_mod", self.layer
            )
            old_backend = engine.resolved_backend()

            if selected_engine != old_backend:
                engine.set_backend(selected_engine)
            try:
                y, last = self.layer(
                    x.flip(1) if reverse else x, state
                )
            finally:
                if selected_engine != old_backend:
                    engine.set_backend(old_backend)
        if reverse:
            y = y.flip(1)
        if flat:
            last = last.reshape(x.shape[0], self.embd)
        return y, last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self._fast_run(x)
        return y

    def init_state(self, batch: int, *, device=None, dtype=None, layout: str = "heads") -> torch.Tensor:
        if device is None:
            device = self._esa_device_ref.device
        shape = (batch, self.head, self.embd // self.head) if layout == "heads" else (batch, self.embd) if layout == "flat" else None
        if shape is None:
            raise ValueError("layout must be heads or flat")
        return torch.zeros(shape, dtype=torch.float32 if dtype is None else dtype, device=device)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        backend: str | None = None,
        compass: int | str | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run canonical ESA and return output plus final recurrent state.

        This differentiable interface is intended for compositional wrappers
        such as VESA that need the production ESA engine during both training
        and recurrent inference.
        """
        # Compass remains a performance hint; Fast-EsaQ determines scan
        # execution using its own backend policy.
        return self._fast_run(x, state, reverse=reverse, backend=backend)

    @torch.no_grad()
    def prefill(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        backend: str | None = None,
        compass: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a full sequence and return outputs plus final recurrent state."""
        return self._fast_run(x, state, backend=backend)

    def decode_step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a token with optimized Half-Life Fast-EsaQ dispatch."""

        if (
            self.esa_variant == "current"
            and x.ndim == 2
            and state is not None
            and state.ndim == 3
            and state.dtype == torch.float32
            and x.shape[1] == self.embd
            and state.shape == (x.shape[0], self.head, self.embd // self.head)
            and state.device == x.device
            and (not self.auto_move_input or x.device == self._esa_device_ref.device)
            and (not self.strict_checks or self.batch is None or x.shape[0] == self.batch)
            and not self.compiled
        ):
            y, new_state = self.layer.decode_step(x, state)
            return y[:, 0, :], new_state

        squeezed = x.ndim == 2
        if squeezed:
            x = x.unsqueeze(1)

        if x.ndim != 3 or x.size(1) != 1:
            raise ValueError("decode_step expects [B,C] or [B,1,C]")

        y, new_state = self._fast_run(x, state)

        return (y[:, 0] if squeezed else y), new_state

    lightning_prefill = prefill
    lightning_step = decode_step
