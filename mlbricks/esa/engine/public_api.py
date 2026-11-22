"""Unified, checkpoint-compatible MLBricks ESA public API.

The public ESA class constructs a concrete half-life or learned-bounds module.
No submodule wrapper is introduced, so state_dict keys remain unchanged.
Backends select *execution*, not architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core import ESA as _HalfLifeESA
from .core import BaseESA as _LearnedBoundsESA
from .core import normalize_variant

_BACKENDS = frozenset(("auto", "native", "pytorch"))


def _check_backend(backend: str) -> str:
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError("backend must be 'auto', 'native', or 'pytorch'")
    return backend


def _modes(backend: str) -> tuple[str, str]:
    """Return (scan, fused_decode) while retaining exact ESA math."""
    return {
        "auto": ("auto", "auto"),
        "native": ("parallel", "on"),
        "pytorch": ("reference", "off"),
    }[_check_backend(backend)]


class ESA(nn.Module):
    """Single public constructor for current and historical ESA variants.

    The object returned is an ``isinstance(obj, ESA)`` and is also a concrete
    CurrentESA or BaseESA. No extra ``.impl`` or ``.engine`` parameter prefix.

    ``variant`` defaults to current half-life ESA; ``base`` preserves the
    historical learned-bounds architecture. ``backend`` controls execution.
    """

    def __new__(cls, embd: int = 384, head: int = 6, *,
                variant: str = "current", backend: str = "auto",
                device: str | torch.device | None = None,
                **kwargs: Any):
        if cls is ESA:
            selected = normalize_variant(variant)
            target = CurrentESA if selected == "current" else BaseESA
            return object.__new__(target)
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        # The concrete constructors below call their original model init.
        # Direct instantiation of ESA routes through __new__.
        super().__init__()

    def _setup_api(self, *, embd: int, head: int, variant: str,
                   backend: str, device: str | torch.device | None):
        self.esa_variant = variant
        self.requested_backend = _check_backend(backend)
        self.embd = embd
        self.head = head
        if device is not None:
            self.to(device)

    def set_backend(self, backend: str) -> "ESA":
        """Switch execution policy without changing weights or architecture."""
        backend = _check_backend(backend)
        scan, fused = _modes(backend)
        self.requested_backend = backend
        self.scan = scan
        if hasattr(self, "fused_decode"):
            self.fused_decode = fused
            self._fused_unavailable = False
        return self

    def resolved_backend(self) -> str:
        """Report policy, not an unverified native-performance claim.

        Auto dispatch is shape/device-dependent and may fall back at runtime;
        returning 'auto' rather than 'native' avoids pretending a kernel ran.
        """
        return self.requested_backend

    def _guard_native(self, x: torch.Tensor) -> None:
        if self.requested_backend != "native":
            return
        if not x.is_cuda:
            raise RuntimeError("backend='native' requires a CUDA input; use backend='auto' on CPU")
        if x.ndim == 3 and x.size(1) == 1:
            if self.esa_variant == "base":
                raise RuntimeError("Native Base ESA one-token decode kernel is unavailable; use backend='auto'")
            if torch.is_grad_enabled():
                raise RuntimeError("Native ESA single-token decode requires torch.no_grad() or torch.inference_mode()")

    def prefill(self, x: torch.Tensor, state: torch.Tensor | None = None):
        return self.forward(x, state)

    def decode_step(self, x: torch.Tensor, state: torch.Tensor | None = None):
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("decode_step expects [batch, dim] or [batch, 1, dim]")
        return self.forward(x, state)


class CurrentESA(ESA, _HalfLifeESA):
    """Current half-life ESA with original checkpoint parameter names."""

    def __init__(self, embd: int = 384, head: int = 6, *,
                 variant: str = "current", backend: str = "auto",
                 device: str | torch.device | None = None,
                 chunk_size: int = 32, **kwargs: Any):
        if normalize_variant(variant) != "current":
            raise ValueError("CurrentESA requires variant='current'")
        scan, fused = _modes(backend)
        _HalfLifeESA.__init__(self, dim=embd, heads=head, chunk_size=chunk_size,
                              scan=scan, fused_decode=fused, **kwargs)
        self._setup_api(embd=embd, head=head, variant="current",
                        backend=backend, device=device)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        self._guard_native(x)
        return _HalfLifeESA.forward(self, x, state)


class BaseESA(ESA, _LearnedBoundsESA):
    """Historical learned-bounds ESA. Original learned variant is retained."""

    def __init__(self, embd: int = 384, head: int = 6, *,
                 variant: str = "base", backend: str = "auto",
                 device: str | torch.device | None = None,
                 **kwargs: Any):
        if normalize_variant(variant) != "base":
            raise ValueError("BaseESA requires variant='base'")
        scan, _ = _modes(backend)
        _LearnedBoundsESA.__init__(self, dim=embd, heads=head, scan=scan, **kwargs)
        # Keep self.variant == 'learned' for the legacy reference forward!
        self._setup_api(embd=embd, head=head, variant="base",
                        backend=backend, device=device)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        self._guard_native(x)
        return _LearnedBoundsESA.forward(self, x, state)


def detect_esa_variant(state: dict[str, torch.Tensor]) -> str:
    """Identify a mixer architecture from its unique retention parameters."""
    if not isinstance(state, dict):
        raise TypeError("state must be a state_dict mapping")
    has_current = "log_half_life" in state
    has_base = "raw_alpha" in state
    if has_current == has_base:
        raise ValueError("Cannot determine ESA architecture: expected exactly one of log_half_life or raw_alpha")
    return "current" if has_current else "base"


def save_esa_checkpoint(model: ESA, path: str | Path, *, extra: dict | None = None) -> None:
    """Store architecture metadata with weights, preserving original key names."""
    if not isinstance(model, ESA):
        raise TypeError("Expected an ESA or BaseESA public API model")
    payload = {
        "format": "mlbricks.esa",
        "format_version": 1,
        "architecture": model.esa_variant,
        "config": {"embd": model.embd, "head": model.head, "chunk_size": getattr(model, "chunk_size", 32)},
        "state_dict": model.state_dict(),
    }
    if extra is not None:
        if not isinstance(extra, dict):
            raise TypeError("extra must be a dictionary")
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_esa_checkpoint(path: str | Path, *, variant: str | None = None,
                        backend: str = "auto", device: str | torch.device = "cpu") -> ESA:
    """Load a saved mixer, including historical weight-only state_dict files.

    Never silently reinterpret old base weights as current half-life ESA.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a mapping")
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict mapping")
    inferred = detect_esa_variant(state)
    saved = payload.get("architecture", payload.get("variant"))
    if saved is not None and normalize_variant(saved) != inferred:
        raise ValueError("Checkpoint metadata does not match stored weights")
    if variant is not None and normalize_variant(variant) != inferred:
        raise ValueError(f"Requested {normalize_variant(variant)!r} but checkpoint is {inferred!r}")
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError("Invalid checkpoint config")
    if "qgv.weight" not in state:
        raise ValueError("Expected ESA mixer state_dict with qgv.weight")
    embd = int(state["qgv.weight"].shape[1])
    head_tensor = state["log_half_life"] if inferred == "current" else state["raw_alpha"]
    head = int(head_tensor.numel())
    if cfg.get("embd", embd) != embd or cfg.get("head", head) != head:
        raise ValueError("Checkpoint config dimensions disagree with stored weights")
    model = ESA(embd=embd, head=head, variant=inferred, backend=backend,
                chunk_size=int(cfg.get("chunk_size", 32))) if inferred == "current" else ESA(
                embd=embd, head=head, variant=inferred, backend=backend)
    model.load_state_dict(state, strict=True)
    return model.to(device)
