"""Shared image-engine utilities for VESA and VisionBolt.

The vision API intentionally separates *engine* (how spatial structure is
presented to a mixer) from *mixer* (ESA or Bolt).
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


_ENGINE_ALIASES = {
    "serpentine": "serpentine",
    "snake": "serpentine",
    "scan": "serpentine",
    "default": "serpentine",
    "vit": "vit",
    "visiontransformer": "vit",
    "vision_transformer": "vit",
    "vision transformer": "vit",
    "cnn": "cnn",
    "conv": "cnn",
    "convolution": "cnn",
    "diffusion": "diffusion",
    "dit": "diffusion",
    "ar": "ar",
    "autoregressive": "ar",
    "auto_regressive": "ar",
}

_SCAN_ALIASES = {
    "cross": "cross",
    "cross_serpentine": "cross",
    "cross-serpentine": "cross",
    "horizontal": "horizontal",
    "h": "horizontal",
    "vertical": "vertical",
    "v": "vertical",
    "raster": "raster",
    "row": "raster",
    "row_major": "raster",
}

_POSITION_ALIASES = {
    "auto": "auto",
    "2d_sincos": "2d_sincos",
    "2d-sincos": "2d_sincos",
    "2dsincos": "2d_sincos",
    "sincos": "2d_sincos",
    "sinusoidal": "2d_sincos",
    "learned": "learned",
    "learned2d": "learned",
    "learned_2d": "learned",
    "none": None,
    "off": None,
    "disabled": None,
}

CLASSIFIER_ENGINES = {"serpentine", "vit", "cnn"}
ALL_ENGINES = CLASSIFIER_ENGINES | {"diffusion", "ar"}


def canonical_engine(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    compact = name.replace(" ", "")
    if name in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[name]
    if compact in _ENGINE_ALIASES:
        return _ENGINE_ALIASES[compact]
    raise ValueError(
        "engine must be one of: Serpentine, ViT, CNN, Diffusion, AR"
    )


def canonical_scan(value: str) -> str:
    name = str(value).strip().lower().replace("-", "_")
    if name not in _SCAN_ALIASES:
        raise ValueError("scan must be one of: cross, horizontal, vertical, raster")
    return _SCAN_ALIASES[name]


def canonical_position(value, *, engine: str):
    """Normalize a spatial position policy.

    ``None`` always means no positional representation. ``"auto"`` resolves
    to no position for scan/CNN/diffusion/AR engines and to 2-D sin/cos for
    the ViT engine.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("position must be None or a string policy")
    key = value.strip().lower().replace("-", "_")
    if key not in _POSITION_ALIASES:
        raise ValueError("position must be one of: auto, None, 2d_sincos, learned")
    normalized = _POSITION_ALIASES[key]
    if normalized == "auto":
        return "2d_sincos" if engine == "vit" else None
    return normalized


def _horizontal_serpentine(height: int, width: int, device: torch.device) -> Tensor:
    rows = []
    for row in range(height):
        start = row * width
        indices = torch.arange(start, start + width, device=device)
        rows.append(indices if row % 2 == 0 else indices.flip(0))
    return torch.cat(rows)


def _vertical_serpentine(height: int, width: int, device: torch.device) -> Tensor:
    columns = []
    for col in range(width):
        indices = torch.arange(col, height * width, width, device=device)
        columns.append(indices if col % 2 == 0 else indices.flip(0))
    return torch.cat(columns)


def scan_indices(
    height: int,
    width: int,
    *,
    scan: str,
    layer_index: int,
    device: torch.device,
) -> Tensor:
    """Return a deterministic directional image traversal.

    ``cross`` cycles H-forward, H-reverse, V-forward, V-reverse. This gives
    recurrent/directional mixers access to both image axes without adding a
    spatial positional embedding.
    """
    scan = canonical_scan(scan)
    base_raster = torch.arange(height * width, device=device)
    h = _horizontal_serpentine(height, width, device)
    v = _vertical_serpentine(height, width, device)

    if scan == "raster":
        return base_raster if layer_index % 2 == 0 else base_raster.flip(0)
    if scan == "horizontal":
        return h if layer_index % 2 == 0 else h.flip(0)
    if scan == "vertical":
        return v if layer_index % 2 == 0 else v.flip(0)

    phase = layer_index % 4
    if phase == 0:
        return h
    if phase == 1:
        return h.flip(0)
    if phase == 2:
        return v
    return v.flip(0)


def sinusoidal_2d_positions(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return [H*W, D] deterministic 2-D sine/cosine positions.

    The implementation supports arbitrary positive ``dim`` by splitting the
    channels between Y and X and padding any odd remainder.
    """
    if height <= 0 or width <= 0 or dim <= 0:
        raise ValueError("height, width, and dim must be positive")

    def encode_1d(length: int, channels: int) -> Tensor:
        if channels <= 0:
            return torch.empty(length, 0, device=device, dtype=dtype)
        pairs = channels // 2
        if pairs == 0:
            # A single channel still carries a monotonic bounded coordinate.
            pos = torch.arange(length, device=device, dtype=torch.float32)
            denom = max(length - 1, 1)
            return (pos / denom).to(dtype).unsqueeze(1)
        idx = torch.arange(pairs, device=device, dtype=torch.float32)
        inv = torch.exp(-math.log(10000.0) * idx / max(pairs, 1))
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        angles = pos * inv[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < channels:
            pad = torch.zeros(length, channels - emb.shape[1], device=device)
            emb = torch.cat([emb, pad], dim=1)
        return emb[:, :channels].to(dtype)

    ydim = dim // 2
    xdim = dim - ydim
    y = encode_1d(height, ydim)
    x = encode_1d(width, xdim)
    yy = y[:, None, :].expand(height, width, ydim)
    xx = x[None, :, :].expand(height, width, xdim)
    return torch.cat([yy, xx], dim=-1).reshape(height * width, dim)



def apply_scan_native_or_pytorch(
    x: Tensor,
    height: int,
    width: int,
    *,
    scan: str,
    layer_index: int,
    backend: str = "auto",
) -> Tensor:
    """Apply a directional scan using the compiled vision op when available."""
    try:
        from .vision_native import reorder as native_reorder
        out = native_reorder(
            x, height, width, scan=canonical_scan(scan),
            layer_index=layer_index, backend=backend, inverse=False,
        )
        if out is not None:
            return out
    except RuntimeError:
        if str(backend).strip().lower() == "native":
            raise
    order = scan_indices(
        height, width, scan=scan, layer_index=layer_index, device=x.device
    )
    return x.index_select(1, order)


def restore_scan_native_or_pytorch(
    x: Tensor,
    height: int,
    width: int,
    *,
    scan: str,
    layer_index: int,
    backend: str = "auto",
) -> Tensor:
    """Restore directional-scan tokens to canonical row-major order."""
    try:
        from .vision_native import reorder as native_reorder
        out = native_reorder(
            x, height, width, scan=canonical_scan(scan),
            layer_index=layer_index, backend=backend, inverse=True,
        )
        if out is not None:
            return out
    except RuntimeError:
        if str(backend).strip().lower() == "native":
            raise
    order = scan_indices(
        height, width, scan=scan, layer_index=layer_index, device=x.device
    )
    return x.index_select(1, torch.argsort(order))


def add_sinusoidal_2d_native_or_pytorch(
    x: Tensor, height: int, width: int, *, backend: str = "auto"
) -> Tensor:
    """Add deterministic 2-D sin/cos positions using native CUDA/C++ if available."""
    try:
        from .vision_native import add_sincos2d as native_add
        out = native_add(x, height, width, backend=backend)
        if out is not None:
            return out
    except RuntimeError:
        if str(backend).strip().lower() == "native":
            raise
    pos = sinusoidal_2d_positions(
        height, width, x.shape[-1], device=x.device, dtype=x.dtype
    )
    return x + pos.unsqueeze(0)

def inverse_permutation(order: Tensor) -> Tensor:
    return torch.argsort(order)


def apply_order(x: Tensor, order: Tensor) -> Tensor:
    return x.index_select(1, order)


def restore_order(x: Tensor, order: Tensor) -> Tensor:
    return x.index_select(1, inverse_permutation(order))


__all__ = [
    "ALL_ENGINES",
    "CLASSIFIER_ENGINES",
    "canonical_engine",
    "canonical_scan",
    "canonical_position",
    "scan_indices",
    "sinusoidal_2d_positions",
    "apply_scan_native_or_pytorch",
    "restore_scan_native_or_pytorch",
    "add_sinusoidal_2d_native_or_pytorch",
    "inverse_permutation",
    "apply_order",
    "restore_order",
]
