"""Optional user-provided, pointwise Q/K transformation interface for BOLT.

BOLT does not construct, own, or select positional encodings. An external
callable must accept a [B,H,T,D] tensor and keyword start_pos and return a
same-shaped tensor. It must act consistently on tokens in prefill and decode.
Transform modules can be supplied directly so their parameters are registered.
"""
from __future__ import annotations


def resolve_qk_transform(transform):
    if transform is None:
        return None
    if isinstance(transform, str):
        if transform.strip().lower() in ("none", "off"):
            return None
        raise ValueError(
            "BOLT no longer provides built-in positional encodings. "
            "Pass an external Q/K transform callable, or None."
        )
    if not callable(transform):
        raise TypeError("Q/K transform must be callable or None")
    return transform
