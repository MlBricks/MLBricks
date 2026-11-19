"""User-supplied, token-local Q/K transform. BOLT has no built-in positional encoding."""
from __future__ import annotations


def resolve_qk_transform(transform):
    if transform is None:
        return None
    if isinstance(transform, str) and transform.strip().lower() in ("none", "off"):
        return None
    if isinstance(transform, str):
        raise ValueError("BOLT does not provide positional encodings; supply a callable Q/K transform")
    if not callable(transform):
        raise TypeError("Q/K transform must be callable or None")
    return transform
