"""Current half-life ESA from the independently benchmarked FastESA 0.4.0.

This module hosts the exact FastESA implementation, with directly initialized,
learnable FP32 half-lives. The public ``mlbricks.ESA`` selector defaults to this
retention engine. Internal historical wrappers keep the legacy Thunder engine.
"""
from __future__ import annotations

from .engine import (
    CurrentESA, BaseESA, ESAConfig as CurrentESAConfig,
    save_esa_checkpoint as save_current_esa_checkpoint,
    load_esa_checkpoint as load_current_esa_checkpoint,
    detect_esa_variant as detect_current_esa_variant,
    GraphDecoder as CurrentESAGraphDecoder,
)
from .engine.core import (
    DEFAULT_CURRENT_HALF_LIVES,
    ESAModel as CurrentESAModel,
    BaseESAModel as BaseESAModel,
    make_model as _make_current_esa_model,
    forward_tokens as current_esa_forward_tokens,
    load_checkpoint as load_current_esa_model_checkpoint,
)

def make_current_esa_model(*, vocab_size: int = 256, dim: int = 384,
                           heads: int = 6, layers: int = 6,
                           seed: int | None = None, **kwargs):
    """Build a fresh current half-life language model (not pretrained)."""
    return _make_current_esa_model('current', vocab_size=vocab_size,
                                   dim=dim, heads=heads, layers=layers,
                                   seed=seed, **kwargs)

def make_esa(*, architecture: str = 'retention', embd: int = 384,
             head: int = 6, backend: str = 'auto', device=None, **kwargs):
    """Select retention (default) or legacy ESA; older aliases are supported."""
    key = str(architecture).strip().lower().replace('-', '_')
    if key in {'retention', 'current', 'half_life', 'halflife', 'fastesa'}:
        return CurrentESA(embd=embd, head=head, backend=backend,
                          device=device, **kwargs)
    if key in {'legacy', 'thunder', 'mlbricks'}:
        from .layer import ESA as LegacyESA
        return LegacyESA(embd=embd, head=head, backend=backend,
                   device=device, **kwargs)
    raise ValueError("architecture must be 'retention' or 'legacy'")

__all__ = [
    'CurrentESA', 'BaseESA', 'CurrentESAConfig', 'CurrentESAModel',
    'BaseESAModel', 'DEFAULT_CURRENT_HALF_LIVES',
    'save_current_esa_checkpoint', 'load_current_esa_checkpoint',
    'load_current_esa_model_checkpoint', 'detect_current_esa_variant',
    'current_esa_forward_tokens', 'make_current_esa_model', 'make_esa',
    'CurrentESAGraphDecoder',
]
