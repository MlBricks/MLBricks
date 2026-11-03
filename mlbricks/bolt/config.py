"""Professional, checkpoint-explicit BOLT language-model configuration."""
from __future__ import annotations
from dataclasses import dataclass
from ..gaussian import GaussianConfig


@dataclass
class BoltConfig(GaussianConfig):
    """Configuration for :class:`BoltModel`.

    ``profile`` selects BOLT latent width. New models default to standard
    (32) and the quality-selected memory initialization. Legacy ``GaussianConfig``
    checkpoints should continue to load through ``Gaussian`` unchanged.
    """
    heads: int = 12  # validated BOLT attention layout by default
    profile: str | None = None  # None keeps legacy `bolt_cache` keywords usable
    # Enabled by default in new BOLT models. Setting False constructs an
    # attention-only model with no AVQ modules, parameters, or recurrent cache.
    state_memory: bool = True
    state_dim: int | None = None
    device: str = 'auto'
    precision: str = 'auto'
    attention_scale: str = 'latent'
    memory_init: str = 'balanced'
    projection: str = 'packed'

    def __post_init__(self):
        if self.profile is None:
            self.profile = 'compact' if self.bolt_cache == 'tight' else 'standard'
        if self.profile not in ('standard', 'compact'):
            raise ValueError('profile must be standard or compact')
        if not isinstance(self.state_memory, bool):
            raise TypeError('state_memory must be True or False')
        # Historical callers can still use bolt_mode='fast' / 'legacy'.
        # A False state_memory flag always disables AVQ for modern BOLT.
        # If a legacy mode is provided, it retains its historical meaning.
        if self.bolt_mode == 'fast':
            self.state_memory = False
        elif self.bolt_mode == 'legacy':
            self.state_memory = False
        else:
            self.bolt_mode = 'quality' if self.state_memory else 'fast'
        if self.state_dim is not None:
            self.avq_dim = int(self.state_dim)
        if self.bolt_mode != 'legacy' and self.latent_dim in (16,32):
            self.bolt_cache = 'normal' if self.profile == 'standard' else 'tight'
            self.latent_dim = 32 if self.profile == 'standard' else 16
        if self.attention_scale not in ('head', 'latent'):
            raise ValueError('attention_scale must be head or latent')
        if self.memory_init not in ('legacy', 'balanced'):
            raise ValueError('memory_init must be legacy or balanced')
        if self.projection not in ('packed', 'separate'):
            raise ValueError('projection must be packed or separate')
        self.precision = str(self.precision).lower()
        if self.precision not in ('auto','fp16','bf16','fp32'):
            raise ValueError('precision must be auto, fp16, bf16, fp32')
        super().__post_init__()
        self.state_dim = self.avq_dim
        self.state_memory = self.bolt_mode == 'quality'
