"""BOLT architecture and runtime configuration.

All active model settings live in the BOLT namespace.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..runtime import normalize_backend


@dataclass
class BoltModelSettings:
    vocab_size: int
    context: int = 2048
    layers: int = 6
    dim: int = 384
    heads: int = 6
    latent_dim: int = 32  # legacy-only when bolt_mode='legacy'
    bolt_mode: str = "quality"  # quality (matched-width AVQ), fast (BOLT), legacy
    bolt_cache: str = "normal"  # normal=32, tight=16
    avq_dim: int | None = None
    position: str = "none"
    ffn: str = "standard"
    residual: str = "standard"
    norm: str = "rmsnorm"
    activation: str = "gelu"
    ffn_mult: int = 4
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    backend: str = "auto"
    eps: float = 1e-6
    ffn_state_dim: int = 256
    ffn_depth_embedding_dim: int = 64
    ffn_virtual_refinements: int = 2
    ffn_virtual_hidden_dim: int = 128
    ffn_micro_hidden_dim: int = 64
    ffn_micro_refinements: int = 1
    residual_update_ratio: float = 0.18
    residual_stream_ratio: float = 1.08
    residual_update_softness: float = 8.0
    residual_stream_softness: float = 8.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "context", "layers", "dim", "heads", "latent_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        self.backend = normalize_backend(self.backend, warn_legacy=True)
        self.bolt_mode = str(self.bolt_mode).strip().lower()
        self.bolt_cache = str(self.bolt_cache).strip().lower()
        if self.bolt_mode not in ("quality", "fast", "legacy"):
            raise ValueError("bolt_mode must be quality, fast or legacy")
        if self.bolt_cache not in ("normal", "tight"):
            raise ValueError("bolt_cache must be normal or tight")
        # Default AVQ memory to the complete internal attention representation.
        # Explicit nonmatching dimensions are retained only for ablation studies.
        if self.avq_dim is None:
            cache_width = 16 if self.bolt_cache == "tight" else 32
            self.avq_dim = int(self.heads) * cache_width
        if int(self.avq_dim) < 1:
            raise ValueError("avq_dim must be positive")
        self.ffn = str(self.ffn).strip().lower().replace("-", "_")
        aliases = {
            "default": "standard", "ffn": "standard", "standard": "standard",
            "saffn": "saffn", "stateaware": "saffn", "state_aware": "saffn", "ffnbrick": "saffn",
            "virtual": "virtual", "virtual_state_aware": "virtual", "virtual_ffnbrick": "virtual",
            "micro": "micro", "micro_virtual": "micro", "micro_ffnbrick": "micro",
        }
        if self.ffn not in aliases:
            raise ValueError("ffn must be standard, saffn, virtual, or micro")
        self.ffn = aliases[self.ffn]
        self.residual = str(self.residual).strip().lower().replace("-", "_")
        if self.residual in {"default", "residual"}:
            self.residual = "standard"
        if self.residual in {"controller", "res_controller"}:
            self.residual = "rescontroller"
        if self.residual not in {"standard", "rescontroller"}:
            raise ValueError("residual must be standard or rescontroller")
        self.norm = str(self.norm).strip().lower().replace("-", "_")
        if self.norm not in {"rmsnorm", "rms", "layernorm", "layer_norm"}:
            raise ValueError("norm must be rmsnorm or layernorm")

    # ESA-style compatibility names.
    @property
    def block(self) -> int:
        return self.context

    @property
    def n_layer(self) -> int:
        return self.layers

    @property
    def embd(self) -> int:
        return self.dim

    @property
    def head(self) -> int:
        return self.heads


@dataclass
class BoltConfig(BoltModelSettings):
    """Configuration for :class:`BoltModel`.

    ``profile`` selects the per-head BOLT latent width (32 or 16).
    By default AVQ width equals ``heads * latent_width``, so the memory
    return is Identity, without a learned return projection.
    """
    heads: int = 12  # validated BOLT attention layout by default
    avq_dim: int | None = None  # default = heads * (16 compact or 32 standard)
    profile: str | None = None  # None keeps legacy `bolt_cache` keywords usable
    # Enabled by default in new BOLT models. Setting False constructs an
    # attention-only model with no AVQ modules, parameters, or recurrent cache.
    state_memory: bool = True
    state_dim: int | None = None
    device: str = 'auto'
    precision: str = 'auto'
    attention_scale: str | None = None
    memory_init: str | None = None
    projection: str = 'packed'
    avq_return_map: str = 'auto'
    avq_norm_eps: float | None = None

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
        elif self.avq_dim is None:
            self.avq_dim = int(self.heads) * (16 if self.profile == 'compact' else 32)
        compact_192 = self.profile == 'compact' and self.heads == 12 and self.avq_dim == 192
        # None means omitted: explicit older settings must remain intact.
        if self.attention_scale is None:
            self.attention_scale = 'head' if compact_192 else 'latent'
        if self.memory_init is None:
            self.memory_init = 'legacy' if compact_192 else 'balanced'
        if self.avq_return_map not in ('auto', 'linear', 'identity'):
            raise ValueError('avq_return_map must be auto, linear or identity')
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
