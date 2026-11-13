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
    bolt_mode: str = "quantised"  # default Enhanced MQA; Keyless=BOLT32; base=MHA
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
        modes = {'enhanced_mqa': 'quantised', 'bolt32': 'keyless',
                 'bolt': 'keyless', 'classic': 'base'}
        self.bolt_mode = modes.get(self.bolt_mode, self.bolt_mode)
        if self.bolt_mode not in ('quantised', 'keyless', 'base', 'quality', 'fast', 'legacy'):
            raise ValueError('bolt_mode must be quantised, Keyless, or base')
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
    avq_dim: int | None = None  # default = dim for quantised/base, heads * 32 for Keyless
    bolt_mode: str = 'quantised'
    layer_modes: tuple[str, ...] | None = None
    avq: str | None = None
    forget_gate_init: float = 0.95
    head_output_gate: bool = True
    profile: str | None = None  # None keeps legacy `bolt_cache` keywords usable
    # Enabled by default in new BOLT models. Setting False constructs an
    # attention-only model with no AVQ modules, parameters, or recurrent cache.
    state_memory: bool = True
    state_dim: int | None = None
    device: str = 'auto'
    precision: str = 'fp32'
    attention_scale: str | None = None
    memory_init: str | None = None
    projection: str = 'packed'
    avq_return_map: str = 'auto'
    avq_norm_eps: float | None = None

    def __post_init__(self):
        if not isinstance(self.state_memory, bool):
            raise TypeError('state_memory must be True or False')
        if self.avq is not None:
            if not isinstance(self.avq, str) or self.avq.strip().lower() not in ('enable', 'disable'):
                raise ValueError('avq must be "enable" or "disable"')
            self.avq = self.avq.strip().lower()
            self.state_memory = self.avq == 'enable'
        aliases = {'enhanced_mqa':'quantised', 'classic':'base',
                   'bolt':'keyless', 'bolt32':'keyless',
                   'quality':'keyless', 'fast':'keyless'}
        original_mode = str(self.bolt_mode).strip().lower()
        self.bolt_mode = aliases.get(original_mode, original_mode)
        if original_mode in ('fast', 'legacy') and self.avq is None:
            self.state_memory = False
        if self.bolt_mode not in ('quantised','keyless','base','legacy'):
            raise ValueError('bolt_mode must be quantised, Keyless, or base')
        if self.layer_modes is not None:
            self.layer_modes = tuple(aliases.get(str(m).strip().lower(), str(m).strip().lower())
                                     for m in self.layer_modes)
            if len(self.layer_modes) != self.layers or any(m not in ('quantised','keyless','base') for m in self.layer_modes):
                raise ValueError('layer_modes must have one valid mode per layer')
        if not (0. < float(self.forget_gate_init) < 1.):
            raise ValueError('forget_gate_init must be between 0 and 1')
        if self.profile is None:
            self.profile = 'standard'
        if self.profile != 'standard' or self.bolt_cache == 'tight' or self.latent_dim == 16:
            raise ValueError('BOLT16 is removed: use the 32D Keyless cache')
        self.bolt_cache = 'normal'
        if self.state_dim is not None:
            self.avq_dim = int(self.state_dim)
        elif self.avq_dim is None:
            self.avq_dim = int(self.heads) * 32 if self.bolt_mode == 'keyless' else int(self.dim)
        # Defaults match the research width, while custom AVQ widths use a
        # learned return projection to preserve the general-purpose API.
        if self.attention_scale is None:
            self.attention_scale = 'latent'
        if self.memory_init is None:
            self.memory_init = 'balanced'
        if self.avq_return_map not in ('auto', 'linear', 'identity'):
            raise ValueError('avq_return_map must be auto, linear, or identity')
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
