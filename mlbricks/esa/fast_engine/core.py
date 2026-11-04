"""MLBricks ESA: current half-life ESA and historical BaseESA.

Preserves the original recurrence equations, parameters, and checkpoint names.
"""
from __future__ import annotations

from pathlib import Path
import math
from types import SimpleNamespace
import re
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ._original import progress_learned as learned
from ._original import progress_hybrid as hybrid

Variant = Literal["base", "current"]


def normalize_variant(variant: str) -> Variant:
    """Canonicalize ESA architecture names; historical names stay valid."""
    key = str(variant).strip().lower().replace("-", "_")
    if key in {"base", "base_esa", "fast", "fastesa", "learned", "progress_learned"}:
        return "base"
    if key in {"current", "esa", "half_life", "halflife", "quality",
               "hybrid", "progress_hybrid"}:
        return "current"
    raise ValueError(f"Unknown ESA variant {variant!r}; choose 'current' or 'base'.")


_SCAN_NAMES = {
    "auto": "auto", "parallel": "parallel", "short": "short",
    "adaptive": "adaptive", "hierarchical": "hierarchical",
    "reference": "reference", "sequential": "sequential",
    # Historic API, accepted for checkpoint and notebook reproducibility.
    "swqr": "parallel", "swqr_short": "short",
    "swqr_adaptive": "adaptive", "thunder": "hierarchical",
    "original": "reference", "direct": "sequential",
}

def normalize_scan(scan: str) -> str:
    key = str(scan).strip().lower().replace("-", "_")
    if key not in _SCAN_NAMES:
        raise ValueError(f"Unknown scan {scan!r}; choose auto, parallel, short, adaptive, hierarchical or reference")
    return _SCAN_NAMES[key]


class BaseESA(learned.ESAFullMixer):
    """Historical learned-bounds ESA with weight-compatible backends.

    scan='auto' (default): T4 shape-aware SWQR CUDA (Short at measured small
    shapes, original SWQR elsewhere), Thunder CPU/compile fallback, direct for
    cached single-token decoding.
    scan='swqr': native SWQR (CUDA); scan='thunder': PyTorch hierarchical scan;
    scan='original': unchanged PROGRESS reference.
    """

    def __init__(self, dim: int, heads: int = 6, scan: str = "auto",
                 cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8,
                 fused_decode: str = "auto"):
        super().__init__(dim=dim, heads=heads, variant="learned")
        scan = normalize_scan(scan)
        if not isinstance(cumpass, int) or cumpass < 1:
            raise ValueError("cumpass must be a positive integer")
        if parallel_chunk not in {16, 32, 64, 128, 256, 512}:
            raise ValueError("parallel_chunk must be a power of two in [16,512]")
        self.scan = normalize_scan(scan)
        self.cumpass = cumpass
        if cuda_warps not in (4, 8, 16):
            raise ValueError("cuda_warps must be 4, 8, or 16")
        self.parallel_chunk = parallel_chunk
        self.cuda_warps = cuda_warps
        if fused_decode not in {"auto", "on", "off"}:
            raise ValueError("fused_decode must be 'auto', 'on', or 'off'")
        self.fused_decode = fused_decode
        self._fused_unavailable = False

    def forward(self, x, state=None):
        if self.scan == "reference":
            return super().forward(x, state)
        from .scan import (scan as auto_scan, sequential_scan as direct_scan,
                           parallel_scan as swqr_scan, hierarchical_scan as thunder_scan,
                           short_scan as swqr_short_scan, adaptive_scan as swqr_adaptive_scan)
        if x.ndim != 3:
            raise ValueError("Expected input [batch, sequence, dim]")
        B, T, D = x.shape
        if T < 1:
            raise ValueError("Input must contain at least one token")
        if state is not None and (tuple(state.shape) != (B, self.heads, self.dh)
                                  or state.device != x.device):
            raise ValueError(f"Initial state must be {(B, self.heads, self.dh)} on {x.device}")
        packed = self.qgv(x)
        # BaseESA's original learned-bounds recurrence, fused exactly like
        # Half-Life ESA for inference-only, one-token CUDA decoding.
        if (T == 1 and x.is_cuda and not torch.is_grad_enabled()
                and self.fused_decode != "off" and not self._fused_unavailable):
            try:
                from .base_decode import decode_base
                read, final = decode_base(
                    packed, x, self.raw_alpha, self.raw_beta, state, self.heads)
                return self.W_O(read), final
            except (RuntimeError, OSError, ImportError) as exc:
                if self.fused_decode == "on":
                    raise
                import warnings
                self._fused_unavailable = True
                warnings.warn(f"Fused BaseESA decode unavailable ({exc}); "
                              "using unfused recurrence", RuntimeWarning, stacklevel=2)
        q, g, vraw = packed.chunk(3, dim=-1)
        gate = g.float().reshape(B, T, self.heads, self.dh)
        v = torch.tanh(vraw.float().reshape(B, T, self.heads, self.dh))
        alpha, beta = self.bounds()
        A = alpha[None, None, :, None] + (beta-alpha)[None, None, :, None]*torch.sigmoid(gate)
        U = (1.0-A)*v
        # No sequence-parallel scan for a one-token decode, even in SWQR mode.
        if self.scan == "sequential" and T != 1:
            raise ValueError("Sequential scan only supports one token")
        if T == 1:
            E = direct_scan(A, U, state)
        elif self.scan == "parallel":
            E = swqr_scan(A, U, state, chunk=self.parallel_chunk)
        elif self.scan == "short":
            E = swqr_short_scan(A, U, state, warps=self.cuda_warps)
        elif self.scan == "adaptive":
            E = swqr_adaptive_scan(A, U, state, warps=self.cuda_warps,
                                   chunk=self.parallel_chunk)
        elif self.scan == "hierarchical":
            E = thunder_scan(A, U, state, compass=self.cumpass)
        else:
            E = auto_scan(A, U, state, compass=self.cumpass, chunk=self.parallel_chunk)
        final = E[:, -1]
        z = F.rms_norm(E.reshape(B,T,D),(D,),eps=1e-5)
        y = (torch.sigmoid(q.float())*z).to(x.dtype)
        return self.W_O(y), final


class ESA(hybrid.HybridESA):
    """Current half-life ESA with the original PROGRESS hybrid weights.

    - scan='reference': exact original PROGRESS chunk-scan implementation.
    - scan='auto' (default): CUDA SWQR for prefill/training, Thunder for CPU,
      direct recurrence for one-token decode.
    - fused_decode='auto' (default): native fused CUDA gate/write/learned
      RMSNorm/query readout for CUDA inference with T=1. Compilation failures
      fall back with an explicit warning. 'on' raises, 'off' disables fusion.

    CUDA Graph replay is opt-in via fastesa.graph.GraphDecoder.
    No architecture, state_dict keys, parameters or recurrence are changed.
    """

    _scans = set(_SCAN_NAMES)

    def __init__(self, dim: int, heads: int = 6, chunk_size: int = 32,
                 scan: str = "auto", fused_decode: str = "auto",
                 cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8):
        super().__init__(dim=dim, heads=heads, chunk_size=chunk_size)
        scan = normalize_scan(scan)
        if fused_decode not in {"auto", "on", "off"}:
            raise ValueError("fused_decode must be 'auto', 'on', or 'off'")
        if cumpass < 1 or not isinstance(cumpass, int):
            raise ValueError("cumpass must be a positive integer")
        if parallel_chunk not in {16, 32, 64, 128, 256, 512}:
            raise ValueError("parallel_chunk must be 16, 32, 64, 128, 256 or 512")
        if cuda_warps not in {4, 8, 16}:
            raise ValueError("cuda_warps must be 4, 8, or 16")
        self.scan = normalize_scan(scan)
        self.fused_decode = fused_decode
        self.cumpass = cumpass
        self.parallel_chunk = parallel_chunk
        self.cuda_warps = cuda_warps
        self._fused_unavailable = False

    def forward(self, x, state=None):
        if self.scan == "reference":
            return super().forward(x, state)
        from .scan import (scan as auto_scan, sequential_scan as direct_scan,
                           parallel_scan as swqr_scan, hierarchical_scan as thunder_scan,
                           short_scan as swqr_short_scan, adaptive_scan as swqr_adaptive_scan)
        if x.ndim != 3 or x.size(-1) != self.dim:
            raise ValueError(f"Input must be [B,T,{self.dim}]")
        B, T, D = x.shape
        if T < 1:
            raise ValueError("Input sequence cannot be empty")
        if state is not None and (tuple(state.shape) != (B, self.heads, self.dh)
                                  or state.device != x.device):
            raise ValueError(f"State must be [{B},{self.heads},{self.dh}] on {x.device}")
        packed = self.qgv(x)

        # Inference-only (no gradients) CUDA fusion. The trained weights and
        # exact half-life mathematical formula are unchanged. Compilation is
        # lazy; users should call prepare_quality_decode() before graph capture.
        if (T == 1 and x.is_cuda and not torch.is_grad_enabled()
                and self.fused_decode != "off" and not self._fused_unavailable):
            try:
                from .decode import decode
                read, final = decode(
                    packed, x, self.log_half_life, self.raw_decay_bias,
                    self.state_norm.weight, state, self.heads)
                return self.W_O(read), final
            except (RuntimeError, OSError, ImportError) as exc:
                if self.fused_decode == "on":
                    raise
                import warnings
                self._fused_unavailable = True
                warnings.warn(f"Fused CUDA decode unavailable ({exc}); "
                              "using unfused recurrence", RuntimeWarning, stacklevel=2)

        q, g, raw_v = packed.chunk(3, dim=-1)
        z = g.reshape(B, T, self.heads, self.dh).float()
        half_life = self.log_half_life.float().exp()
        decay = F.softplus(z + self.raw_decay_bias[None, None, :, None])
        A = torch.exp(-math.log(2.0) * decay / half_life[None, None, :, None])
        value = torch.tanh(raw_v.reshape(B, T, self.heads, self.dh).float())
        U = (1.0 - A) * value
        if self.scan == "sequential" and T != 1:
            raise ValueError("Sequential scan only supports one token")
        if T == 1:
            E = direct_scan(A, U, state)
        elif self.scan == "parallel":
            E = swqr_scan(A, U, state, chunk=self.parallel_chunk)
        elif self.scan == "short":
            E = swqr_short_scan(A, U, state, warps=self.cuda_warps)
        elif self.scan == "adaptive":
            E = swqr_adaptive_scan(A, U, state, warps=self.cuda_warps,
                                   chunk=self.parallel_chunk)
        elif self.scan == "hierarchical":
            E = thunder_scan(A, U, state, compass=self.cumpass)
        else:
            E = auto_scan(A, U, state, compass=self.cumpass, chunk=self.parallel_chunk)
        final = E[:, -1]
        norm = self.state_norm(E.reshape(B, T, D))
        read = (torch.sigmoid(q.float()) * norm).to(x.dtype)
        return self.W_O(read), final


class BaseESAModel(learned.KaggleRecall):
    """Historical learned-bounds model with accelerated recurrent mixers."""
    def __init__(self, vocab_size: int = 256, dim: int = 384,
                 heads: int = 6, layers: int = 6,
                 scan: str = "auto", cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8):
        super().__init__(vocab_size=vocab_size, dim=dim, heads=heads,
                         layers=layers, variant="learned")
        if normalize_scan(scan) != "reference":
            for block in self.blocks:
                block.esa = BaseESA(dim, heads, scan=scan,
                                    cumpass=cumpass, parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
        self.scan = normalize_scan(scan)


class ESAModel(hybrid.RecallModel):
    """Current half-life model with accelerated execution backend."""
    def __init__(self, vocab_size: int = 256, dim: int = 384,
                 heads: int = 6, layers: int = 6, chunk_size: int = 32,
                 scan: str = "auto", fused_decode: str = "auto",
                 cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8):
        super().__init__(vocab_size=vocab_size, dim=dim, heads=heads,
                         layers=layers, variant="hybrid", chunk_size=chunk_size)
        if normalize_scan(scan) != "reference":
            for block in self.blocks:
                original = block.esa
                accelerated = ESA(
                    dim, heads, chunk_size=chunk_size, scan=scan,
                    fused_decode=fused_decode, cumpass=cumpass,
                    parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
                accelerated.load_state_dict(original.state_dict(), strict=True)
                block.esa = accelerated
        self.scan = normalize_scan(scan)


def make_esa(variant: str = "current", *, dim: int = 384, heads: int = 6,
             chunk_size: int = 32, scan: str = "auto", fused_decode: str = "auto",
             cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8) -> nn.Module:
    """Create standalone mixer, with original trained state_dict key layout."""
    if normalize_variant(variant) == "base":
        if chunk_size != 32:
            raise ValueError("PROGRESS learned uses fixed chunk_size=32; cannot change it without changing the implementation")
        return BaseESA(dim=dim, heads=heads, scan=scan, fused_decode=fused_decode,
                       cumpass=cumpass, parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
    return ESA(dim=dim, heads=heads, chunk_size=chunk_size, scan=scan,
                      fused_decode=fused_decode, cumpass=cumpass, parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)


def make_model(variant: str = "current", *, vocab_size: int = 256,
               dim: int = 384, heads: int = 6, layers: int = 6,
               chunk_size: int = 32, seed: int | None = None,
               scan: str = "auto", fused_decode: str = "auto", cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8) -> nn.Module:
    """Build original recall/LM-shaped model.

    If seed is provided, reproduce the variant's EXACT notebook initializer.
    This is an architecture factory, not a pretrained weights download.
    """
    v = normalize_variant(variant)
    if v == "base":
        if chunk_size != 32:
            raise ValueError("PROGRESS learned uses fixed chunk_size=32")
        model = BaseESAModel(vocab_size=vocab_size, dim=dim, heads=heads, layers=layers,
                             scan=scan, cumpass=cumpass, parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
        if seed is not None:
            learned.deterministic_init(model, int(seed))
        return model
    model = ESAModel(vocab_size=vocab_size, dim=dim, heads=heads,
                            layers=layers, chunk_size=chunk_size, scan=scan,
                            fused_decode=fused_decode, cumpass=cumpass,
                            parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
    if seed is not None:
        args = SimpleNamespace(seed=int(seed), vocab_size=vocab_size, dim=dim,
                               heads=heads, layers=layers, chunk_size=chunk_size)
        reference = hybrid.build_model(args, "hybrid")
        model.load_state_dict(reference.state_dict(), strict=True)
    return model


def forward_tokens(model: nn.Module, tokens: torch.Tensor, states=None):
    """Uniform streaming API for the *unchanged* notebook LM models.

    Returns (last-token logits [B,V], new_states [layers][B,H,D/H]).
    Requires tokens shaped [B,T] with T >= 1.
    """
    if isinstance(model, learned.KaggleRecall):
        return model.forward_chunk(tokens, states)
    if isinstance(model, hybrid.RecallModel):
        return model(tokens, states)
    raise TypeError("Expected a model created by make_model")


def load_checkpoint(path: str | Path, *, variant: str | None = None,
                    device: str | torch.device = "cpu", chunk_size: int = 32,
                    trusted: bool = False, scan: str = "auto", fused_decode: str = "auto",
                    cumpass: int = 16, parallel_chunk: int = 128, cuda_warps: int = 8) -> nn.Module:
    """Restore a `latest.pt` model produced by an ORIGINAL PROGRESS notebook.

    Only model weights are restored, not optimizer/scaler/RNG state. Architecture
    is inferred from original weight tensors; variant is checked against the
    checkpoint's variant label, when present.

    By default deserialization runs in torch `weights_only` safe mode. For
    hybrid notebooks that save arbitrary NumPy RNG objects, only pass
    trusted=True if YOU trust the checkpoint producer and contents.
    """
    try:
        obj = torch.load(Path(path), map_location="cpu", weights_only=not trusted)
    except Exception as e:
        if not trusted:
            raise ValueError(
                "Could not safely read checkpoint; original notebook may contain "
                "NumPy/Python RNG objects. Only if you trust the file, call "
                "load_checkpoint(..., trusted=True)."
            ) from e
        raise
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint should contain a model state dict or {'model': state_dict}.")
    state = obj.get("model", obj)
    if not isinstance(state, dict) or "emb.weight" not in state:
        raise ValueError("Expected original PROGRESS recall model state_dict with emb.weight")
    indicated = obj.get("variant", None)
    if indicated is not None and variant is not None:
        if normalize_variant(indicated) != normalize_variant(variant):
            raise ValueError(f"Checkpoint is {indicated!r}, not {variant!r}")
    if variant is None:
        if indicated is None:
            if "blocks.0.esa.raw_alpha" in state:
                variant = "base"
            elif "blocks.0.esa.log_half_life" in state:
                variant = "current"
            else:
                raise ValueError("Unknown checkpoint architecture; specify a variant")
        else:
            variant = indicated
    v = normalize_variant(variant)
    vocab_size, dim = state["emb.weight"].shape
    layer_ids = [int(m.group(1)) for name in state
                 if (m := re.match(r"^blocks\.(\d+)\.esa\.", name))]
    if not layer_ids:
        raise ValueError("No ESA blocks found in checkpoint")
    layers = max(layer_ids) + 1
    if v == "base":
        heads = int(state["blocks.0.esa.raw_alpha"].numel())
    else:
        heads = int(state["blocks.0.esa.log_half_life"].numel())
    model = make_model(v, vocab_size=int(vocab_size), dim=int(dim),
                       heads=heads, layers=layers, chunk_size=chunk_size,
                       scan=scan, fused_decode=fused_decode, cumpass=cumpass, parallel_chunk=parallel_chunk, cuda_warps=cuda_warps)
    model.load_state_dict(state, strict=True)
    return model.to(device)


# Backwards compatible names: historical scripts and checkpoints keep working.
FastESA = BaseESA
QualityESA = ESA
FastESAModel = BaseESAModel
QualityESAModel = ESAModel
