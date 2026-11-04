"""Optional accelerated BOLT state scan. No runtime JIT or Triton dependency.

To build native recurrence operators, set MLBRICKS_BUILD_BOLT_ACCEL=1 when
building/installing from source. When absent, BOLT automatically uses the
existing native-modern implementation or the exact PyTorch reference.
"""
from __future__ import annotations
import importlib
import torch

_SEGMENTED = None
_SHORT = None
_ATTEMPTED = False


def _load():
    global _SEGMENTED, _SHORT, _ATTEMPTED
    if not _ATTEMPTED:
        _ATTEMPTED = True
        if torch.cuda.is_available():
            for name, attr in (("mlbricks._bolt_scan_segmented", "_SEGMENTED"),
                               ("mlbricks._bolt_scan_short", "_SHORT")):
                try:
                    obj = importlib.import_module(name)
                except ImportError:
                    obj = None
                if attr == "_SEGMENTED":
                    _SEGMENTED = obj
                else:
                    _SHORT = obj
    return _SEGMENTED, _SHORT


def available():
    a, b = _load()
    return a is not None or b is not None


class _SegmentedScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, v):
        seg, _ = _load()
        if seg is None:
            raise RuntimeError("Segmented BOLT CUDA extension not available")
        a, v = a.float().contiguous(), v.float().contiguous()
        s0 = torch.zeros(a.shape[0], a.shape[-1], device=a.device, dtype=torch.float32)
        states = seg.forward(a, v, s0)
        ctx.save_for_backward(a, states, s0)
        return states

    @staticmethod
    def backward(ctx, ds):
        a, states, s0 = ctx.saved_tensors
        da, dv, _ = _load()[0].backward(a, states, s0, ds.float().contiguous())
        return da, dv


class _ShortScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, za, zv, warps):
        _, short = _load()
        if short is None:
            raise RuntimeError("Short-sequence BOLT CUDA extension not available")
        s0 = torch.zeros(za.shape[0], za.shape[-1], device=za.device, dtype=torch.float32)
        states = short.forward(za, zv, s0, int(warps), True)
        ctx.save_for_backward(za, zv, states, s0)
        ctx.warps = int(warps)
        return states

    @staticmethod
    def backward(ctx, ds):
        za, zv, states, s0 = ctx.saved_tensors
        dza, dzv, _ = _load()[1].backward(
            za, zv, states, s0, ds.float().contiguous(), ctx.warps, True
        )
        return dza, dzv, None


def scan(za, zv, *, a=None, v=None, latent_dim=32):
    """Return FP32 states or None when the accelerated path is unavailable.

    The short kernel reads raw gates; for long contexts we preserve the old
    gate activation/rounding order and use the segmented FP32 scan.
    """
    if not za.is_cuda or za.dtype not in (torch.float16, torch.float32):
        return None
    segmented, short = _load()
    length = za.size(1)
    if short is not None and length <= 512 and length >= 1:
        return _ShortScan.apply(za, zv, 8 if int(latent_dim) == 16 else 16)
    if segmented is not None and length >= 1:
        if a is None:
            a = torch.sigmoid(za).float()
        if v is None:
            v = (0.1 * torch.tanh(zv)).float()
        return _SegmentedScan.apply(a, v)
    return None
