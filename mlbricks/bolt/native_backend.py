"""Optional native BOLT/AVQ CUDA extension with autograd-safe dispatch.

MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 builds `mlbricks._bolt_modern_cuda` with the CUDA toolkit.
For development only, MLBRICKS_BOLT_MODERN_ALLOW_JIT=1 enables optional JIT compilation.
The module is completely independent of the legacy `_gauss_cuda` extension.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch

_EXTENSION = None
_ATTEMPTED = False

def get_extension():
    global _EXTENSION, _ATTEMPTED
    if _ATTEMPTED:
        return _EXTENSION
    _ATTEMPTED = True
    if not torch.cuda.is_available():
        return None
    try:
        _EXTENSION = importlib.import_module('mlbricks._bolt_modern_cuda')
        return _EXTENSION
    except ImportError:
        pass
    if os.environ.get('MLBRICKS_BOLT_MODERN_ALLOW_JIT', '0').lower() not in ('1','yes','true','on'):
        return None
    try:
        from torch.utils.cpp_extension import load
        root = Path(__file__).parent / 'native_modern'
        _EXTENSION = load(
            name='mlbricks_bolt_modern_cuda_dev',
            sources=[str(root/'bindings.cpp'), str(root/'kernels.cu')],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
            with_cuda=True, verbose=False,
        )
    except Exception:
        _EXTENSION = None
    return _EXTENSION


def supports_features(q, u, g):
    """True for the Direct-Q no-P inference kernel on supported CUDA tensors."""
    ext = get_extension()
    return (ext is not None and hasattr(ext, 'features_forward')
            and q.is_cuda and u.is_cuda and g.is_cuda
            and q.device == u.device == g.device
            and q.dtype in (torch.float16, torch.float32)
            and q.dtype == u.dtype == g.dtype
            and q.ndim == 4 and q.shape == u.shape == g.shape
            and not any(t.requires_grad for t in (q, u, g)))


def features(q, u, g):
    """Inference-only fused Direct-Q and gated C transform; gradients use PyTorch."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, u, g)):
        raise RuntimeError('Native Direct-Q features are inference-only; use the PyTorch autograd training path')
    ext = get_extension()
    if ext is None or not hasattr(ext, 'features_forward'):
        raise RuntimeError('Rebuild the modern BOLT Direct-Q CUDA extension')
    return ext.features_forward(q.contiguous(), u.contiguous(), g.contiguous())


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,v):
        ext=get_extension()
        if ext is None:raise RuntimeError('AVQ CUDA extension is not installed')
        s=ext.avq_scan_forward(a.contiguous(),v.contiguous())
        ctx.save_for_backward(a,s)
        return s

    @staticmethod
    def backward(ctx,grad_s):
        a,s=ctx.saved_tensors
        da,dv=get_extension().avq_scan_backward(grad_s.contiguous(),a.contiguous(),s.contiguous())
        return da,dv


def avq_scan(a,v):
    return _Scan.apply(a,v)


def avq_step(a,v,previous):
    return get_extension().avq_decode_state(a.contiguous(),v.contiguous(),previous.contiguous())


def decode_append(q,c_now,cache,scratch,position,scale):
    return get_extension().decode_append(q.contiguous(),c_now.contiguous(),cache,scratch, int(position),float(scale))

__all__=['get_extension','supports_features','features','avq_scan','avq_step','decode_append']
