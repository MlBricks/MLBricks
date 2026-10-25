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


def supports_features(q,u,g,p):
    return (get_extension() is not None and q.is_cuda and u.is_cuda and g.is_cuda and p.is_cuda
            and q.dtype in (torch.float16,torch.float32) and q.dtype == u.dtype == g.dtype
            and p.dtype==torch.float32)

class _Features(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, u, g, p):
        ext=get_extension()
        if ext is None:
            raise RuntimeError('BOLT CUDA extension is not installed')
        # q/u/g: [B,H,T,R] p: [H,R]
        qc, c = ext.features_forward(q.contiguous(),u.contiguous(),g.contiguous(),p.contiguous())
        ctx.save_for_backward(q, u, g, p)
        return qc, c

    @staticmethod
    def backward(ctx, grad_q, grad_c):
        q,u,g,p=ctx.saved_tensors
        ext=get_extension()
        if grad_q is None:
            grad_q=torch.zeros(q.shape,device=q.device,dtype=torch.float32)
        if grad_c is None:
            grad_c=torch.zeros_like(u)
        dq,du,dg,dp=ext.features_backward(grad_q.contiguous().float(), grad_c.contiguous().to(u.dtype),
                                           q.contiguous(),u.contiguous(),g.contiguous(),p.contiguous())
        return dq,du,dg,dp


def features(q,u,g,p):
    return _Features.apply(q,u,g,p)

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
