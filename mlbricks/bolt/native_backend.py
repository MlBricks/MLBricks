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


_SPLIT_READ_CALLS = 0

def split_read_calls() -> int:
    """Diagnostic counter for actual split CUDA calls in this Python process."""
    return _SPLIT_READ_CALLS


# MLBRICKS_BOLT_WIDTH_DISPATCH_V1
# Conservative experimental policy from two-GPU synthetic + Wikipedia runs.
# These widths are PER HEAD latent dimensions, not total attention/AVQ width.
_BOLT_SPLIT_THRESHOLDS = {16: 1536, 32: 896}


def split_read_threshold(latent_dim: int) -> int:
    """Resolve native split-reader crossover; explicit env override wins.

    MLBRICKS_BOLT_SPLIT_MIN_TOKENS='auto' or unset: measured per-width policy.
    Any positive integer: manual global threshold for controlled benchmarking.
    Unknown dimensions keep the old 8192-token conservative threshold.
    """
    latent_dim = int(latent_dim)
    setting = os.getenv('MLBRICKS_BOLT_SPLIT_MIN_TOKENS', 'auto').strip().lower()
    if setting in ('auto', ''):
        return _BOLT_SPLIT_THRESHOLDS.get(latent_dim, 8192)
    try:
        n = int(setting)
    except ValueError as exc:
        raise ValueError('MLBRICKS_BOLT_SPLIT_MIN_TOKENS must be auto or a positive integer') from exc
    if n <= 0:
        raise ValueError('MLBRICKS_BOLT_SPLIT_MIN_TOKENS must be positive')
    return n


def decode_append(q, c_now, cache, scratch, position, scale):
    """Decode with existing native kernels; optionally use per-width split read.

    Default remains LEGACY when MLBRICKS_BOLT_SPLIT_CONTEXT_READ is unset/0.
    With opt-in 1, R=16 -> split from 1536; R=32 -> split from 896.
    Does not alter AVQ recurrence, CUDA kernels, cache dtype, or public API.
    """
    global _SPLIT_READ_CALLS
    ext = get_extension()
    position = int(position)
    scale = float(scale)
    enabled = os.getenv('MLBRICKS_BOLT_SPLIT_CONTEXT_READ', '0').strip().lower()
    if enabled in ('1', 'true', 'yes', 'on'):
        # Q shape in the native Direct-Q reader is [B, H, 1, R].
        # R is the 16/32 cache latent width, NOT the AVQ state width.
        if q.ndim != 4 or q.shape[-2] != 1:
            raise ValueError('split attention requires query [batch, heads, 1, latent_dim]')
        threshold = split_read_threshold(q.shape[-1])
        if position + 1 >= threshold:
            if ext is None or not hasattr(ext, 'decode_append_split'):
                raise RuntimeError('Native split-context BOLT CUDA extension unavailable')
            try:
                chunk = int(os.getenv('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS', '512'))
            except ValueError as exc:
                raise ValueError('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS must be a positive integer') from exc
            if chunk <= 0:
                raise ValueError('MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS must be positive')
            # Backend already checks capacity, cache type, and supported R.
            result = ext.decode_append_split(q.contiguous(), c_now.contiguous(),
                                             cache, scratch, position, scale, chunk)
            _SPLIT_READ_CALLS += 1
            return result
    if ext is None or not hasattr(ext, 'decode_append'):
        raise RuntimeError('Native BOLT decode extension unavailable')
    return ext.decode_append(q.contiguous(), c_now.contiguous(),
                             cache, scratch, position, scale)


__all__=['get_extension','supports_features','features','avq_scan','avq_step','decode_append']
