"""Session-owned, optional BOLT layer CUDA Graph decode acceleration.

A graph is tied to a single preallocated BoltCache, not the shared nn.Module.
Capture never occurs while an outer CUDA capture or torch.compile trace is active.
Unsupported layouts stay on the existing, mathematically defined eager path.

This is an experimental fast path; CUDA GPU qualification is required before
advertising performance or graph parity beyond its declared eligibility gates.
"""
from __future__ import annotations

import importlib
import time
import weakref
from dataclasses import dataclass
from typing import Any

import torch


def _outer_graph_active(device: torch.device | None = None) -> bool:
    """Never nest a layer-owned CUDAGraph inside a model-level capture/compile."""
    checker = getattr(getattr(torch, 'compiler', None), 'is_compiling', None)
    if callable(checker) and checker():
        return True
    dynamo = getattr(torch, '_dynamo', None)
    if dynamo is not None and callable(getattr(dynamo, 'is_compiling', None)) and dynamo.is_compiling():
        return True
    if device is not None and device.type == 'cuda':
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except (RuntimeError, AssertionError):
            # No CUDA context is ready yet; normal eligibility will reject.
            pass
    return False


def _is_nvidia_cuda(x: torch.Tensor) -> bool:
    # ROCm intentionally exposes torch.cuda. Do not load NVIDIA CUDA extensions there.
    return bool(x.device.type == 'cuda' and torch.cuda.is_available()
                and getattr(torch.version, 'hip', None) is None
                and getattr(torch.version, 'cuda', None) is not None)


def _dynamic_extension():
    try:
        ext = importlib.import_module('mlbricks._bolt_dynamic_cuda')
    except (ImportError, OSError):
        return None
    return ext if callable(getattr(ext, 'decode', None)) else None


def _fingerprint(owner, x: torch.Tensor, cache) -> tuple:
    return (
        x.device, x.dtype, tuple(x.shape),
        cache.c.data_ptr(), cache.c.dtype, tuple(cache.c.shape),
        cache.state.data_ptr() if cache.state is not None else None,
        cache.scratch.data_ptr() if cache.scratch is not None else None,
        str(owner.backend), str(getattr(owner, 'graph', 'auto')),
        bool(torch.is_autocast_enabled('cuda')),
        torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None,
        tuple((p.data_ptr(), p._version) for p in owner.parameters()),
    )


@dataclass
class LayerGraphStatus:
    backend: str
    reason: str
    captures: int = 0
    replays: int = 0
    capture_seconds: float = 0.0
    variants: int = 0


class LayerGraphCaptureRecovered(RuntimeError):
    """Graph capture failed but every persistent tensor was safely restored."""


class BoltLayerGraphSession:
    """CUDA Graph session for a SINGLE preallocated modern BOLT layer cache.

    Native dynamic kernel: batch=1, no RoPE, latent=16/32, FP16/FP32 C,
    persistent FP32 AVQ state. Different sessions never reuse graph state.
    """
    def __init__(self, owner, cache, x: torch.Tensor, ext, *, chunk_tokens: int = 512):
        self.owner_ref = weakref.ref(owner)
        self.cache_ref = weakref.ref(cache)
        self.ext = ext
        self.device = x.device
        self.chunk_tokens = int(chunk_tokens)
        self.x_static = torch.empty_like(x)
        self.pos_gpu = torch.zeros(1, device=x.device, dtype=torch.int64)
        self.variants: dict[tuple, tuple[Any, torch.Tensor]] = {}
        self.fingerprint = _fingerprint(owner, x, cache)
        self.capture_seconds = 0.0
        self.replays = 0
        self.captures = 0
        self.disabled_reason = ''

    def matches(self, owner, cache, x):
        return (self.owner_ref() is owner and self.cache_ref() is cache and
                self.fingerprint == _fingerprint(owner, x, cache))

    def _variant(self, pos: int) -> tuple:
        owner = self.owner_ref()
        threshold = 1536 if owner.latent_dim == 16 else 896
        if pos + 1 < threshold:
            return ('fast', 0, 1)
        bucket = min(max(1024, 1 << (pos.bit_length())), self.cache_ref().capacity)
        chunks = (bucket + self.chunk_tokens - 1) // self.chunk_tokens
        return ('split', bucket, chunks)

    def _decode_core(self, key: tuple):
        owner, cache = self.owner_ref(), self.cache_ref()
        mode, _bucket, chunks = key
        q, _, current = owner._project(self.x_static, start_pos=0)
        q = q.float().contiguous()
        if current.dtype != cache.c.dtype:
            current = current.to(cache.c.dtype)
        # torch.library exposes cache/scratch mutations to PyTorch's compiler.
        # The native extension does the exact same GPU-position CUDA read.
        from .custom_ops import bolt_dynamic_decode
        value = bolt_dynamic_decode(q, current.contiguous(), cache.c,
                                    cache.scratch, self.pos_gpu, owner._scale(),
                                    self.chunk_tokens, chunks, mode == 'split')
        value = value.transpose(1, 2).contiguous().reshape(1, 1, -1)
        # Keep AVQ state in its original persistent FP32 allocation.
        # The native eager reader returns FP32. Preserve that input dtype for
        # AVQ to keep its FP32 recurrence and output accumulation semantics.
        if owner.avq is not None:
            value, state = owner.avq.step(value, cache.state)
            cache.state.copy_(state)
        return owner.out_proj(value)

    def _capture(self, key: tuple, x: torch.Tensor, pos: int):
        owner, cache = self.owner_ref(), self.cache_ref()
        # Snapshot before warmup/capture: both can update AVQ and cache.
        saved_c = cache.c.detach().clone()
        saved_state = cache.state.detach().clone() if cache.state is not None else None
        saved_scratch = cache.scratch.detach().clone()
        saved_pos = self.pos_gpu.detach().clone()
        saved_x = self.x_static.detach().clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        capture_error = None
        try:
            with torch.cuda.stream(stream):
                self.x_static.copy_(x)
                self.pos_gpu.fill_(pos)
                for _ in range(2):
                    self._decode_core(key)
                    self.pos_gpu.fill_(pos)
            torch.cuda.current_stream(device=self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(graph, stream=stream):
                output = self._decode_core(key)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            capture_error = exc
        finally:
            # Only offer the graph after every live buffer has been restored.
            # If CUDA itself became unrecoverable, restore raises: never proceed.
            with torch.no_grad():
                cache.c.copy_(saved_c)
                cache.scratch.copy_(saved_scratch)
                if saved_state is not None:
                    cache.state.copy_(saved_state)
                self.pos_gpu.copy_(saved_pos)
                self.x_static.copy_(saved_x)
            torch.cuda.synchronize(self.device)
        if capture_error is not None:
            raise LayerGraphCaptureRecovered(
                'BOLT CUDA Graph capture restored all cache/state buffers: '
                + str(capture_error)
            ) from capture_error
        self.capture_seconds += time.perf_counter() - started
        self.captures += 1
        self.variants[key] = (graph, output)

    @torch.no_grad()
    def run(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        owner, cache = self.owner_ref(), self.cache_ref()
        if owner is None or cache is None:
            raise RuntimeError('BOLT graph session owner/cache was released')
        if not self.matches(owner, cache, x):
            raise RuntimeError('BOLT graph session pointers, weights or precision changed')
        if pos != cache.length or not 0 <= pos < cache.capacity:
            raise ValueError('BOLT graph cache position/capacity mismatch')
        key = self._variant(pos)
        if key not in self.variants:
            self._capture(key, x, pos)
        graph, output = self.variants[key]
        self.x_static.copy_(x)
        self.pos_gpu.fill_(pos)
        # Fatal replay faults propagate rather than continuing with a mutated cache.
        graph.replay()
        # Return a distinct tensor because graph output storage is recycled on replay.
        result = output.clone()
        cache.length = pos + 1
        self.replays += 1
        return result


def _eligibility(owner, x, cache, pos: int) -> tuple[str, Any]:
    if owner.backend == 'pytorch' or getattr(owner, 'graph', 'auto') == 'off':
        return 'disabled by policy', None
    if owner.training or torch.is_grad_enabled():
        return 'training or autograd enabled', None
    if not _is_nvidia_cuda(x):
        return 'not NVIDIA CUDA', None
    if _outer_graph_active(x.device):
        return 'outer compile/capture owns graph', None
    if getattr(owner, 'position', None) is not None:
        return 'RoPE graph path not qualified', None
    if owner.latent_dim not in (16, 32) or x.ndim != 3 or tuple(x.shape[:2]) != (1, 1):
        return 'only batch1 one-token BOLT16/32 qualified', None
    if (cache.c.device != x.device or cache.c.dtype not in (torch.float16, torch.float32)
            or cache.c.shape[0] != 1 or cache.c.shape[1] != owner.num_heads
            or cache.c.shape[-1] != owner.latent_dim or not cache.c.is_contiguous()):
        return 'unsupported cache layout', None
    if pos != cache.length or pos < 0 or pos >= cache.capacity:
        return 'cache position mismatch', None
    if cache.state is None and owner.avq is not None:
        return 'missing AVQ state', None
    if cache.state is not None and owner.avq is None:
        return 'unexpected AVQ state', None
    if cache.state is not None and (cache.state.dtype != torch.float32 or cache.state.device != x.device
                                    or tuple(cache.state.shape) != (1, owner.avq_dim)):
        return 'AVQ state must be FP32 on same device', None
    # Do not substitute the separately opt-in V2 Strict cached recurrence with
    # another implementation until their FP32 state parity is GPU-qualified.
    from . import v2_strict
    if v2_strict.enabled():
        return 'V2 Strict AVQ graph composition not GPU-qualified', None
    if not owner.causal or owner.dropout != 0.0:
        return 'causal zero-dropout only', None
    # The dynamic reader supports FP16 and FP32 cache values. Restrict the
    # automatic graph path to the benchmarked FP16 cache until wider GPU tests.
    # FP32 input under CUDA autocast is a normal hybrid-model configuration.
    if cache.c.dtype != torch.float16 or x.dtype not in (torch.float16, torch.float32):
        return 'layer graph currently qualified for FP16 cache', None
    if x.dtype == torch.float32 and not torch.is_autocast_enabled('cuda'):
        return 'FP32 input requires CUDA autocast for FP16 cache', None
    if cache.capacity < 128:
        return 'cache capacity below native scratch threshold', None
    ext = _dynamic_extension()
    if ext is None:
        return 'mlbricks._bolt_dynamic_cuda not installed', None
    return '', ext


def try_auto_layer_graph(owner, x: torch.Tensor, cache, pos: int):
    """Return layer output or None to allow existing optimized/native eager decode.

    graph='on' is strict (except under an enclosing capture); graph='auto'
    never sacrifices correctness for an unqualified graph. Capture failures
    are made visible; ambiguous CUDA failures are NOT silently retried.
    """
    policy = getattr(owner, 'graph', 'auto')
    if getattr(cache, '_mlbricks_graph_unavailable', None):
        reason = cache._mlbricks_graph_unavailable
        owner._mlbricks_graph_status = LayerGraphStatus('eager', reason)
        if policy == 'on':
            raise RuntimeError('Required BOLT graph unavailable: ' + reason)
        return None
    reason, ext = _eligibility(owner, x, cache, pos)
    if reason:
        owner._mlbricks_graph_status = LayerGraphStatus('eager', reason)
        if policy == 'on' and reason not in ('outer compile/capture owns graph',):
            raise RuntimeError('BOLT graph="on" requested but unavailable: ' + reason)
        return None
    # The base cache allows lazy scratch allocation, but the graph must own
    # stable scratch addresses. Allocate before constructing its session.
    if cache.scratch is None:
        cache.scratch = torch.empty((1, owner.num_heads, cache.capacity),
                                    device=x.device, dtype=torch.float32)
    session = getattr(cache, '_mlbricks_layer_graph', None)
    if not isinstance(session, BoltLayerGraphSession) or not session.matches(owner, cache, x):
        session = BoltLayerGraphSession(owner, cache, x, ext)
        cache._mlbricks_layer_graph = session
    # Auto policy amortizes initial capture over repeated steps; explicit 'on'
    # captures on first eligible decode. We never block native eager otherwise.
    uses = int(getattr(cache, '_mlbricks_graph_calls', 0)) + 1
    cache._mlbricks_graph_calls = uses
    if policy == 'auto' and uses < 3:
        owner._mlbricks_graph_status = LayerGraphStatus('eager', 'graph warmup (first two tokens)')
        return None
    try:
        y = session.run(x, pos)
    except LayerGraphCaptureRecovered as exc:
        cache._mlbricks_graph_unavailable = str(exc)
        owner._mlbricks_graph_status = LayerGraphStatus('eager', str(exc))
        if policy == 'on':
            raise RuntimeError('Required BOLT graph capture failed') from exc
        return None
    except Exception as exc:
        # Capture may have been rolled back but a CUDA replay failure can be
        # unrecoverable. Fail loudly, never double-advance the recurrence.
        owner._mlbricks_graph_status = LayerGraphStatus('error', str(exc))
        raise
    owner._mlbricks_graph_status = LayerGraphStatus(
        'cuda_graph', '', session.captures, session.replays,
        session.capture_seconds, len(session.variants))
    return y
