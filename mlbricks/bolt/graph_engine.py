"""Opt-in BOLT single-token CUDA Graph replay (experimental integration).

The native graph path is deliberately restricted to the *validated* WikiLM-like
six-layer decoder protocol.  It is not yet a general BoltModel CUDA graph
implementation.  Other architectures with ``decode_one`` or ``decode_step``
continue through an eager, one-token-at-a-time fallback.

The graph-safe native reader uses a persistent device-resident int64 position,
so the cache advances on replay without baking a Python position into CUDA.
Native extension compilation is opt-in: MLBRICKS_BUILD_BOLT_DYNAMIC_GRAPH=1.
"""
from __future__ import annotations

import copy
import importlib
import math
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaptureInfo:
    mode: str
    bucket: int
    chunks: int
    capture_s: float


def _extension_or_none():
    if not torch.cuda.is_available():
        return None
    try:
        return importlib.import_module("mlbricks._bolt_dynamic_cuda")
    except ImportError:
        return None


class DynamicBOLTGraphEngine:
    """One-token graph replay for the qualified BOLT-16/32 LM adapter protocol.

    ``first_token`` is the token already predicted by prefill; the engine yields
    it first, then consumes cache slots for subsequent tokens.  The caller owns
    and must retain ``model`` and the preallocated fixed-capacity cache buffers.
    Never move the model or replace cache tensors after constructing the engine.
    """

    def __init__(self, model, caches, first_token, prompt_length: int,
                 native_ext=None, *, enable_graph=True, strict_graph=False,
                 chunk_tokens=512):
        if not isinstance(prompt_length, int) or prompt_length < 1:
            raise ValueError("prompt_length must be a positive integer")
        if not isinstance(first_token, torch.Tensor) or first_token.numel() != 1:
            raise ValueError("first_token must be a one-element tensor")
        self.model = model
        self.caches = list(caches)
        if not self.caches:
            raise ValueError("at least one cache is required")
        self.position = prompt_length
        self.device = first_token.device
        self.input_token = first_token.detach().clone().long().reshape(1, 1)
        self.pos_gpu = torch.tensor([self.position], dtype=torch.int64, device=self.device)
        self.ext = native_ext if native_ext is not None else _extension_or_none()
        self.chunk_tokens = int(chunk_tokens)
        if not 128 <= self.chunk_tokens <= 4096 or self.chunk_tokens % 8:
            raise ValueError("chunk_tokens must be a multiple of 8 from 128 to 4096")
        self.strict_graph = bool(strict_graph)
        self.graphs = {}
        self.capture_info = []
        self.fallback_reason = None
        self._sent_first = False
        self.created_at = time.perf_counter()
        self.capacity = min(
            int(c.c.shape[2]) if hasattr(c, "c") else int(c.k.shape[2])
            for c in self.caches
        )
        if self.position > self.capacity:
            raise ValueError("prompt_length exceeds cache capacity")
        self._can_graph = self._check_support(enable_graph)
        if not self._can_graph and self.strict_graph:
            raise RuntimeError("CUDA Graph unavailable: " + str(self.fallback_reason))
        self._ptrs = self._pointer_signature()

    def _check_support(self, enable_graph):
        def reject(reason):
            self.fallback_reason = reason
            return False

        if not enable_graph:
            return reject("disabled by caller")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return reject("CUDA GPU unavailable")
        if self.ext is None or not callable(getattr(self.ext, "decode", None)):
            return reject("optional dynamic native CUDA extension unavailable")
        if self.model.training:
            return reject("model must be in eval mode")
        if any(p.requires_grad for p in self.model.parameters()):
            return reject("graph path requires frozen model weights")
        if getattr(self.model, "kind", None) not in ("bolt16", "bolt32"):
            return reject("model layout not graph-qualified; eager fallback")
        for name in ("layers", "embedding", "final_norm", "head"):
            if not hasattr(self.model, name):
                return reject("missing qualified LM adapter attribute: " + name)
        if len(self.caches) != len(self.model.layers):
            return reject("cache/layer count mismatch")
        if self.model.embedding.weight.dtype != torch.float16:
            return reject("experimental graph path qualified for FP16 weights only")
        r = 16 if self.model.kind == "bolt16" else 32
        for block, cache in zip(self.model.layers, self.caches):
            mixer = getattr(block, "mixer", None)
            if mixer is None or not all(hasattr(block, n) for n in ("norm1", "norm2", "ffn")):
                return reject("unsupported residual block")
            if (getattr(mixer, "latent_dim", None) != r
                or getattr(mixer, "position", None) is not None
                or not all(hasattr(mixer, n) for n in ("_project", "_scale", "out_proj"))
                or not hasattr(cache, "c") or not hasattr(cache, "state")
                or cache.c.dtype not in (torch.float16, torch.float32)
                or cache.state is None or cache.state.dtype != torch.float32
                or cache.scratch is None or cache.scratch.dtype != torch.float32
                or cache.c.device != self.device or cache.state.device != self.device
                or cache.c.shape[0] != 1 or cache.c.shape[1] != mixer.num_heads
                or cache.c.shape[-1] != r or cache.c.size(2) != self.capacity):
                return reject("unsupported cache/mixer; use eager fallback")
        return True

    def _pointer_signature(self):
        result = {"token": self.input_token.data_ptr(), "position": self.pos_gpu.data_ptr()}
        for name, param in self.model.named_parameters():
            result["weight:" + name] = param.data_ptr()
        for i, cache in enumerate(self.caches):
            for attr in ("c", "state", "scratch", "k", "v"):
                value = getattr(cache, attr, None)
                if isinstance(value, torch.Tensor):
                    result[f"{i}.{attr}"] = value.data_ptr()
        return result

    def _variant(self):
        r = 16 if self.model.kind == "bolt16" else 32
        threshold = 1536 if r == 16 else 896
        if self.position + 1 < threshold:
            return ("fast", 0, 1)
        bucket = max(1024, 1 << (self.position).bit_length())
        bucket = min(bucket, self.capacity)
        chunks = (bucket + self.chunk_tokens - 1) // self.chunk_tokens
        return ("split", bucket, chunks)

    @staticmethod
    def _clone_caches(caches):
        result = []
        for cache in caches:
            cloned = copy.copy(cache)
            for attr in ("c", "state", "scratch"):
                value = getattr(cache, attr, None)
                if isinstance(value, torch.Tensor):
                    setattr(cloned, attr, value.clone())
            result.append(cloned)
        return result

    def _decode_dynamic(self, caches, token, pos_gpu, mode, chunks):
        # Do not read the position back to Python inside this captured function.
        x = self.model.embedding(token)
        for layer, cache in zip(self.model.layers, caches):
            mixer = layer.mixer
            q, _, current = mixer._project(layer.norm1(x), start_pos=0)
            q = q.float().contiguous()
            if current.dtype != cache.c.dtype:
                current = current.to(cache.c.dtype)
            out = self.ext.decode(q, current.contiguous(), cache.c, cache.scratch,
                                  pos_gpu, float(mixer._scale()), self.chunk_tokens,
                                  chunks, mode == "split")
            out = out.transpose(1, 2).contiguous().reshape(1, 1, -1)
            if mixer.avq is not None:
                out, state = mixer.avq.step(out, cache.state, _v2_cache_inplace=True)
                if state.data_ptr() != cache.state.data_ptr():
                    cache.state.copy_(state)
            x = x + mixer.out_proj(out)
            x = x + layer.ffn(layer.norm2(x))
        logits = self.model.head(self.model.final_norm(x))
        return logits.argmax(dim=-1)

    def _run_one(self, caches, token, pos_gpu, mode, chunks):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            next_token = self._decode_dynamic(caches, token, pos_gpu, mode, chunks)
            token.copy_(next_token)
            pos_gpu.add_(1)

    def _capture_core(self, mode, bucket, chunks):
        key = (mode, bucket, chunks)
        if key in self.graphs:
            return self.graphs[key]
        scratch_caches = self._clone_caches(self.caches)
        scratch_token = self.input_token.clone()
        scratch_pos = self.pos_gpu.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                scratch_pos.copy_(self.pos_gpu)
                scratch_token.copy_(self.input_token)
                self._run_one(scratch_caches, scratch_token, scratch_pos, mode, chunks)
        torch.cuda.current_stream(device=self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        del scratch_caches, scratch_token, scratch_pos
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(graph, stream=stream):
            self._run_one(self.caches, self.input_token, self.pos_gpu, mode, chunks)
        torch.cuda.synchronize(self.device)
        self.capture_info.append(CaptureInfo(mode, bucket, chunks, time.perf_counter()-started))
        if self._pointer_signature() != self._ptrs:
            raise RuntimeError("CUDA graph capture changed live tensor pointers")
        # Guarded subclass restores every live tensor before making this graph visible.
        return graph

    def _capture(self, mode, bucket, chunks):
        # Overridden by GuardedBOLTGraphEngine; never use unguarded capture.
        raise RuntimeError("Capture must use GuardedBOLTGraphEngine")

    def _eager_decode_next(self):
        with torch.no_grad():
            if callable(getattr(self.model, "decode_one", None)):
                if (self.device.type == "cuda" and
                    next(self.model.parameters()).dtype == torch.float16):
                    with torch.autocast("cuda", dtype=torch.float16):
                        token, _ = self.model.decode_one(self.input_token, self.caches, self.position)
                else:
                    token, _ = self.model.decode_one(self.input_token, self.caches, self.position)
            elif callable(getattr(self.model, "decode_step", None)):
                logits, next_caches = self.model.decode_step(
                    self.input_token, self.caches, position=self.position)
                token = logits[:, -1:, :].argmax(-1)
                self.caches[:] = list(next_caches)
            else:
                raise NotImplementedError(
                    "Eager fallback requires model.decode_one(token, caches, pos) "
                    "or model.decode_step(token, caches, position=pos)")
            self.input_token.copy_(token.reshape(1, 1))
            self.position += 1
            self.pos_gpu.fill_(self.position)
        return int(self.input_token.item())

    def next_id(self):
        if not self._sent_first:
            self._sent_first = True
            return int(self.input_token.item())
        if self.position >= self.capacity:
            raise ValueError("cache capacity exhausted")
        if self._can_graph:
            mode, bucket, chunks = self._variant()
            try:
                graph = self._capture(mode, bucket, chunks)
            except GraphCaptureRecovered as exc:
                if self.strict_graph:
                    raise RuntimeError("Strict CUDA graph capture failed") from exc
                self._can_graph = False
                self.fallback_reason = "capture recovered; eager fallback: " + str(exc)
            else:
                # A failed replay might have mutated state: never silently retry.
                graph.replay()
                self.position += 1
                for cache in self.caches:
                    cache.length = self.position
                if self._pointer_signature() != self._ptrs:
                    raise RuntimeError("live tensor pointers changed after graph replay")
                return int(self.input_token.item())
        return self._eager_decode_next()

    def generate(self, max_new_tokens: int, *, eos_id=None):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if max_new_tokens and self.position + max_new_tokens - 1 > self.capacity:
            raise ValueError("requested generation exceeds preallocated cache capacity")
        for _ in range(max_new_tokens):
            token = self.next_id()
            yield token
            if eos_id is not None and token == eos_id:
                return

    @property
    def using_graph(self):
        return self._can_graph

    def status(self):
        return dict(using_graph=self._can_graph, position=self.position,
                    cache_capacity=self.capacity, graph_variants=len(self.graphs),
                    captures=[vars(c) for c in self.capture_info],
                    fallback_reason=self.fallback_reason,
                    pointer_stable=(self._ptrs == self._pointer_signature()))


class GraphCaptureRecovered(RuntimeError):
    """A host-side graph capture error was rolled back without changing pointers."""


class GuardedBOLTGraphEngine(DynamicBOLTGraphEngine):
    """Capture with complete in-place rollback of all live GPU buffers.

    Fatal CUDA faults are never classified as recoverable; if restoring state
    fails, the original failure is propagated instead of continuing to eager.
    """
    _ATTRS = ("c", "state", "scratch", "k", "v")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _save_state(self):
        self._sync()
        cache_states = []
        for cache in self.caches:
            cache_states.append((int(cache.length), {
                attr: getattr(cache, attr).detach().clone()
                for attr in self._ATTRS
                if isinstance(getattr(cache, attr, None), torch.Tensor)
            }))
        return dict(position=self.position, token=self.input_token.detach().clone(),
                    pos_gpu=self.pos_gpu.detach().clone(), caches=cache_states,
                    ptrs=self._pointer_signature())

    def _restore_state(self, saved):
        with torch.no_grad():
            self.input_token.copy_(saved["token"])
            self.pos_gpu.copy_(saved["pos_gpu"])
            self.position = saved["position"]
            for cache, (length, values) in zip(self.caches, saved["caches"]):
                for name, value in values.items():
                    getattr(cache, name).copy_(value)
                cache.length = length
        self._sync()
        if self._pointer_signature() != saved["ptrs"]:
            raise RuntimeError("graph capture rollback changed tensor pointers")

    def _capture(self, mode, bucket, chunks):
        key = (mode, bucket, chunks)
        if key in self.graphs:
            return self.graphs[key]
        saved = self._save_state()
        try:
            graph = self._capture_core(mode, bucket, chunks)
            self._restore_state(saved)
        except Exception as exc:
            # If the context is poisoned, even restoration will fail: propagate.
            try:
                self._restore_state(saved)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "CUDA capture failed and rollback could not be verified; "
                    "do not continue in this CUDA context") from rollback_exc
            raise GraphCaptureRecovered(f"{type(exc).__name__}: {exc}") from exc
        self.graphs[key] = graph
        return graph


class BoltGraphEngine(GuardedBOLTGraphEngine):
    """Opt-in graph engine; unsupported model layouts use eager decoding.

    Example::

        engine = BoltGraphEngine.from_prompt(model, prompt_ids, max_new_tokens=256)
        for token_id in engine.generate(256, eos_id=eos_id):
            print(token_id, flush=True)

    ``from_prompt`` supports the qualified WikiLM adapter and the public
    BoltModel eager fallback.  The public BoltModel native graph fast-path is
    NOT qualified and must not be silently enabled by this patch.
    """

    @classmethod
    @torch.no_grad()
    def from_prompt(cls, model, prompt_ids, *, max_new_tokens: int,
                    max_context=None, native_ext=None,
                    enable_graph=True, strict_graph=False, chunk_tokens=512):
        if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must be a torch tensor [batch, sequence]")
        if prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
            raise ValueError("only batch=1 with a nonempty prompt is supported")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        required = int(prompt_ids.shape[1]) + max(0, max_new_tokens - 1)
        capacity = required if max_context is None else int(max_context)
        if capacity < required:
            raise ValueError("max_context is smaller than prompt plus generated tokens")
        if model.training:
            raise ValueError("call model.eval() before generation")
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device, dtype=torch.long)
        if callable(getattr(model, "prepare_generation", None)):
            model.prepare_generation(batch_size=1, max_context=capacity)
            prepared = model._prepared_generation
            logits, caches = model.prefill(prompt_ids, cache_buffers=prepared["caches"])
            first = logits[:, -1:, :].argmax(dim=-1)
        else:
            first, caches = model.prefill(prompt_ids, capacity)
        return cls(model, caches, first, int(prompt_ids.shape[1]), native_ext,
                   enable_graph=enable_graph, strict_graph=strict_graph,
                   chunk_tokens=chunk_tokens)


__all__ = ["BoltGraphEngine", "DynamicBOLTGraphEngine", "GuardedBOLTGraphEngine",
           "GraphCaptureRecovered", "CaptureInfo"]
