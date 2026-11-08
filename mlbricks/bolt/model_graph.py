"""Experimental full-model, one-token CUDA Graph replay for public BOLT LMs.

This path differs deliberately from the per-layer V5 CUDA Graphs.  It executes
embedding, every BOLT mixer, residual/norm/FFN and LM head in **one graph**.
Nested BOLT graphs are never invoked. An int64 CUDA position allows the same
captured graph to advance the cache across tokens. The existing guarded engine
owns capture/rollback, pointer checks and separate fast/split graph variants.

Restrictions are conservative. Unsupported modes use the existing eager model
path. No inference speedup is claimed until the GPU qualification suite runs.
"""
from __future__ import annotations

import os
import importlib
from typing import Any

import torch

from .graph_engine import BoltGraphEngine, GraphCaptureRecovered


class BoltModelGraphEngine(BoltGraphEngine):
    """Whole-model CUDA Graph engine for a *qualified* ``BoltModel``.

    Construct with ``from_prompt`` after ``model.eval()`` and converting the
    model to FP16. Use ``generate_tensor`` to keep token IDs on GPU and avoid
    one CPU synchronization per token. ``strict_graph=True`` makes unsupported
    configurations an error; otherwise callers must check ``using_graph``.
    """

    @staticmethod
    def eligibility(model: Any, prompt=None, max_new_tokens: int = 64) -> str | None:
        """Return a rejection reason without modifying the model or caches."""
        if not torch.cuda.is_available():
            return 'CUDA is unavailable'
        if not isinstance(max_new_tokens, int) or max_new_tokens < 2:
            return 'need at least two generated tokens to replay a decode graph'
        if model.training:
            return 'model is training; call eval() first'
        if not hasattr(model, 'blocks') or not hasattr(model, 'token_embedding'):
            return 'not a public BOLT language model'
        if not hasattr(model, 'final_norm') or not hasattr(model, 'lm_head'):
            return 'missing complete language-model layers'
        if not hasattr(model, 'config') or not getattr(model, 'blocks', None):
            return 'model has no BOLT blocks'
        if getattr(model.config, 'position', 'none') != 'none' or model.position is not None:
            return 'position embeddings/RoPE are not graph-qualified'
        if getattr(model.config, 'ffn', None) != 'standard':
            return 'stateful/nonstandard FFNs are not graph-qualified'
        if getattr(model.config, 'residual', None) != 'standard':
            return 'nonstandard residual controllers are not graph-qualified'
        if float(getattr(model.config, 'dropout', 0.0)) != 0.0:
            return 'dropout must be zero'
        if getattr(model.config, 'precision', 'fp32') != 'fp16':
            return 'FP16 configured inference required (do not silently cast weights)'
        if str(getattr(model, 'backend', 'auto')) == 'pytorch':
            return 'explicit PyTorch backend'
        if str(getattr(model, 'graph', getattr(model.config, 'graph', 'auto'))).lower() in ('off', 'false', '0'):
            return 'graph disabled'
        try:
            p = next(model.parameters())
        except StopIteration:
            return 'model has no parameters'
        if p.device.type != 'cuda' or p.dtype != torch.float16:
            return 'model weights must already be CUDA FP16'
        if any(v.dtype != torch.float16 or v.device != p.device for v in model.parameters()):
            return 'all model weights must be CUDA FP16'
        if getattr(model, '_compile_generation_regions', False):
            return 'independently compiled decode regions are not graph-qualified'
        if bool(getattr(torch.cuda, 'is_current_stream_capturing', lambda: False)()):
            return 'already inside an outer CUDA Graph capture'
        if os.getenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '').strip().lower() in ('1', 'true', 'yes', 'on'):
            return 'opt-in V2 Strict CUDA state route not qualified in whole-model graph'
        try:
            native = importlib.import_module('mlbricks._bolt_dynamic_cuda')
        except ImportError:
            return 'optional device-position CUDA dynamic decoder extension unavailable'
        if not callable(getattr(native, 'decode', None)):
            return 'dynamic decoder does not export decode()'
        if prompt is not None:
            if not isinstance(prompt, torch.Tensor) or prompt.ndim != 2 or prompt.size(0) != 1:
                return 'only one sequence with [1, T] input is supported'
            if prompt.size(1) == 0:
                return 'empty prompt'
            if prompt.size(1) + max_new_tokens > model.config.context:
                return 'generation would exceed model context'
        for block in model.blocks:
            if not (hasattr(block, 'norm1') and hasattr(block, 'norm2')
                    and hasattr(block, 'ffn') and hasattr(block, 'residual1')
                    and hasattr(block, 'residual2')):
                return 'nonstandard block layout'
            layer = block.attn
            if not getattr(layer, 'is_modern', False):
                return 'legacy BOLT layer'
            mixer = getattr(layer, '_modern', None)
            if mixer is None or mixer.latent_dim not in (16, 32):
                return 'only BOLT16 and BOLT32 are supported'
            if mixer.position is not None or not mixer.causal or float(mixer.dropout) != 0.0:
                return 'RoPE/noncausal/dropout BOLT mixer not qualified'
            if getattr(mixer, 'backend', '') == 'pytorch':
                return 'BOLT layer explicitly selects PyTorch'
            if getattr(mixer, 'avq', None) is not None:
                if mixer.avq.return_map != 'identity':
                    return 'only Identity AVQ return map is graph-qualified'
                if mixer.avq.width != mixer.avq_dim:
                    return 'AVQ state width does not match attention width'
        return None

    @classmethod
    def supports(cls, model, prompt=None, max_new_tokens=64) -> bool:
        """Low-cost policy gate. Does not claim capture has succeeded."""
        return cls.eligibility(model, prompt, max_new_tokens) is None

    def _check_support(self, enable_graph):
        def reject(message):
            self.fallback_reason = message
            return False
        if not enable_graph:
            return reject('disabled by caller')
        reason = self.eligibility(self.model)
        if reason is not None:
            return reject(reason)
        if self.ext is None or not callable(getattr(self.ext, 'decode', None)):
            return reject('dynamic CUDA BOLT decoder extension unavailable')
        if len(self.caches) != len(self.model.blocks):
            return reject('cache count differs from model blocks')
        for block, cache in zip(self.model.blocks, self.caches):
            mixer = block.attn._modern
            # Public prefill allocates the scratch lazily on first eager decode.
            # Allocate it once *before* capturing, preserving its address.
            if (hasattr(cache, 'scratch') and cache.scratch is None
                    and hasattr(cache, 'c') and cache.c.ndim == 4):
                cache.scratch = torch.empty(cache.c.shape[:3], device=self.device,
                                            dtype=torch.float32)
            if (not hasattr(cache, 'c') or not hasattr(cache, 'state')
                or not hasattr(cache, 'scratch') or cache.c.dtype != torch.float16
                or cache.scratch is None or cache.scratch.dtype != torch.float32
                or cache.c.device != self.device or cache.c.ndim != 4
                or cache.c.size(0) != 1 or cache.c.size(1) != mixer.num_heads
                or cache.c.size(2) != self.capacity or cache.c.size(3) != mixer.latent_dim):
                return reject('unsupported or unallocated BOLT cache')
            if mixer.avq is not None:
                if (cache.state is None or cache.state.dtype != torch.float32
                    or cache.state.shape != (1, mixer.avq_dim)
                    or cache.state.device != self.device):
                    return reject('FP32 AVQ cache state missing or malformed')
        return True

    def _variant(self):
        # Use the largest BOLT width's crossover, so all layers in the graph
        # use the same fast/split mode without mixing incompatible modes.
        widths = [block.attn._modern.latent_dim for block in self.model.blocks]
        threshold = min(1536 if r == 16 else 896 for r in widths)
        if self.position + 1 < threshold:
            return 'fast', 0, 1
        bucket = max(1024, 1 << self.position.bit_length())
        bucket = min(bucket, self.capacity)
        chunks = (bucket + self.chunk_tokens - 1) // self.chunk_tokens
        return 'split', bucket, chunks

    def _decode_dynamic(self, caches, token, pos_gpu, mode, chunks):
        # This function is *captured* into one outer CUDA Graph. Neither a
        # BoltPreset.decode_with_cache nor a per-layer graph session is used.
        x = self.model.token_embedding(token)
        previous_mixer = None
        ffn_state = None
        for block, cache in zip(self.model.blocks, caches):
            mixer = block.attn._modern
            q, _, c_now = mixer._project(block.norm1(x), start_pos=0)
            if c_now.dtype != cache.c.dtype:
                c_now = c_now.to(cache.c.dtype)
            out = self.ext.decode(
                q.float().contiguous(), c_now.contiguous(), cache.c,
                cache.scratch, pos_gpu, float(mixer._scale()),
                self.chunk_tokens, chunks, mode == 'split'
            )
            out = out.transpose(1, 2).contiguous().reshape(1, 1, -1)
            if mixer.avq is not None:
                # Cache state stays FP32. The optional strict V2 external
                # replacement is explicitly rejected by eligibility().
                out, state = mixer.avq.step(out, cache.state, _v2_cache_inplace=True)
                if state.data_ptr() != cache.state.data_ptr():
                    cache.state.copy_(state)
            mixer_update = mixer.out_proj(out)
            x, previous_mixer, ffn_state = block.finish(
                x, mixer_update, previous_mixer, ffn_state
            )
        logits = self.model.lm_head(self.model.final_norm(x))
        return logits.argmax(dim=-1)

    def _weight_versions(self):
        return tuple((name, id(p), p.data_ptr(), p._version)
                     for name, p in self.model.named_parameters())

    def _capture(self, mode, bucket, chunks):
        key = (mode, bucket, chunks)
        before = self._weight_versions()
        if self.graphs and getattr(self, '_graph_versions', before) != before:
            raise RuntimeError('Model weights changed after graph capture; create a new graph engine')
        if key not in self.graphs:
            graph = super()._capture(mode, bucket, chunks)
            if self._weight_versions() != before:
                raise RuntimeError('Model weights mutated during CUDA graph capture')
            self._graph_versions = before
            return graph
        if getattr(self, '_graph_versions', self._weight_versions()) != self._weight_versions():
            raise RuntimeError('Model weights changed after graph capture; create a new graph engine')
        return self.graphs[key]

    @torch.no_grad()
    def generate_tensor(self, max_new_tokens: int):
        """Generate GPU-resident token tensor without per-token host synchronization.

        The first output token is the prefill prediction. Graph replay then
        advances GPU position, all layer caches, and next token in one launch.
        Only a *verified* capture error permits eager fallback; failed replay
        is fatal because GPU cache state might have advanced already.
        """
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be nonnegative')
        if max_new_tokens and self.position + max_new_tokens - 1 > self.capacity:
            raise ValueError('generation exceeds preallocated cache capacity')
        result = torch.empty((1, max_new_tokens), device=self.device, dtype=torch.long)
        if not max_new_tokens:
            return result
        result[:, 0:1].copy_(self.input_token)
        self._sent_first = True
        for i in range(1, max_new_tokens):
            if self._can_graph:
                mode, bucket, chunks = self._variant()
                try:
                    graph = self._capture(mode, bucket, chunks)
                except GraphCaptureRecovered as exc:
                    if self.strict_graph:
                        raise RuntimeError('Strict whole-model capture failed') from exc
                    self._can_graph = False
                    self.fallback_reason = f'capture recovered: {exc}'
                else:
                    if self._pointer_signature() != self._ptrs:
                        raise RuntimeError('Model cache pointers changed before replay')
                    # A replay failure may have mutated state; never retry it.
                    graph.replay()
                    self.position += 1
                    for cache in self.caches:
                        cache.length = self.position
                    result[:, i:i+1].copy_(self.input_token)
                    continue
            logits, next_caches = self.model.decode_step(
                self.input_token, self.caches, position=self.position
            )
            self.caches[:] = list(next_caches)
            self.input_token.copy_(logits[:, -1:, :].argmax(-1))
            self.position += 1
            self.pos_gpu.fill_(self.position)
            result[:, i:i+1].copy_(self.input_token)
        return result


__all__ = ['BoltModelGraphEngine']
