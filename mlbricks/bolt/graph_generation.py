"""Experimental, Triton-free greedy CUDA Graph decoding for modern BOLT models.

The graph captures a complete BoltModel decoder step with device-resident cache
position, full-capacity masked PyTorch SDPA, in-place AVQ state and token feedback.
This is a *one-shot* generator: construct a new session for each prompt. It does
not use torch.compile, custom kernels or Triton.
"""
from __future__ import annotations

import torch
from .presets import BoltPresetCache
from .tensor_versions import inference_safe_version


class BoltCudaGraphGenerator:
    """One-shot graph for fixed prompt and generation lengths.

    Only greedy sampling, modern BOLT, default FFN/residual, and no positional
    encoding are currently supported. CUDA graph capture cost is deliberately
    excluded from the steady-state decode benchmark, but included in real setup.
    """
    def __init__(self, model, input_ids: torch.Tensor, max_new_tokens: int, *, warmup: int = 2):
        if not torch.cuda.is_available() or input_ids.device.type != 'cuda':
            raise RuntimeError('CUDA Graph generation requires a CUDA GPU and CUDA input_ids')
        if model.training:
            raise RuntimeError('call model.eval() before CUDA Graph generation')
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError('input_ids must be [batch, prompt_length>=1]')
        if max_new_tokens < 0 or int(warmup) < 0:
            raise ValueError('max_new_tokens and warmup must be non-negative')
        self.model = model
        self.prompt = input_ids.detach()
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_length = int(input_ids.size(1))
        self.capacity = self.prompt_length + self.max_new_tokens
        if self.capacity > model.config.context:
            raise ValueError('generation would exceed model context')
        if model.config.position != 'none':
            raise NotImplementedError('CUDA Graph backend currently requires position="none"')
        if model.config.ffn != 'standard' or model.config.residual != 'standard':
            raise NotImplementedError('CUDA Graph backend currently supports standard FFN and residual')
        if model.config.dropout != 0:
            raise NotImplementedError('CUDA Graph backend currently requires dropout=0')
        for block in model.blocks:
            a = block.attn
            if not a.is_modern or not a._modern.optimized_inference or a._modern.backend != 'pytorch':
                raise ValueError('CUDA Graph backend requires modern BOLT with backend="pytorch" '
                                 'and enable_bolt_optimized_inference(True)')
        if next(model.parameters()).device != input_ids.device:
            raise ValueError('model and input_ids must be on the same CUDA device')
        # Ensure optional kernels are loaded before capture, never compile in a graph.
        # This leaves the default PyTorch-only generator untouched.
        if hasattr(model, 'prepare_bolt_v05_native'):
            model.prepare_bolt_v05_native()
        self._used = False
        if self.max_new_tokens == 0:
            self.output = input_ids.clone()
            self.graph = None
            return
        # torch.inference_mode() can create Parameters with no _version counter.
        # Track data pointers and tracked versions where available.
        self._weights = tuple((p, inference_safe_version(p), p.data_ptr()) for p in model.parameters())
        self.caches = [BoltPresetCache.allocate(
            input_ids.size(0), a.num_heads, self.capacity, a.latent_dim, a.avq_dim,
            device=input_ids.device, dtype=next(model.parameters()).dtype)
            for a in (block.attn._modern for block in model.blocks)]
        self.position = torch.full((), self.prompt_length, device=input_ids.device, dtype=torch.long)
        self.positions = torch.arange(self.capacity, device=input_ids.device, dtype=torch.long)
        self.tokens = torch.empty((input_ids.size(0), self.capacity), device=input_ids.device,
                                  dtype=input_ids.dtype)
        self.static_token = torch.zeros_like(input_ids[:, :1])
        self._seed()
        if self.max_new_tokens == 1:
            self.graph = None
            return

        # Warm on another CUDA stream before capture. The subsequent re-prefill
        # restores both key and AVQ state without replacing their tensor addresses.
        stream = torch.cuda.Stream(device=input_ids.device)
        stream.wait_stream(torch.cuda.current_stream(input_ids.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                self.position.fill_(self.prompt_length)
                self._graph_step()
        torch.cuda.current_stream(input_ids.device).wait_stream(stream)
        self.position.fill_(self.prompt_length)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._graph_step()
        self._seed()  # undo state changes from warmup/capture

    @torch.inference_mode()
    def _seed(self):
        self.position.fill_(self.prompt_length)
        logits, _ = self.model.prefill(self.prompt, cache_buffers=self.caches)
        self.tokens[:, :self.prompt_length].copy_(self.prompt)
        self.static_token.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
        self.tokens[:, self.prompt_length:self.prompt_length+1].copy_(self.static_token)

    def _graph_step(self):
        # Call only from warmup or capture, never directly after capture.
        x = self.model.token_embedding(self.static_token)
        previous_mixer = None
        previous_state = None
        for block, cache in zip(self.model.blocks, self.caches):
            normalized = block.norm1(x)
            update = block.attn._modern.graph_decode_with_cache(
                normalized, cache, position=self.position, positions=self.positions)
            x, previous_mixer, previous_state = block.finish(
                x, update, previous_mixer, previous_state)
        logits = self.model.lm_head(self.model.final_norm(x))
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self.static_token.copy_(next_token)
        self.tokens.scatter_(1, (self.position + 1).reshape(1,1).expand_as(next_token), next_token)
        self.position.add_(1)

    @torch.inference_mode()
    def generate(self):
        if self._used:
            raise RuntimeError('CUDA Graph generator is one-shot; construct a new session')
        self._used = True
        if self.max_new_tokens == 0:
            return self.output
        if self.model.training:
            raise RuntimeError('model was switched to training after graph capture')
        if any(inference_safe_version(p) != v or p.data_ptr() != ptr for p,v,ptr in self._weights):
            raise RuntimeError('model parameters changed after graph capture; recreate graph')
        for _ in range(self.max_new_tokens - 1):
            self.graph.replay()
        # One-shot: don't reuse after generation unless the session is rebuilt.
        return self.tokens


__all__ = ['BoltCudaGraphGenerator']
