"""Simple stock-arithmetic QKV language model with optional automatic CUDA Graph decoding.

Experimental candidate: graph selection is limited to the FP16 Tesla T4
configuration until the extended GPU correctness matrix is verified.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .bolt import Attention
from .bolt.auto_device import place_new_module, input_on_module
from .bolt.tensor_versions import inference_safe_version
from .bolt.inference_mirror import fp16_inference_mirror, drop_fp16_inference_mirror
from .components import FFN, RMSNorm


class _QKVBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads, backend='pytorch', dropout=0.0, causal=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = FFN(dim, 4 * dim, dropout=0.0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))

    def prefill(self, x, k_cache, v_cache):
        a, (k, v) = self.attn.prefill_with_cache(self.norm1(x))
        k_cache[:, :, :x.size(1), :].copy_(k)
        v_cache[:, :, :x.size(1), :].copy_(v)
        x = x + a
        return x + self.ffn(self.norm2(x))

    def decode(self, x, k_cache, v_cache, pos, positions):
        normed = self.norm1(x)
        b, _, _ = normed.shape
        qkv = self.attn.qkv(normed)
        q, k, v = qkv.chunk(3, dim=-1)
        h, d = self.attn.num_heads, self.attn.head_dim
        q = q.view(b, 1, h, d).transpose(1, 2)
        k = k.view(b, 1, h, d).transpose(1, 2)
        v = v.view(b, 1, h, d).transpose(1, 2)
        index = pos.reshape(1, 1, 1, 1).expand(b, h, 1, d)
        k_cache.scatter_(2, index, k)
        v_cache.scatter_(2, index, v)
        mask = (positions <= pos).view(1, 1, 1, -1)
        z = F.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=mask,
            is_causal=False, dropout_p=0.0,
            scale=1.0/math.sqrt(float(d)),
        )
        z = z.transpose(1, 2).reshape(b, 1, h*d)
        x = x + self.attn.out_proj(z)
        return x + self.ffn(self.norm2(x))


class AttentionModel(nn.Module):
    """Ready-made QKV causal LM; normal users call train/eval/generate only.

    The module starts on CUDA automatically when available, just like BoltModel.
    Initial parameters remain FP32; this class never silently casts model weights.
    """
    def __init__(self, vocab_size, *, dim=384, heads=12, layers=6, context=2048):
        super().__init__()
        if min(int(vocab_size), int(dim), int(heads), int(layers), int(context)) <= 0:
            raise ValueError('model dimensions and vocab_size must be positive')
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.vocab_size, self.dim, self.heads = int(vocab_size), int(dim), int(heads)
        self.context = int(context)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(_QKVBlock(dim, heads) for _ in range(layers))
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self._mlbricks_last_inference = 'not-run'
        self._mlbricks_last_precision = 'original'
        place_new_module(self)

    def train(self, mode: bool = True):
        if mode:
            drop_fp16_inference_mirror(self)
        return super().train(mode)

    def _apply(self, fn):
        # Explicit model.to()/half()/float() must release a stale GPU mirror.
        drop_fp16_inference_mirror(self)
        return super()._apply(fn)

    def forward(self, input_ids, *, targets=None):
        input_ids = input_on_module(input_ids, self)
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError('input_ids must be nonempty [batch, sequence]')
        if input_ids.size(1) > self.context:
            raise ValueError('sequence exceeds context')
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            return logits
        targets = targets.to(logits.device)
        return logits, F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                       targets.reshape(-1), ignore_index=-1)

    def _allocate_cache(self, batch, capacity):
        p = next(self.parameters())
        shape = (batch, self.heads, capacity, self.dim//self.heads)
        # Zero initialize: masked future positions must never contain stale data.
        return [(torch.zeros(shape, device=p.device, dtype=p.dtype),
                 torch.zeros(shape, device=p.device, dtype=p.dtype)) for _ in self.blocks]

    @torch.no_grad()
    def _prefill_cache(self, prompt, caches):
        x = self.token_embedding(prompt)
        for block, (k, v) in zip(self.blocks, caches):
            x = block.prefill(x, k, v)
        return self.lm_head(self.final_norm(x))[:, -1:, :]

    @torch.no_grad()
    def _decode_cache(self, token, caches, pos, positions):
        x = self.token_embedding(token)
        for block, (k, v) in zip(self.blocks, caches):
            x = block.decode(x, k, v, pos, positions)
        return self.lm_head(self.final_norm(x))

    def _auto_graph_eligible(self, ids, max_new_tokens, temperature, top_k, *, allow_fp32=False):
        p = next(self.parameters())
        return bool(
            (allow_fp32 or not self.training) and p.is_cuda and p.dtype in ((torch.float16, torch.float32) if allow_fp32 else (torch.float16,))
            and ids.is_cuda and torch.cuda.get_device_capability(p.device) == (7, 5)
            and ids.size(0) == 1 and max_new_tokens >= 2
            and temperature <= 0 and top_k is None
        )

    def inference_info(self):
        p = next(self.parameters())
        mirrored = self._mlbricks_last_precision == 'fp32_master_fp16_inference'
        reason = ('validated FP16 T4 graph via an independent inference mirror'
                  if self._mlbricks_last_inference=='strict_cuda_graph' and mirrored
                  else 'validated FP16 T4 greedy generation'
                  if self._mlbricks_last_inference=='strict_cuda_graph'
                  else 'insufficient GPU memory for separate FP16 inference mirror'
                  if getattr(self,'_mlbricks_fp16_mirror_oom',False)
                  else 'FP32 retained; graph unsupported for request or device'
                  if p.dtype!=torch.float16
                  else 'configuration not in the validated CUDA Graph matrix')
        return {'device': str(p.device), 'dtype': str(p.dtype),
                'precision_policy':('fp32_master_fp16_inference_mirror' if mirrored
                                    else 'preserve_model_dtype'),
                'inference_dtype':'torch.float16' if mirrored else str(p.dtype),
                'last_execution':self._mlbricks_last_inference,
                'reason':reason,
                'graph_eligibility':'experimental_fp16_t4_greedy'}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, *, temperature=0.0, top_k=None):
        ids = input_on_module(input_ids, self)
        if ids.ndim != 2 or ids.size(1) < 1:
            raise ValueError('input_ids must be nonempty [batch, sequence]')
        if max_new_tokens < 0:
            raise ValueError('max_new_tokens must be non-negative')
        if ids.size(1) + max_new_tokens > self.context:
            raise ValueError('generation exceeds context')
        if max_new_tokens == 0:
            self._mlbricks_last_inference='no-tokens'
            return ids
        self._mlbricks_fp16_mirror_oom = False
        self._mlbricks_last_precision = 'original'
        # Mirror is only created for the already-validated FP16 T4 graph path.
        # The full FP32 model remains intact for training/checkpoint saving.
        if (next(self.parameters()).dtype == torch.float32 and
                self._auto_graph_eligible(ids, max_new_tokens, temperature,
                                          top_k, allow_fp32=True)):
            try:
                mirror, _ = fp16_inference_mirror(
                    self, lambda: type(self)(
                        self.vocab_size, dim=self.dim, heads=self.heads,
                        layers=len(self.blocks), context=self.context))
                result = mirror.generate(ids, max_new_tokens,
                                         temperature=temperature, top_k=top_k)
                self._mlbricks_last_inference = mirror._mlbricks_last_inference
                self._mlbricks_last_precision = 'fp32_master_fp16_inference'
                return result
            except torch.cuda.OutOfMemoryError:
                drop_fp16_inference_mirror(self)
                self._mlbricks_fp16_mirror_oom = True
        was_training = self.training
        self.eval()
        try:
            if self._auto_graph_eligible(ids, max_new_tokens, temperature, top_k):
                with torch.autocast('cuda', dtype=torch.float16):
                    session = QKVCudaGraphGenerator(self, ids, max_new_tokens)
                    result = session.generate()
                self._mlbricks_last_inference = 'strict_cuda_graph'
                return result
            self._mlbricks_last_inference='pytorch_eager'
            capacity=ids.size(1)+max_new_tokens
            caches=self._allocate_cache(ids.size(0), capacity)
            logits=self._prefill_cache(ids, caches)
            if temperature > 0:
                scores=logits[:, -1, :]/float(temperature)
                if top_k is not None and 0 < top_k < scores.size(-1):
                    cutoff = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                    scores = scores.masked_fill(scores < cutoff, float('-inf'))
                token = torch.multinomial(scores.softmax(-1), 1)
            else:
                token = logits.argmax(-1)
            tokens = [token]
            positions = torch.arange(capacity, device=ids.device)
            pos = torch.full((), ids.size(1), device=ids.device, dtype=torch.long)
            for _ in range(max_new_tokens-1):
                logits=self._decode_cache(token, caches, pos, positions)
                if temperature > 0:
                    scores=logits[:, -1, :]/float(temperature)
                    if top_k is not None and 0 < top_k < scores.size(-1):
                        cutoff = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
                        scores = scores.masked_fill(scores < cutoff, float('-inf'))
                    token = torch.multinomial(scores.softmax(-1), 1)
                else:
                    token = logits.argmax(-1)
                tokens.append(token)
                pos.add_(1)
            return torch.cat([ids, torch.cat(tokens, dim=1)], dim=1)
        finally:
            if was_training:
                self.train()


class QKVCudaGraphGenerator:
    """One-shot graph using exactly the same static-cache arithmetic as eager."""
    def __init__(self, model, prompt, max_new_tokens, *, warmup=2):
        if model.training or not prompt.is_cuda:
            raise RuntimeError('QKV CUDA Graph requires an eval CUDA model')
        if prompt.size(1) + max_new_tokens > model.context or max_new_tokens < 2:
            raise ValueError('invalid graph context or generation length')
        self.model = model
        self.prompt = prompt.detach()
        self.max_new_tokens = max_new_tokens
        self.length = prompt.size(1)
        self.capacity = self.length + max_new_tokens
        self._used = False
        self._weights = tuple((p, inference_safe_version(p), p.data_ptr()) for p in model.parameters())
        self.caches = model._allocate_cache(prompt.size(0), self.capacity)
        self.position = torch.full((), self.length, device=prompt.device, dtype=torch.long)
        self.positions = torch.arange(self.capacity, device=prompt.device)
        self.tokens = torch.empty((prompt.size(0),self.capacity), device=prompt.device, dtype=prompt.dtype)
        self.static_token = torch.zeros_like(prompt[:, :1])
        self._seed()
        stream = torch.cuda.Stream(device=prompt.device)
        stream.wait_stream(torch.cuda.current_stream(prompt.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(warmup):
                self.position.fill_(self.length)
                self._step()
        torch.cuda.current_stream(prompt.device).wait_stream(stream)
        self.position.fill_(self.length)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._step()
        self._seed()

    @torch.inference_mode()
    def _seed(self):
        self.position.fill_(self.length)
        logits=self.model._prefill_cache(self.prompt,self.caches)
        self.tokens[:, :self.length].copy_(self.prompt)
        self.static_token.copy_(logits.argmax(-1))
        self.tokens[:, self.length:self.length+1].copy_(self.static_token)

    def _step(self):
        logits=self.model._decode_cache(self.static_token,self.caches,
                                        self.position,self.positions)
        self.logits = logits
        new_token=logits.argmax(-1)
        self.static_token.copy_(new_token)
        self.tokens.scatter_(1,(self.position+1).reshape(1,1).expand_as(new_token),new_token)
        self.position.add_(1)

    @torch.inference_mode()
    def generate(self):
        if self._used:
            raise RuntimeError('CUDA Graph generator is one-shot')
        self._used = True
        if self.model.training:
            raise RuntimeError('model changed to training after graph capture')
        if any(inference_safe_version(p)!=version or p.data_ptr()!=address
               for p,version,address in self._weights):
            raise RuntimeError('model weights changed after graph capture')
        for _ in range(self.max_new_tokens-1):
            self.graph.replay()
        return self.tokens

__all__ = ['AttentionModel', 'QKVCudaGraphGenerator']
