"""Keyless BOLT with one shared C head, pre-attention AVQ and no positional scheme.

The C cache contains only [B,1,T,D/H] values; no duplicate key cache is stored.
Both PyTorch training and incremental decoding follow the same equations.
The older per-head-C/post-AVQ native kernels are intentionally not dispatched.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .enhanced_mqa import EnhancedMQA


@dataclass
class SharedCCache:
    c: torch.Tensor                 # [B,1,capacity,head_dim]
    state: torch.Tensor             # [B,avq_dim], FP32 (or empty width when AVQ off)
    length: int = 0

    @property
    def capacity(self):
        return self.c.shape[2]

    @classmethod
    def allocate(cls, batch, capacity, head_dim, state_dim, *, device, dtype):
        return cls(torch.empty(batch, 1, capacity, head_dim, device=device, dtype=dtype),
                   torch.zeros(batch, state_dim, device=device, dtype=torch.float32))

    def reset(self):
        self.length = 0
        self.state.zero_()
        return self

    def load_prefix(self, c, state):
        n = c.size(2)
        if c.ndim != 4 or c.shape[:2] != self.c.shape[:2] or c.size(3) != self.c.size(3):
            raise ValueError('Shared C prefix dimensions do not match the allocated cache')
        if n > self.capacity or state.shape != self.state.shape:
            raise ValueError('Shared C prefix exceeds cache capacity or AVQ state size')
        self.c[:, :, :n, :].copy_(c)
        self.state.copy_(state.float())
        self.length = n
        return self


class SharedCKeyless(EnhancedMQA):
    """H independently learned query heads, one shared key/value-like C history.

    Z = PreAVQ(X); Q=Wq Z; U=Wu Z; G=Wg Z;
    C = U*(1+tanh(G)); O_h=SDPA(Q_h,C,C).

    A user-provided token-local Q/K transform may be set via set_qk_transform.
    When used, only raw C is cached; transformed keys are recomputed as needed.
    """
    def __init__(self, d_model, num_heads, *, kv_heads=None, **kwargs):
        if kv_heads is not None:
            raise ValueError('Keyless uses one shared C, not conventional kv_heads')
        super().__init__(d_model, num_heads, kv_heads=1, **kwargs)
        # Q uses d_model -> d_model; U/G each produce one shared head_dim.
        # Drop inherited K/V projections instead of storing unused weights.
        del self.k_proj
        del self.v_proj
        self.u_proj = nn.Linear(d_model, self.head_dim, bias=kwargs.get('bias', False))
        self.g_proj = nn.Linear(d_model, self.head_dim, bias=kwargs.get('bias', False))
        self.mode = 'keyless'
        self.kv_heads = None  # C is shared; there are no conventional K/V heads

    @property
    def cache_bytes_per_token(self):
        return 2 * self.head_dim  # FP16/BF16, shared C only

    @property
    def cache_bytes_per_head_token(self):
        return self.cache_bytes_per_token / self.num_heads

    @property
    def cache_saving_percent(self):
        return 100.0 * (1.0 - 1.0 / (2.0 * self.num_heads))

    def resolved_backend(self):
        if self.backend == 'native':
            return 'native-unavailable (legacy per-head-C kernels are incompatible)'
        return 'pytorch_sdpa Shared-C Keyless'

    def execution_info(self):
        return {'backend': 'pytorch_sdpa', 'reason': 'new shared-C keyless architecture'}

    def _check_backend(self):
        if self.backend == 'native':
            raise RuntimeError('Shared-C Keyless requires backend=auto/pytorch until new native kernels are validated')

    def _project(self, z, *, start_pos: int):
        b, t, _ = z.shape
        q = self.q_proj(z).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        u = self.u_proj(z).reshape(b, 1, t, self.head_dim)
        g = self.g_proj(z).reshape(b, 1, t, self.head_dim)
        c = u * (1.0 + torch.tanh(g))
        if self.position is not None:
            q = self.position(q, start_pos=start_pos)
        return q, c

    def _attend_shared(self, q, c, *, causal):
        key = self.position(c, start_pos=0) if self.position is not None else c
        if key.dtype != q.dtype:
            key = key.to(q.dtype)
            c = c.to(q.dtype)
        return F.scaled_dot_product_attention(
            q, key.expand(-1, self.num_heads, -1, -1),
            c.expand(-1, self.num_heads, -1, -1),
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.,
            scale=self.head_dim ** -0.5,
        )

    def forward(self, x):
        self._check_backend()
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f'x must have shape [B,T,{self.d_model}]')
        z = self.avq(x) if self.avq is not None else x
        q, c = self._project(z, start_pos=0)
        return self._finish(self._attend_shared(q, c, causal=self.causal), z)

    def allocate_cache(self, *, batch_size, capacity, device=None, dtype=None):
        weight = next(self.parameters())
        return SharedCCache.allocate(batch_size, capacity, self.head_dim, self.avq_dim,
                                     device=device or weight.device, dtype=dtype or weight.dtype)

    @torch.no_grad()
    def prefill_with_cache(self, x, *, start_pos=0):
        self._check_backend()
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f'x must have shape [B,T,{self.d_model}]')
        if start_pos:
            raise ValueError('Shared-C prefill must start at zero; continue with decode_step')
        if self.avq is not None:
            z, state = self.avq(x, return_state=True)
        else:
            z = x
            state = x.new_zeros((x.size(0), 0), dtype=torch.float32)
        q, c = self._project(z, start_pos=0)
        o = self._finish(self._attend_shared(q, c, causal=self.causal), z)
        return o, (c.contiguous(), state)

    prefill = prefill_with_cache

    @torch.no_grad()
    def decode_step(self, x, cache):
        self._check_backend()
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1:] != (1, self.d_model):
            raise ValueError(f'expected [B,1,{self.d_model}]')
        if not isinstance(cache, (tuple, list)) or len(cache) != 2:
            raise TypeError('Shared-C cache must be (C_history, FP32_AVQ_state)')
        previous, state = cache
        if previous.ndim != 4 or previous.size(1) != 1 or previous.size(3) != self.head_dim:
            raise ValueError('Shared-C history shape mismatch')
        z, state = self.avq.step(x, state) if self.avq is not None else (x, state)
        q, c_now = self._project(z, start_pos=previous.size(2))
        c = torch.cat((previous, c_now), dim=2)
        o = self._finish(self._attend_shared(q, c, causal=False), z)
        return o, (c, state)

    @torch.no_grad()
    def decode_with_cache(self, x, cache: SharedCCache, *, position: int):
        self._check_backend()
        if not isinstance(cache, SharedCCache):
            raise TypeError('expected SharedCCache')
        pos = int(position)
        if pos != cache.length or pos >= cache.capacity or pos < 0:
            raise ValueError('decode position must match cache length and fit capacity')
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1:] != (1, self.d_model):
            raise ValueError(f'expected [B,1,{self.d_model}]')
        z, state = self.avq.step(x, cache.state) if self.avq is not None else (x, cache.state)
        q, c_now = self._project(z, start_pos=pos)
        cache.c[:, :, pos:pos+1].copy_(c_now)
        cache.state.copy_(state)
        cache.length += 1
        o = self._finish(self._attend_shared(q, cache.c[:, :, :cache.length], causal=False), z)
        return o

    def extra_repr(self):
        return (f'd_model={self.d_model}, heads={self.num_heads}, one_shared_C={self.head_dim}, '
                f'avq={self.state_memory}, retention={self.retention}')


__all__ = ['SharedCKeyless', 'SharedCCache']
