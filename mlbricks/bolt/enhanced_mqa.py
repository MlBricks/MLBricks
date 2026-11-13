"""Enhanced multi-query attention with PRE-attention AVQ and output head gates.

Independent architecture from BOLT's shared-C attention. This module keeps
one separate K and V head, so an FP16 cache costs 4 * head_dim bytes/token/layer
(128 B for d_model=384, heads=12). The shipped BOLT32 stores one 32D C vector
PER HEAD and costs 2 * heads * 32 B/token/layer (768 B for 12 heads). These are cache-size statements, not training
peak-memory or quality guarantees.

Only the PyTorch/SDPA backend is implemented for Enhanced MQA. It is portable
across CPU and PyTorch-supported accelerators; native BOLT kernels must not be
silently applied to different MQA mathematics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .presets import BoltStateMemory
from ..position import RoPE


@dataclass
class EnhancedMQACache:
    """Fixed-capacity, separate K and V cache plus FP32 pre-AVQ state."""
    k: torch.Tensor  # [batch, 1, capacity, head_dim] (RoPE already applied)
    v: torch.Tensor  # [batch, 1, capacity, head_dim]
    state: torch.Tensor  # [batch, d_model] FP32, one state per sequence
    length: int = 0

    @property
    def capacity(self) -> int:
        return self.k.size(2)

    @classmethod
    def allocate(cls, batch, capacity, head_dim, state_dim, *, device, dtype, kv_heads=1):
        shape = (batch, kv_heads, capacity, head_dim)
        return cls(
            torch.empty(shape, device=device, dtype=dtype),
            torch.empty(shape, device=device, dtype=dtype),
            torch.zeros((batch, state_dim), device=device, dtype=torch.float32),
        )

    def reset(self):
        self.length = 0
        self.state.zero_()
        return self

    def load_prefix(self, k, v, state):
        n = k.size(2)
        if k.shape != v.shape or k.shape[:2] != self.k.shape[:2] or k.size(3) != self.k.size(3):
            raise ValueError('Enhanced MQA K/V prefix shapes are incompatible')
        if n > self.capacity or state.shape != self.state.shape:
            raise ValueError('Enhanced MQA prefix exceeds capacity or state shape')
        self.k[:, :, :n].copy_(k)
        self.v[:, :, :n].copy_(v)
        self.state.copy_(state.float())
        self.length = n
        return self


class EnhancedMQA(nn.Module):
    """Pre-AVQ384 MQA + learnable sigmoid retention + query-head output gating.

    ``forget_gate_init=0.95`` initializes the sigmoid *bias*, not a fixed gate.
    The project matrix and bias continue learning during training. ``head_gate``
    is initialized to give a factor of exactly 1, but is also fully trainable.

    Training/prefill recurrences are parallel differentiable affine scans;
    cached decode updates the identical recurrent state one token at a time.
    """
    is_modern = True

    def __init__(self, d_model: int, num_heads: int, *, forget_gate_init: float = 0.95,
                 head_output_gate: bool = True, bias: bool = False,
                 dropout: float = 0.0, causal: bool = True,
                 backend: str = 'auto', position=None, state_memory: bool = True,
                 kv_heads: int = 1, avq_dim: int | None = None):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError('d_model must be positive and divisible by num_heads')
        if not 0 < float(forget_gate_init) < 1:
            raise ValueError('forget_gate_init must be strictly between 0 and 1')
        if backend not in ('auto', 'pytorch', 'native'):
            raise ValueError('backend must be auto, pytorch or native')
        if position is not None and not isinstance(position, RoPE):
            if isinstance(position, str) and position.lower() in ('rope', 'rotary'):
                position = RoPE(d_model // num_heads)
            elif isinstance(position, str) and position.lower() == 'none':
                position = None
            else:
                raise ValueError('position must be none, rope or a RoPE instance')

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        if kv_heads not in (1, num_heads):
            raise ValueError('kv_heads must be 1 (MQA) or num_heads (classic MHA)')
        self.kv_heads = int(kv_heads)
        self.latent_dim = self.head_dim  # cache-width compatibility attribute
        memory_width = self.d_model if avq_dim is None else int(avq_dim)
        if memory_width < 1:
            raise ValueError('avq_dim must be positive')
        self.avq_dim = memory_width if state_memory else 0
        self.state_memory = bool(state_memory)
        self.backend = backend
        self.graph = 'off'  # native BOLT graphs do not apply to this attention
        self.precision = 'auto'
        self.causal = bool(causal)
        self.dropout = float(dropout)
        self.position = position
        self.forget_gate_init = float(forget_gate_init)
        self.head_output_gate = bool(head_output_gate)

        # The pre-AVQ block processes the current hidden state BEFORE any Q/K/V
        # projection. Its state is stored separately from the attention K/V cache.
        if self.state_memory:
            self.avq = BoltStateMemory(
                self.d_model, memory_width, init_policy='balanced',
                latent_dim=self.head_dim, device='cpu', backend='pytorch',
                return_map='auto',
            )
            with torch.no_grad():
                self.avq.project.bias[:memory_width].fill_(
                    math.log(self.forget_gate_init / (1. - self.forget_gate_init))
                )
        else:
            self.avq = None
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        if self.head_output_gate:
            self.head_gate = nn.Linear(self.d_model, self.num_heads, bias=True)
            nn.init.zeros_(self.head_gate.weight)
            nn.init.zeros_(self.head_gate.bias)
        else:
            self.head_gate = None

    @property
    def cache_bytes_per_token(self) -> int:
        """FP16/BF16 K+V bytes per token/layer, not including state or weights."""
        return 4 * self.kv_heads * self.head_dim

    @property
    def cache_bytes_per_head_token(self) -> int:
        # Interface compatibility: MQA stores a *single* K/V head in total.
        return self.cache_bytes_per_token / self.num_heads

    @property
    def cache_saving_percent(self) -> float:
        return 100 * (1 - self.kv_heads / self.num_heads)

    def set_backend(self, backend, *, recursive=True):
        if backend not in ('auto', 'pytorch', 'native'):
            raise ValueError('backend must be auto, pytorch or native')
        self.backend = backend
        return self

    def graph_allowed(self) -> bool:
        # Native BOLT graphs implement different attention mathematics.
        return False

    def set_position(self, position):
        if isinstance(position, str) and position.lower() in ('rope', 'rotary'):
            position = RoPE(self.head_dim)
        elif isinstance(position, str) and position.lower() == 'none':
            position = None
        if position is not None and not isinstance(position, RoPE):
            raise ValueError('position must be none or RoPE')
        self.position = position
        return self

    def extra_repr(self):
        return (f'd_model={self.d_model}, heads={self.num_heads}, kv_heads={self.kv_heads}, '
                f'avq={self.state_memory}, head_gate={self.head_output_gate}')

    def set_graph(self, graph):
        # Record the public policy; incompatible BOLT native graphs remain disabled.
        self.graph = str(graph).lower()
        return self

    def resolved_backend(self):
        if self.backend == 'native':
            return 'native-unavailable (requires a compatible CUDA BOLT base kernel)'
        return 'pytorch_sdpa Enhanced MQA (no native BOLT dispatch)'

    def execution_info(self):
        return {'backend': 'pytorch_sdpa', 'reason': 'Enhanced MQA architecture'}

    def allocate_cache(self, *, batch_size, capacity, device=None, dtype=None):
        weight = next(self.parameters())
        return EnhancedMQACache.allocate(batch_size, capacity, self.head_dim,
                                        self.avq_dim, device=device or weight.device,
                                        dtype=dtype or weight.dtype, kv_heads=self.kv_heads)

    def _check_backend(self):
        if self.backend == 'native':
            raise RuntimeError('backend=native requires a compatible CUDA kernel; quantised/base SDPA has no native BOLT implementation')

    def _project(self, z, *, start_pos: int):
        b, t, _ = z.shape
        q = self.q_proj(z).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z).reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.position is not None:
            q = self.position(q, start_pos=start_pos)
            k = self.position(k, start_pos=start_pos)
        return q, k, v

    def _attend(self, q, k, v, *, causal):
        # Broadcasting the one K/V head avoids materializing a persistent
        # 12-head KV cache. PyTorch SDPA handles broadcast stride-zero heads.
        if self.kv_heads == 1:
            k = k.expand(-1, self.num_heads, -1, -1)
            v = v.expand(-1, self.num_heads, -1, -1)
        if k.dtype != q.dtype:
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.,
            scale=self.head_dim ** -0.5,
        )

    def _finish(self, o, z):
        b, _, t, _ = o.shape
        if self.head_gate is not None:
            # Factor = 1 at initialization, learned per-token and per-Q-head.
            gate = 2. * torch.sigmoid(self.head_gate(z))
            o = o * gate.transpose(1, 2).unsqueeze(-1).to(o.dtype)
        y = o.transpose(1, 2).contiguous().reshape(b, t, self.d_model)
        return self.out_proj(y)

    def forward(self, x):
        self._check_backend()
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f'x must have shape [B,T,{self.d_model}]')
        z = self.avq(x) if self.avq is not None else x
        q, k, v = self._project(z, start_pos=0)
        return self._finish(self._attend(q, k, v, causal=self.causal), z)

    @torch.no_grad()
    def prefill_with_cache(self, x, *, start_pos=0):
        self._check_backend()
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError(f'x must have shape [B,T,{self.d_model}]')
        if start_pos != 0:
            raise ValueError('Enhanced MQA prefill must start at 0; use decode_step to continue')
        if self.avq is not None:
            z, state = self.avq(x, return_state=True)
        else:
            z = x
            state = x.new_zeros((x.size(0), 0), dtype=torch.float32)
        q, k, v = self._project(z, start_pos=0)
        out = self._finish(self._attend(q, k, v, causal=self.causal), z)
        return out, (k.contiguous(), v.contiguous(), state)

    prefill = prefill_with_cache

    @torch.no_grad()
    def decode_step(self, x, cache):
        self._check_backend()
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.size(1) != 1 or x.size(-1) != self.d_model:
            raise ValueError(f'expected [B,1,{self.d_model}]')
        if not isinstance(cache, (tuple, list)) or len(cache) != 3:
            raise TypeError('Enhanced MQA cache must be (K, V, FP32_state)')
        old_k, old_v, state = cache
        z, state = self.avq.step(x, state) if self.avq is not None else (x, state)
        q, k_now, v_now = self._project(z, start_pos=old_k.size(2))
        k = torch.cat((old_k, k_now.to(old_k.dtype)), dim=2)
        v = torch.cat((old_v, v_now.to(old_v.dtype)), dim=2)
        out = self._finish(self._attend(q, k, v, causal=False), z)
        return out, (k, v, state)

    @torch.no_grad()
    def decode_with_cache(self, x, cache: EnhancedMQACache, *, position: int):
        self._check_backend()
        if not isinstance(cache, EnhancedMQACache):
            raise TypeError('expected EnhancedMQACache')
        pos = int(position)
        if pos != cache.length or not 0 <= pos < cache.capacity:
            raise ValueError('decode position must equal cache.length and fit within capacity')
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.size(1) != 1 or x.size(-1) != self.d_model:
            raise ValueError(f'expected [B,1,{self.d_model}]')
        z, state = self.avq.step(x, cache.state) if self.avq is not None else (x, cache.state)
        q, k_now, v_now = self._project(z, start_pos=pos)
        cache.k[:, :, pos:pos + 1].copy_(k_now)
        cache.v[:, :, pos:pos + 1].copy_(v_now)
        cache.state.copy_(state)
        cache.length = pos + 1
        out = self._finish(self._attend(q, cache.k[:, :, :pos + 1],
                                        cache.v[:, :, :pos + 1], causal=False), z)
        return out


class ClassicMHA(EnhancedMQA):
    """Original multi-head Q/K/V attention, with optional pre-attention AVQ.

    With state_memory=False this is standard causal MHA + RoPE with no AVQ.
    The optional AVQ is independent of the attention K/V representation.
    """
    def __init__(self, d_model, num_heads, *, state_memory=True,
                 head_output_gate=True, **kwargs):
        kwargs.pop('kv_heads', None)
        super().__init__(d_model, num_heads, kv_heads=num_heads,
                         state_memory=state_memory,
                         head_output_gate=head_output_gate, **kwargs)

    def resolved_backend(self):
        if self.backend == 'native':
            return 'native-unavailable (requires a compatible CUDA classic kernel)'
        return 'pytorch_sdpa classic MHA (optional AVQ)'


__all__ = ['EnhancedMQA', 'EnhancedMQACache', 'ClassicMHA']
