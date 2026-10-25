"""Configurable BOLT and post-softmax AVQ presets.

Two INDEPENDENT controls:
    mode='quality' -> BOLT plus AVQ width 198 (default)
    mode='fast'    -> BOLT without AVQ
    cache='normal'-> 32 latent features/head (default)
    cache='tight' -> 16 latent features/head

The new optional native BOLT feature/AVQ CUDA extension is separate from legacy
normalized-latent BOLT; those checkpoints are NOT compatible. The AVQ=198 preset is
new and has not been validated on the user's benchmark (which used AVQ=192).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from . import native_backend as _native


@dataclass
class BoltPresetCache:
    """Preallocated shared-latent cache and optional per-layer AVQ state."""
    c: torch.Tensor  # [batch, heads, capacity, latent]
    state: torch.Tensor | None = None  # [batch, avq_dim], float32
    length: int = 0
    scratch: torch.Tensor | None = None  # [batch, heads, capacity] float32, native softmax

    @property
    def capacity(self):
        return self.c.size(2)

    @classmethod
    def allocate(cls, batch, heads, capacity, latent_dim, avq_dim, *, device, dtype):
        return cls(torch.empty(batch, heads, capacity, latent_dim, device=device, dtype=dtype),
                   torch.zeros(batch, avq_dim, device=device, dtype=torch.float32) if avq_dim else None)

    def reset(self):
        self.length = 0
        if self.state is not None:
            self.state.zero_()
        return self

    def load_prefix(self, c, state):
        n = int(c.size(2))
        if c.shape[:2] != self.c.shape[:2] or c.size(3) != self.c.size(3) or n > self.capacity:
            raise ValueError('BOLT cache prefix shape exceeds capacity or has incompatible dimensions')
        if (self.state is None) != (state is None):
            raise ValueError('BOLT cache AVQ state presence mismatch')
        self.c[:, :, :n, :].copy_(c)
        if state is not None:
            if state.shape != self.state.shape:
                raise ValueError('BOLT cache AVQ state shape mismatch')
            self.state.copy_(state)
        self.length=n
        return self


class _RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x):
        z=x.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True)+self.eps) * self.weight.float()).to(x.dtype)


def _affine_scan(a, b):
    """Differentiable parallel prefix for S[t]=a[t]*S[t-1]+b[t]."""
    a,b=a.float(),b.float()
    step=1
    while step<a.shape[1]:
        a_old,b_old=a,b
        a=torch.cat((a_old[:,:step], a_old[:,step:]*a_old[:,:-step]),dim=1)
        b=torch.cat((b_old[:,:step], b_old[:,step:]+a_old[:,step:]*b_old[:,:-step]),dim=1)
        step*=2
    return b


class PostSoftmaxAVQ(nn.Module):
    """AVQ read-gated memory after causal BOLT attention (198D default)."""
    def __init__(self, input_dim, width=198):
        super().__init__()
        self.use_native = False
        if width<1:
            raise ValueError('avq_dim must be positive')
        self.width=int(width)
        self.project=nn.Linear(input_dim,3*width,bias=True)
        self.norm=_RMSNorm(width)
        self.back=nn.Linear(width,input_dim,bias=False)
        self.gamma=nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            nn.init.normal_(self.project.weight,0.,.01)
            nn.init.zeros_(self.project.bias)
            self.project.bias[:width].fill_(math.log(.98/.02))
            self.project.weight[2*width:].zero_()  # zero read-gate initially
            nn.init.normal_(self.back.weight,0.,.02)

    def components(self,o):
        raw=self.project(o)
        a,v,r=raw.split(self.width,dim=-1)
        return torch.sigmoid(a).float(), (.1*torch.tanh(v)).float(), torch.tanh(r).float()

    def forward(self,o,*,return_state=False):
        a,v,r=self.components(o)
        if self.use_native and o.is_cuda:
            s=_native.avq_scan(a.contiguous(),v.contiguous())
        else:
            s=_affine_scan(a,v)
        h=self.norm(s)*(1.+r)
        update=self.back(h).to(o.dtype)
        enhanced=o+self.gamma.to(o.dtype)*update
        return (enhanced,s[:,-1,:].contiguous()) if return_state else enhanced

    def step(self,o,state):
        a,v,r=self.components(o)
        if self.use_native and o.is_cuda:
            s=_native.avq_step(a[:,0,:],v[:,0,:],state.float())
        else:
            s=a[:,0,:]*state.float()+v[:,0,:]
        h=(self.norm(s)*(1.+r[:,0,:]))[:,None,:]
        update=self.back(h).to(o.dtype)
        return o+self.gamma.to(o.dtype)*update, s


class BoltPreset(nn.Module):
    """BOLT attention with optional post-softmax AVQ and compact Value cache.

    mode: quality (AVQ-198 by default) or fast (no AVQ).
    cache: normal (latent=32) or tight (latent=16).
    When installed, the independent native CUDA extension fuses BOLT feature
    generation and AVQ recurrence, and optionally the cached attention decoder.
    The old rho-normalized BOLT extension is mathematically incompatible.
    """
    def __init__(self,d_model,num_heads,*,mode='quality',cache='normal',avq_dim=198,
                 bias=False,dropout=0.,causal=True,backend='auto',position=None):
        super().__init__()
        if d_model%num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        mode=str(mode).lower()
        cache=str(cache).lower()
        if mode not in ('quality','fast'):
            raise ValueError("mode must be 'quality' or 'fast'")
        if cache not in ('normal','tight'):
            raise ValueError("cache must be 'normal' or 'tight'")
        if backend not in ('auto','pytorch','native'):
            raise ValueError('backend must be auto/pytorch/native')
        self.d_model=int(d_model)
        self.num_heads=int(num_heads)
        self.head_dim=d_model//num_heads
        self.latent_dim=32 if cache=='normal' else 16
        self.mode=mode
        self.cache_mode=cache
        self.avq_dim=int(avq_dim) if mode=='quality' else 0
        self.backend=backend
        self.causal=bool(causal)
        self.dropout=float(dropout)
        # RoPE is supported on the latent Query/Key vectors.
        from ..position import RoPE
        if position is None or position=='none':
            self.position=None
        elif position=='rope' or position=='rotary':
            self.position=RoPE(self.latent_dim)
        elif isinstance(position,RoPE):
            self.position=position
        else:
            raise ValueError('position must be none, rope or a RoPE instance')
        width=self.num_heads*self.latent_dim
        # Separate parameters simplify inspection; benchmark used one packed Q/U/G projection.
        self.q_proj=nn.Linear(d_model,width,bias=bias)
        self.c_proj=nn.Linear(d_model,width,bias=bias)
        self.g_proj=nn.Linear(d_model,width,bias=bias)
        self.p=nn.Parameter(torch.ones(self.num_heads,self.latent_dim))
        self.out_proj=nn.Linear(width,d_model,bias=bias)
        self.avq=PostSoftmaxAVQ(width,self.avq_dim) if mode=='quality' else None
        self.is_modern=True

    @property
    def cache_bytes_per_head_token(self):
        # FP16 compact C only, unlike legacy BOLT's C + rho.
        return 2*self.latent_dim

    @property
    def baseline_cache_bytes_per_head_token(self):
        return 4*self.head_dim

    @property
    def cache_saving_percent(self):
        return 100*(1.-self.cache_bytes_per_head_token/self.baseline_cache_bytes_per_head_token)

    def set_backend(self,backend,*,recursive=True):
        if backend not in ('auto','pytorch','native'):
            raise ValueError('backend must be auto, pytorch, or native')
        self.backend=backend
        return self

    def resolved_backend(self):
        if self.backend == 'pytorch':
            return 'pytorch-SDPA-BOLT/AVQ'
        if _native.get_extension() is not None:
            return 'native-CUDA-BOLT-features+AVQ-scan+cached-decode / SDPA-prefill'
        return 'native-unavailable' if self.backend == 'native' else 'pytorch-SDPA-fallback'

    def _native_enabled(self, tensor):
        if self.backend == 'pytorch':
            return False
        if not tensor.is_cuda or _native.get_extension() is None:
            if self.backend == 'native':
                raise RuntimeError("Modern BOLT backend='native' requires a CUDA tensor and "
                                   "mlbricks._bolt_modern_cuda. Build with MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1")
            return False
        if tensor.dtype not in (torch.float16, torch.float32):
            if self.backend == 'native':
                raise TypeError('Modern BOLT native supports FP16 or FP32 features')
            return False
        return True

    def set_position(self,position):
        from ..position import RoPE
        if position is None or position=='none':self.position=None
        elif position in ('rope','rotary'):self.position=RoPE(self.latent_dim)
        elif isinstance(position,RoPE):self.position=position
        else:raise ValueError('invalid position')
        return self

    def _project(self,x,start_pos=0):
        b,t,_=x.shape
        q=self.q_proj(x).reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        u=self.c_proj(x).reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        g=self.g_proj(x).reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        if self._native_enabled(x) and _native.supports_features(q,u,g,self.p):
            q,c=_native.features(q.contiguous(),u.contiguous(),g.contiguous(),self.p)
        else:
            if self.backend=='native':
                raise RuntimeError('Native BOLT features require CUDA q/u/g with matching FP16/FP32 dtypes and FP32 P')
            c=u*(1.+torch.tanh(g))
            # BOLT learned per-feature query multiplier, not legacy dynamic rho.
            q=q*self.p[None,:,None,:]
        if self.position is not None:
            q=self.position(q,start_pos=int(start_pos))
            k=self.position(c,start_pos=int(start_pos))
        else:k=c
        return q,k,c

    def _attend(self,x,start_pos=0):
        b,t,_=x.shape
        native = self._native_enabled(x)
        if self.avq is not None:self.avq.use_native=native
        q,k,c=self._project(x,start_pos)
        a=F.scaled_dot_product_attention(q.to(k.dtype) if q.dtype!=k.dtype else q,k,c,is_causal=self.causal,
            dropout_p=self.dropout if self.training else 0.,scale=self.head_dim**-.5)
        return a.transpose(1,2).contiguous().reshape(b,t,-1),c

    def forward(self,x):
        if x.ndim!=3 or x.size(-1)!=self.d_model:
            raise ValueError(f'x must be [B,T,{self.d_model}]')
        o,_=self._attend(x)
        if self.avq is not None:o=self.avq(o)
        return self.out_proj(o)

    @torch.no_grad()
    def prefill_with_cache(self,x,*,start_pos=0):
        if x.ndim!=3 or x.size(-1)!=self.d_model:
            raise ValueError(f'x must be [B,T,{self.d_model}]')
        if int(start_pos)!=0 and self.avq is not None:
            raise ValueError('AVQ prefill must begin at position 0; use decode_step for continuation')
        o,c=self._attend(x,start_pos)
        state=None
        if self.avq is not None:
            o,state=self.avq(o,return_state=True)
        return self.out_proj(o),(c.contiguous(),state)

    prefill=prefill_with_cache

    @torch.no_grad()
    def project_decode_state(self,x,*,start_pos):
        if x.ndim==2:x=x[:,None,:]
        if x.ndim!=3 or x.size(1)!=1 or x.size(-1)!=self.d_model:
            raise ValueError('decode expects [B,1,D]')
        q,k,c=self._project(x,start_pos)
        return q,c  # k is obtained by rotating the cached C when RoPE is enabled

    @torch.no_grad()
    def decode_step(self,x,cache):
        if x.ndim==2:x=x[:,None,:]
        if x.ndim!=3 or x.size(1)!=1 or x.size(-1)!=self.d_model:
            raise ValueError('decode_step expects [B,1,D]')
        if not isinstance(cache,(tuple,list)) or len(cache)!=2:
            raise TypeError('modern Bolt cache must be (c_history, avq_state)')
        c_past,state=cache
        t=int(c_past.size(2))
        native=self._native_enabled(x)
        if native and self.position is not None and self.backend=='native':
            raise RuntimeError('Native cached BOLT decode does not yet support RoPE')
        if self.avq is not None:self.avq.use_native=native
        q,k_now,c_now=self._project(x,start_pos=t)
        c=torch.cat((c_past,c_now),dim=2)
        if self.position is not None:
            # Stored unrotated C; rotate all historical Keys with their absolute positions.
            keys=self.position(c,start_pos=0)
        else:keys=c
        if native and self.position is None:
            scratch=torch.empty((*c.shape[:2], c.shape[2]),device=c.device,dtype=torch.float32)
            o=_native.decode_append(q.contiguous(),c_now.contiguous(),c.contiguous(),scratch,t,self.head_dim**-.5)
        else:
            if native and self.backend == 'native' and self.position is not None:
                raise RuntimeError('Native cached BOLT decode does not yet support RoPE; use backend=auto')
            o=F.scaled_dot_product_attention(q,keys,c,is_causal=False,dropout_p=0.,scale=self.head_dim**-.5)
        o=o.transpose(1,2).contiguous().reshape(x.size(0),1,-1)
        if self.avq is not None:
            if state is None:raise ValueError('missing AVQ state')
            o,state=self.avq.step(o,state)
        return self.out_proj(o),(c,state)

    @torch.no_grad()
    def decode_with_cache(self,x,cache:BoltPresetCache,*,position:int):
        if x.ndim==2:x=x[:,None,:]
        pos=int(position)
        if not isinstance(cache,BoltPresetCache):
            raise TypeError('expected BoltPresetCache')
        if pos !=cache.length:
            raise ValueError('position must match cache.length')
        if pos<0 or pos>=cache.capacity:
            raise ValueError('decode position exceeds cache capacity')
        native=self._native_enabled(x)
        if native and self.position is not None and self.backend=='native':
            raise RuntimeError('Native cached BOLT decode does not yet support RoPE')
        if self.avq is not None:self.avq.use_native=native
        q,_,c_now=self._project(x,start_pos=pos)
        if not (native and self.position is None):
            cache.c[:,:,pos:pos+1,:].copy_(c_now)
        c=cache.c[:,:,:pos+1,:]
        keys=self.position(c,start_pos=0) if self.position is not None else c
        if native and self.position is None:
            if cache.scratch is None:
                cache.scratch=torch.empty((*cache.c.shape[:2],cache.capacity),device=x.device,dtype=torch.float32)
            o=_native.decode_append(q.contiguous(),c_now.contiguous(),cache.c,cache.scratch,pos,self.head_dim**-.5)
        else:
            if native and self.backend == 'native' and self.position is not None:
                raise RuntimeError('Native cached BOLT decode does not yet support RoPE; use backend=auto')
            o=F.scaled_dot_product_attention(q,keys,c,is_causal=False,dropout_p=0.,scale=self.head_dim**-.5)
        o=o.transpose(1,2).contiguous().reshape(x.size(0),1,-1)
        if self.avq is not None:
            o,s=self.avq.step(o,cache.state)
            cache.state.copy_(s)
        cache.length=pos+1
        return self.out_proj(o)

    lightning_prefill=prefill_with_cache
    lightning_step=decode_step

    def extra_repr(self):
        return (f'mode={self.mode}, cache={self.cache_mode}, latent={self.latent_dim}, '
                f'avq_dim={self.avq_dim}, backend={self.backend}')


__all__=['BoltPreset','BoltPresetCache','PostSoftmaxAVQ']
