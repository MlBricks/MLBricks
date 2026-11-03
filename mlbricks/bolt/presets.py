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
from . import accelerated as _accelerated


@dataclass
class BoltCache:
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


class BoltStateMemory(nn.Module):
    """AVQ read-gated memory after causal BOLT attention (198D default)."""
    def __init__(self, input_dim, width=198, *, init_policy="balanced", latent_dim=32,
                 device='auto', backend='auto'):
        super().__init__()
        if backend not in ('auto', 'native', 'pytorch'):
            raise ValueError('backend must be auto, native or pytorch')
        self.backend = backend
        self.use_native = backend != 'pytorch'
        if init_policy not in ("legacy", "balanced"):
            raise ValueError("init_policy must be legacy or balanced")
        self.init_policy = init_policy
        self.latent_dim = int(latent_dim)
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
        if init_policy == "balanced":
            # Same gate initialization as the quality-selected configuration.
            # Preserve the global RNG state so later layers are initialized
            # identically when comparing independent memory policies.
            rng = torch.get_rng_state()
            try:
                self.project.reset_parameters()
            finally:
                torch.set_rng_state(rng)
            with torch.no_grad():
                self.gamma.fill_(0.25)
        if str(device).lower() == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was explicitly requested but is unavailable')
        self.to(torch.device(device))

    def components_from_raw(self, raw):
        a, v, r = raw.split(self.width, dim=-1)
        return torch.sigmoid(a).float(), (0.1 * torch.tanh(v)).float(), torch.tanh(r).float()

    def components(self, o):
        return self.components_from_raw(self.project(o))

    def forward(self,o,*,return_state=False):
        if o.device != self.project.weight.device:
            o = o.to(self.project.weight.device, non_blocking=True)
        if self.backend == 'native' and o.is_cuda and not (_accelerated.available() or _native.get_extension()):
            raise RuntimeError('Native BOLT state memory requires a compiled CUDA extension')
        if self.use_native and o.is_cuda and _accelerated.available():
            raw = self.project(o)
            za, zv, zr = raw.split(self.width, dim=-1)
            s = _accelerated.scan(za, zv, latent_dim=self.latent_dim)
            if s is not None:
                r = torch.tanh(zr).float()
            else:
                a, v, r = self.components_from_raw(raw)
                s = _affine_scan(a, v)
        else:
            a,v,r=self.components(o)
            if self.use_native and o.is_cuda and _native.get_extension() is not None:
                s=_native.avq_scan(a.contiguous(),v.contiguous())
            else:
                s=_affine_scan(a,v)
        h=self.norm(s)*(1.+r)
        update=self.back(h).to(o.dtype)
        enhanced=o+self.gamma.to(o.dtype)*update
        return (enhanced,s[:,-1,:].contiguous()) if return_state else enhanced

    def step(self,o,state):
        if o.device != self.project.weight.device:
            o = o.to(self.project.weight.device, non_blocking=True)
        if state.device != o.device:
            state = state.to(o.device, non_blocking=True)
        a,v,r=self.components(o)
        if self.use_native and o.is_cuda and _native.get_extension() is not None:
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
                 bias=False,dropout=0.,causal=True,backend='auto',position=None,
                 projection='separate', attention_scale='head', memory_init='legacy'):
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
        if projection not in ('separate', 'packed'):
            raise ValueError('projection must be separate or packed')
        if attention_scale not in ('head', 'latent'):
            raise ValueError('attention_scale must be head or latent')
        if memory_init not in ('legacy', 'balanced'):
            raise ValueError('memory_init must be legacy or balanced')
        self.projection = projection
        self.attention_scale = attention_scale
        self.memory_init = memory_init
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
        if projection == 'packed':
            # Construct the normal parameters first to preserve RNG sequencing
            # and checkpoint values, then pack them into one trainable Parameter.
            weights = (self.q_proj.weight, self.c_proj.weight, self.g_proj.weight)
            self.register_parameter('packed_qug_weight',
                                    nn.Parameter(torch.cat([w.detach() for w in weights], 0)))
            biases = (self.q_proj.bias, self.c_proj.bias, self.g_proj.bias)
            packed_bias = (nn.Parameter(torch.cat([b.detach() for b in biases], 0))
                           if all(b is not None for b in biases) else None)
            self.register_parameter('packed_qug_bias', packed_bias)
            self.q_proj, self.c_proj, self.g_proj = nn.Identity(), nn.Identity(), nn.Identity()
        self.p=nn.Parameter(torch.ones(self.num_heads,self.latent_dim))
        self.out_proj=nn.Linear(width,d_model,bias=bias)
        self.avq=BoltStateMemory(width,self.avq_dim, init_policy=memory_init, latent_dim=self.latent_dim, device='cpu', backend=backend) if mode=='quality' else None
        self.is_modern=True

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.projection == 'packed':
            w = destination.pop(prefix + 'packed_qug_weight')
            width = self.num_heads * self.latent_dim
            for index, name in enumerate(('q_proj', 'c_proj', 'g_proj')):
                destination[prefix + name + '.weight'] = w[index*width:(index+1)*width].clone()
            b = destination.pop(prefix + 'packed_qug_bias', None)
            if b is not None:
                for index, name in enumerate(('q_proj', 'c_proj', 'g_proj')):
                    destination[prefix + name + '.bias'] = b[index*width:(index+1)*width].clone()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        width = self.num_heads * self.latent_dim
        names = ('q_proj', 'c_proj', 'g_proj')
        if self.projection == 'packed':
            canonical = [prefix + name + '.weight' for name in names]
            if all(key in state_dict for key in canonical):
                state_dict[prefix + 'packed_qug_weight'] = torch.cat(
                    [state_dict.pop(key) for key in canonical], 0).contiguous()
            canonical_bias = [prefix + name + '.bias' for name in names]
            if self.packed_qug_bias is not None and all(key in state_dict for key in canonical_bias):
                state_dict[prefix + 'packed_qug_bias'] = torch.cat(
                    [state_dict.pop(key) for key in canonical_bias], 0).contiguous()
        elif prefix + 'packed_qug_weight' in state_dict:
            packed = state_dict.pop(prefix + 'packed_qug_weight')
            for index, name in enumerate(names):
                state_dict[prefix + name + '.weight'] = packed[index*width:(index+1)*width].clone()
            packed_bias = state_dict.pop(prefix + 'packed_qug_bias', None)
            if packed_bias is not None:
                for index, name in enumerate(names):
                    state_dict[prefix + name + '.bias'] = packed_bias[index*width:(index+1)*width].clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _scale(self):
        dimension = self.latent_dim if self.attention_scale == 'latent' else self.head_dim
        return dimension ** -0.5

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
            return 'pytorch BOLT reference'
        if _accelerated.available():
            return 'optimized CUDA state memory / compact attention'
        if _native.get_extension() is not None:
            return 'native CUDA BOLT features, state memory and cached decode'
        return 'native unavailable' if self.backend == 'native' else 'pytorch BOLT fallback'

    def _native_enabled(self, tensor):
        if self.backend == 'pytorch':
            return False
        if not tensor.is_cuda or (_native.get_extension() is None and not _accelerated.available()):
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
        if self.projection == 'packed':
            qug = F.linear(x, self.packed_qug_weight, self.packed_qug_bias)
            q, u, g = qug.split(self.num_heads * self.latent_dim, dim=-1)
        else:
            q, u, g = self.q_proj(x), self.c_proj(x), self.g_proj(x)
        q=q.reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        u=u.reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        g=g.reshape(b,t,self.num_heads,self.latent_dim).transpose(1,2)
        if self._native_enabled(x) and _native.supports_features(q,u,g,self.p):
            q,c=_native.features(q.contiguous(),u.contiguous(),g.contiguous(),self.p)
        else:
            if self.backend == 'native' and _native.get_extension() is not None:
                raise RuntimeError('Native BOLT features require compatible CUDA Q/U/G tensors and FP32 P')
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
            dropout_p=self.dropout if self.training else 0.,scale=self._scale())
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
        if native and _native.get_extension() is not None and self.position is None:
            scratch=torch.empty((*c.shape[:2], c.shape[2]),device=c.device,dtype=torch.float32)
            o=_native.decode_append(q.contiguous(),c_now.contiguous(),c.contiguous(),scratch,t,self._scale())
        else:
            if native and self.backend == 'native' and self.position is not None:
                raise RuntimeError('Native cached BOLT decode does not yet support RoPE; use backend=auto')
            o=F.scaled_dot_product_attention(q,keys,c,is_causal=False,dropout_p=0.,scale=self._scale())
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
        if not (native and _native.get_extension() is not None and self.position is None):
            cache.c[:,:,pos:pos+1,:].copy_(c_now)
        c=cache.c[:,:,:pos+1,:]
        keys=self.position(c,start_pos=0) if self.position is not None else c
        if native and _native.get_extension() is not None and self.position is None:
            if cache.scratch is None:
                cache.scratch=torch.empty((*cache.c.shape[:2],cache.capacity),device=x.device,dtype=torch.float32)
            o=_native.decode_append(q.contiguous(),c_now.contiguous(),cache.c,cache.scratch,pos,self._scale())
        else:
            if native and self.backend == 'native' and self.position is not None:
                raise RuntimeError('Native cached BOLT decode does not yet support RoPE; use backend=auto')
            o=F.scaled_dot_product_attention(q,keys,c,is_causal=False,dropout_p=0.,scale=self._scale())
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


# Compatibility names: stable imports, professionally named canonical components.
BoltPresetCache = BoltCache

class PostSoftmaxAVQ(BoltStateMemory):
    """Backwards-compatible memory component with historical CPU/gate defaults."""
    def __init__(self, input_dim, width=198, **kwargs):
        kwargs.setdefault('init_policy', 'legacy')
        kwargs.setdefault('device', 'cpu')
        kwargs.setdefault('backend', 'pytorch')
        super().__init__(input_dim, width, **kwargs)

__all__ = ['BoltPreset', 'BoltCache', 'BoltStateMemory',
           'BoltPresetCache', 'PostSoftmaxAVQ']
