"""Pre-attention AVQ state memory and affine scan. No historical attention preset."""
from __future__ import annotations

import math
from contextlib import nullcontext
# MLBRICKS_BOLT_NATIVE_FP16_CACHE_DTYPE_FIX_V1
import torch
from torch import nn
import torch.nn.functional as F

from . import native_backend as _native
from . import accelerated as _accelerated
from . import v2_strict as _v2_strict


class _RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x):
        # AVQ states accumulate in FP32. The paper configuration uses the
        # default FP32 RMSNorm epsilon (torch.finfo(torch.float32).eps).
        z = x.float()
        if self.eps == torch.finfo(torch.float32).eps:
            return F.rms_norm(z, (z.shape[-1],), self.weight.float(), eps=self.eps).to(x.dtype)
        # Keep the established implementation for all legacy configurations.
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
                * self.weight.float()).to(x.dtype)


def _head_retention_rates(num_heads: int) -> list[float]:
    """Return the deterministic per-head retention-rate schedule."""
    num_heads = int(num_heads)
    if num_heads < 1:
        raise ValueError("num_heads must be positive")

    def power_of_two(count: int) -> list[float]:
        start = 2.0 ** (-2.0 ** (-(math.log2(count) - 3.0)))
        return [start * (start ** index) for index in range(count)]

    if num_heads & (num_heads - 1) == 0:
        return power_of_two(num_heads)
    closest = 2 ** int(math.floor(math.log2(num_heads)))
    return (power_of_two(closest)
            + power_of_two(2 * closest)[0::2][:num_heads - closest])


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
    """AVQ read-gated memory, defaulting to the full attention width."""
    def __init__(self, input_dim, width=None, *, init_policy="balanced", latent_dim=32,
                 device='auto', backend='auto', return_map='auto', norm_eps=None):
        super().__init__()
        if backend not in ('auto', 'native', 'pytorch'):
            raise ValueError('backend must be auto, native or pytorch')
        self.backend = backend
        self.use_native = backend != 'pytorch'
        if init_policy not in ("legacy", "balanced"):
            raise ValueError("init_policy must be legacy or balanced")
        self.init_policy = init_policy
        self.latent_dim = int(latent_dim)
        if width is None:
            width = int(input_dim)
        if width < 1:
            raise ValueError('avq_dim must be positive')
        self.width = int(width)
        self._fixed_retention_enabled = False
        self._fixed_retention_heads = None
        self.register_buffer('_fixed_retention_coefficients', None, persistent=False)
        if return_map == 'auto':
            return_map = 'identity' if self.width == int(input_dim) else 'linear'
        if return_map not in ('linear', 'identity'):
            raise ValueError("return_map must be 'auto', 'linear' or 'identity'")
        if return_map == 'identity' and int(input_dim) != self.width:
            raise ValueError('identity AVQ return requires avq_dim == num_heads * latent_dim')
        self.return_map = return_map
        if norm_eps is None:
            norm_eps = (torch.finfo(torch.float32).eps
                        if return_map == 'identity' else 1e-6)
        self.project=nn.Linear(input_dim,3*width,bias=True)
        self.norm=_RMSNorm(width, eps=norm_eps)
        self.back=(nn.Identity() if return_map == 'identity'
                   else nn.Linear(width,input_dim,bias=False))
        self.gamma=nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            nn.init.normal_(self.project.weight,0.,.01)
            nn.init.zeros_(self.project.bias)
            self.project.bias[:width].fill_(math.log(.98/.02))
            self.project.weight[2*width:].zero_()  # zero read-gate initially
            if isinstance(self.back, nn.Linear):
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

    @torch.no_grad()
    def _enable_fixed_retention(self, num_heads: int):
        """Switch this memory to fixed per-head retention and a V/R-only projection.

        The active V and R rows are copied from the existing three-branch
        projection, preserving their initialized values while removing the
        unused gate rows. This is an internal model-construction operation.
        """
        num_heads = int(num_heads)
        if num_heads < 1:
            raise ValueError('num_heads must be positive')
        if self._fixed_retention_enabled:
            if self._fixed_retention_heads != num_heads:
                raise RuntimeError('fixed retention is already configured for a different head count')
            return self

        if self.width <= 0:
            raise ValueError('AVQ width must be positive')

        old_project = self.project
        old_weight = old_project.weight.detach()
        old_bias = old_project.bias.detach() if old_project.bias is not None else None
        if old_weight.shape[0] != 3 * self.width:
            raise RuntimeError('expected the original three-branch AVQ projection')

        # Construct the replacement without advancing the caller's RNG stream;
        # all active rows are copied immediately below.
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(old_weight.device)
            if old_weight.is_cuda else None
        )
        try:
            new_project = nn.Linear(
                old_project.in_features, 2 * self.width,
                bias=old_project.bias is not None,
                device=old_weight.device, dtype=old_weight.dtype,
            )
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, old_weight.device)

        new_weight = torch.cat((old_weight[self.width:2 * self.width],
                                old_weight[2 * self.width:]), dim=0)
        new_project.weight.copy_(new_weight)
        if old_bias is not None:
            new_bias = torch.cat((old_bias[self.width:2 * self.width],
                                  old_bias[2 * self.width:]), dim=0)
            new_project.bias.copy_(new_bias)
        new_project.weight.requires_grad_(old_project.weight.requires_grad)
        if old_project.bias is not None:
            new_project.bias.requires_grad_(old_project.bias.requires_grad)
        self.project = new_project

        rates = torch.tensor(_head_retention_rates(num_heads), dtype=torch.float32)
        head_indices = torch.div(
            torch.arange(self.width, dtype=torch.long) * num_heads,
            self.width, rounding_mode='floor',
        ).clamp_(max=num_heads - 1)
        self._fixed_retention_coefficients = torch.exp(-rates[head_indices])
        self._fixed_retention_heads = num_heads
        self._fixed_retention_enabled = True
        return self

    def components_from_raw(self, raw):
        if self._fixed_retention_enabled:
            v, r = raw.split(self.width, dim=-1)
            a = self._fixed_retention_coefficients.view(1, 1, -1).expand(
                raw.shape[0], raw.shape[1], -1
            )
            return a, (0.1 * torch.tanh(v)).float(), torch.tanh(r).float()
        a, v, r = raw.split(self.width, dim=-1)
        return torch.sigmoid(a).float(), (0.1 * torch.tanh(v)).float(), torch.tanh(r).float()

    def components(self, o):
        return self.components_from_raw(self.project(o))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self._fixed_retention_enabled:
            weight_key = prefix + 'project.weight'
            bias_key = prefix + 'project.bias'
            old_rows = 3 * self.width
            new_rows = 2 * self.width
            if weight_key in state_dict and state_dict[weight_key].shape[0] == old_rows:
                old_weight = state_dict[weight_key]
                state_dict[weight_key] = torch.cat((
                    old_weight[self.width:2 * self.width],
                    old_weight[2 * self.width:]), dim=0
                )
            if bias_key in state_dict and state_dict[bias_key].shape[0] == old_rows:
                old_bias = state_dict[bias_key]
                state_dict[bias_key] = torch.cat((
                    old_bias[self.width:2 * self.width],
                    old_bias[2 * self.width:]), dim=0
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def _return_update(self, h, dtype):
        # The recurrence, normalization and read gate use FP32. Explicit
        # low-precision weights need a matching GEMM input when autocast
        # is disabled. Leave autocast and the identity return map alone.
        if isinstance(self.back, nn.Linear) and not torch.is_autocast_enabled(h.device.type):
            h = h.to(self.back.weight.dtype)
        return self.back(h).to(dtype)

    def forward(self,o,*,return_state=False):
        if o.device != self.project.weight.device:
            o = o.to(self.project.weight.device, non_blocking=True)
        if self.backend == 'native' and o.is_cuda:
            if self._fixed_retention_enabled:
                provider_available = (
                    _accelerated.fixed_available() or _native.get_extension() is not None
                )
            else:
                provider_available = (
                    _accelerated.available() or _native.get_extension() is not None
                )
            if not provider_available:
                raise RuntimeError('Native BOLT state memory requires a compiled CUDA extension')
        if self._fixed_retention_enabled:
            a, v, r = self.components(o)
            s = None
            if self.use_native and o.is_cuda:
                s = _accelerated.scan_coefficients(a, v, latent_dim=self.latent_dim)
                if s is None and _native.get_extension() is not None:
                    s = _native.avq_scan(a.contiguous(), v.contiguous())
            if s is None:
                s = _affine_scan(a, v)
        elif self.use_native and o.is_cuda and _accelerated.available():
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
        update=self._return_update(h, o.dtype)
        enhanced=o+self.gamma.to(o.dtype)*update
        return (enhanced,s[:,-1,:].contiguous()) if return_state else enhanced

    @torch.no_grad()
    def _prefill_chunk(self, o, carry):
        """Evaluate one AVQ chunk with the previous chunk's FP32 state."""
        a, v, r = self.components(o)
        a = a.float()
        # The first affine update incorporates the incoming state; all later
        # coefficients and updates are unchanged. v is a fresh tensor.
        v[:, 0, :].add_(a[:, 0, :] * carry)
        history = _affine_scan(a, v)
        h = self.norm(history) * (1. + r)
        update = self._return_update(h, o.dtype)
        enhanced = o + self.gamma.to(o.dtype) * update
        return enhanced, history[:, -1, :].clone()

    @torch.no_grad()
    def _prefill_chunked(self, o, chunk_size):
        """Bound AVQ intermediates while preserving the full recurrence.

        Only for eval-mode PyTorch cached prefill. The enhanced sequence is
        still materialized for full-context attention; this is not an
        attention window, and no tokens or state updates are dropped.
        """
        if self.training or self.backend != 'pytorch' or self.use_native:
            raise RuntimeError('Chunked AVQ requires eval-mode PyTorch memory')
        if chunk_size <= 0 or o.size(1) == 0:
            raise ValueError('Expected positive chunk size and nonempty sequence')
        if o.device != self.project.weight.device:
            o = o.to(self.project.weight.device, non_blocking=True)
        carry = torch.zeros((o.size(0), self.width), device=o.device,
                            dtype=torch.float32)
        out = torch.empty_like(o)
        for start in range(0, o.size(1), chunk_size):
            stop = min(start + chunk_size, o.size(1))
            part, carry = self._prefill_chunk(o[:, start:stop], carry)
            out[:, start:stop].copy_(part)
            del part
        return out, carry

    def step(self,o,state,*,_v2_cache_inplace=False):
        # MLBRICKS_BOLT_INFERENCE_AMP_V2
        # FP16 projection GEMMs only; FP32 V2 Strict state/residual unchanged.
        with (torch.autocast(device_type=o.device.type, dtype=torch.float16)
              if self.project.weight.dtype == torch.float16 and o.device.type in ('cuda', 'cpu')
              else nullcontext()):
            if o.device != self.project.weight.device:
                o = o.to(self.project.weight.device, non_blocking=True)
            if state.device != o.device:
                state = state.to(o.device, non_blocking=True)
            # Optional V2 strict native one-token path. Keep the original torch
            # projection; do not substitute the V3 GEMV reduction. This path is
            # inference-only and never alters checkpoint/state-dict schemas.
            candidate = _v2_strict.maybe_step(
                self, o, state, inplace_cache=_v2_cache_inplace
            )
            if candidate is not None:
                return candidate
            a,v,r=self.components(o)
            if self.use_native and o.is_cuda and _native.get_extension() is not None:
                s=_native.avq_step(a[:,0,:],v[:,0,:],state.float())
            else:
                s=a[:,0,:]*state.float()+v[:,0,:]
            h=(self.norm(s)*(1.+r[:,0,:]))[:,None,:]
            update=self._return_update(h, o.dtype)
            return o+self.gamma.to(o.dtype)*update, s


__all__ = ["BoltStateMemory"]
