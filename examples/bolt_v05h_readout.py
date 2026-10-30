"""Opt-in experiment: BOLT v0.5h AVQ readout candidate with strict reference ordering.

State is updated with validated v0.5g native state-only kernel. For mode='pt_gate',
tanh and inverse RMS are computed with PyTorch; only pointwise FP32 readout is fused.
For mode='raw_gate', CUDA computes tanh then rounds to FP16/BF16 before FP32 readout.
This is NOT a production API. No parameter or checkpoint changes.
"""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
from mlbricks.bolt import native_v05


def readout_step(self, o, state, *, inplace=False):
    if not inplace or self.training or torch.is_grad_enabled():
        return self._v05h_original_step(o, state, inplace=inplace)
    if not self.fold_output_inference or not o.is_cuda:
        raise RuntimeError('v0.5h needs optimized CUDA inference')
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise TypeError('AVQ recurrent state must be contiguous float32')
    raw = self.project(o).contiguous()
    native_v05.avq_state_only(raw, state)
    sf = state[:, None, :]
    # IMPORTANT: preserve the exact PyTorch arithmetic for RMS reduction.
    inv = torch.rsqrt(sf.square().mean(-1, keepdim=True) + self.norm.eps)
    shape = (o.shape[0], 1, self.width)
    if (self._v05h_out is None or self._v05h_out.shape != shape
            or self._v05h_out.device != o.device):
        self._v05h_out = torch.empty(shape, device=o.device, dtype=torch.float32)
    if self._v05h_mode == 'pt_gate':
        r = torch.tanh(raw[..., 2*self.width:]).float().contiguous()
        native_v05.avq_readout_pt(state, r, inv, self._v05h_out)
    elif self._v05h_mode == 'raw_gate':
        native_v05.avq_readout_raw(state, raw, inv, self._v05h_out)
    else:
        raise ValueError('Unknown v0.5h mode')
    result = o + F.linear(self._v05h_out.to(o.dtype), self._inference_back()).to(o.dtype)
    return result, state


def enable_readout(model, *, mode='pt_gate'):
    if mode not in ('pt_gate','raw_gate'):
        raise ValueError('mode must be pt_gate or raw_gate')
    for block in model.blocks:
        avq = block.attn._modern.avq
        if not avq.v05_native_avq:
            raise RuntimeError('Enable native AVQ before v0.5h readout')
        if hasattr(avq, '_v05h_original_step'):
            raise RuntimeError('Already enabled v0.5h')
        avq._v05h_original_step = avq.step
        avq._v05h_out = None  # created during graph warmup, reused during replay
        avq._v05h_mode = mode
        avq.step = types.MethodType(readout_step, avq)
    return model
