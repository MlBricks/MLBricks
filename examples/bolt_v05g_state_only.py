"""Experimental v0.5g recurrence-only CUDA AVQ with reference PyTorch readout.

This is a research candidate. Not a library API, checkpoint format, or release.
"""
from __future__ import annotations
import types
import torch
import torch.nn.functional as F
from mlbricks.bolt import native_v05


def state_only_avq_step(self, o, state, *, inplace=False):
    if not inplace or self.training or torch.is_grad_enabled():
        return self._v05g_original_step(o,state,inplace=inplace)
    if not self.fold_output_inference:
        raise RuntimeError('v0.5g only supports optimized inference')
    if not o.is_cuda or not state.is_contiguous() or state.dtype != torch.float32:
        raise RuntimeError('v0.5g requires CUDA and contiguous FP32 AVQ state')
    raw=self.project(o).contiguous()
    native_v05.avq_state_only(raw,state)
    r=torch.tanh(raw[...,2*self.width:]).float()
    sf=state[:,None,:]
    features=(sf * torch.rsqrt(sf.square().mean(-1,keepdim=True) + self.norm.eps)) * (1. + r)
    out=o + F.linear(features.to(o.dtype),self._inference_back()).to(o.dtype)
    return out,state


def enable_state_only(model):
    for block in model.blocks:
        avq=block.attn._modern.avq
        if not avq.v05_native_avq: raise RuntimeError('Enable native AVQ first')
        avq._v05g_original_step=avq.step
        avq.step=types.MethodType(state_only_avq_step,avq)
    return model
