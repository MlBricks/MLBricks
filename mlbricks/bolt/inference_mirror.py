"""Internal, version-guarded FP16 inference mirrors for CUDA-first models.

The original model remains FP32 for training and checkpoint serialization.
An inference mirror is an unregistered module cached outside nn.Module._modules,
so state_dict()/parameters()/optimizers never see its weights.

The mirror's reference is invalidated by parameter/buffer version or storage
changes, configuration changes and explicit model.train(). Like PyTorch
version counters, direct writes through `.data` cannot be tracked reliably.
"""
from __future__ import annotations

import torch


def _tensor_signature(module):
    return (
        tuple((name, id(p), p.data_ptr(), p._version, p.device, p.dtype,
               tuple(p.shape)) for name, p in module.named_parameters()),
        tuple((name, id(b), b.data_ptr(), b._version, b.device, b.dtype,
               tuple(b.shape)) for name, b in module.named_buffers()),
        repr(getattr(module, 'config', None)),
        getattr(module, 'backend', None),
    )


def fp16_inference_mirror(master, factory):
    """Get a current FP16 inference mirror; preserve master weights and RNG.

    Never register the mirror as a submodule: ordinary setattr would insert it
    into _modules, corrupting state_dict and training optimizer traversal.
    """
    p = next(master.parameters())
    if p.device.type != 'cuda' or p.dtype != torch.float32:
        raise ValueError('FP16 mirror requires a CUDA FP32 master')
    stamp = _tensor_signature(master)
    cached = master.__dict__.get('_mlbricks_fp16_inference_mirror')
    if cached is not None and cached[0] == stamp:
        return cached[1], True
    master.__dict__.pop('_mlbricks_fp16_inference_mirror', None)

    cuda_index = p.device.index if p.device.index is not None else torch.cuda.current_device()
    # Model constructors initialize parameters. A cache miss must not change
    # application RNG streams (critical for reproducible training/sampling).
    with torch.no_grad(), torch.random.fork_rng(devices=[cuda_index]):
        mirror = factory()
        mirror.to(device=p.device, dtype=torch.float16)
        mirror.load_state_dict(master.state_dict(), strict=True)
        mirror.eval()
    master.__dict__['_mlbricks_fp16_inference_mirror'] = (stamp, mirror)
    return mirror, False


def drop_fp16_inference_mirror(master):
    master.__dict__.pop('_mlbricks_fp16_inference_mirror', None)
