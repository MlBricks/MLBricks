"""Requirements for the reviewed Fast-EsaQ-only public API."""
import inspect
import math

import pytest
import torch

from mlbricks import ESA, ESAConfig, ESAModelConfig
from mlbricks.vesa.config import ESAConfig as VESAESAConfig


def test_auto_backend_cuda_device_and_clean_public_gate_signature():
    sig = inspect.signature(ESA).parameters
    assert sig["backend"].default == "auto"
    assert sig["device"].default == "cuda"
    assert "gate_min" not in sig
    assert "gate_max" not in sig
    assert "cumpass" in sig and "parallel_chunk" in sig and "cuda_warps" in sig
    assert "gate_min" not in ESAConfig.__dataclass_fields__
    assert "gate_max" not in ESAConfig.__dataclass_fields__
    assert "gate_min" not in ESAModelConfig.__dataclass_fields__
    assert "gate_max" not in ESAModelConfig.__dataclass_fields__
    assert "gate_min" not in VESAESAConfig.__dataclass_fields__
    assert "gate_max" not in VESAESAConfig.__dataclass_fields__


def test_no_ignored_bounded_gate_kwargs():
    with pytest.raises(TypeError):
        ESA(embd=24, head=6, backend="pytorch", device="cpu", gate_min=0.8)


def test_current_esa_half_lives_and_learned_decay():
    esa = ESA(embd=384, head=6, backend="pytorch", device="cpu")
    engine = esa.layer
    init = engine.log_half_life.detach().exp()
    expected = torch.tensor([16., 64., 256., 1024., 4096., 8192.])
    assert torch.allclose(init, expected, rtol=1e-6, atol=1e-6)
    assert engine.log_half_life.requires_grad
    assert engine.raw_decay_bias.requires_grad
    x = torch.randn(2, 4, 384)
    y, state = esa.prefill(x)
    assert y.shape == x.shape
    assert state.shape == (2, 6, 64)
    assert state.dtype == torch.float32
