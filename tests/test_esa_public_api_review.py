"""Minimal direct Fast-EsaQ public API."""
import inspect
import pytest
import torch
from mlbricks import ESA, ESAModelConfig
from mlbricks.esa.fast_engine.config import ESAConfig


def test_direct_esa_contract_and_public_gate_signature():
    sig=inspect.signature(ESA).parameters
    assert sig['backend'].default=='auto'
    assert sig['device'].default is None
    assert 'gate_min' not in sig and 'gate_max' not in sig
    for cfg in (ESAConfig, ESAModelConfig):
        assert 'gate_min' not in cfg.__dataclass_fields__
        assert 'gate_max' not in cfg.__dataclass_fields__
    with pytest.raises(TypeError):
        ESA(embd=24,head=6,backend='pytorch',gate_min=0.8)


def test_current_half_lives_and_learned_decay():
    esa=ESA(embd=384,head=6,backend='pytorch',device='cpu')
    init=esa.log_half_life.detach().exp()
    expected=torch.tensor([16.,64.,256.,1024.,4096.,8192.])
    torch.testing.assert_close(init,expected,rtol=1e-6,atol=1e-6)
    assert esa.log_half_life.requires_grad and esa.raw_decay_bias.requires_grad
    y,state=esa.prefill(torch.randn(2,4,384))
    assert y.shape==(2,4,384) and state.shape==(2,6,64)
