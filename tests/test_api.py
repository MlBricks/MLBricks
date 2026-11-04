import inspect
import pytest
import torch
from mlbricks import ESA


def test_public_esa_defaults_and_only_canonical_names():
    sig=inspect.signature(ESA).parameters
    assert sig['backend'].default=='auto' and sig['device'].default is None
    assert 'compass' not in sig and 'gate_min' not in sig


def test_native_esa_forward_cpu():
    layer=ESA(embd=32,head=4,backend='auto',device='cpu')
    x=torch.randn(2,16,32)
    y,state=layer(x)
    assert y.shape==x.shape and state.shape==(2,4,8)
    assert layer.resolved_backend()=='auto'


def test_native_scan_tuning_parameters():
    layer=ESA(embd=32,head=4,cumpass=16)
    assert layer.cumpass==16


def test_canonical_backend_values_only():
    assert ESA(embd=32,head=4,backend='pytorch').resolved_backend()=='pytorch'
    assert ESA(embd=32,head=4,backend='native').resolved_backend()=='native'
    for invalid in ('thunder','pulse','flair'):
        with pytest.raises(ValueError):
            ESA(embd=32,head=4,backend=invalid)


def test_old_dimension_names_rejected():
    with pytest.raises(TypeError):
        ESA(n_embd=32,n_head=4)
