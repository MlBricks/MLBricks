"""CPU contracts for BOLT default automatic backend and graph preference.

These tests do NOT claim NVIDIA graph capture is implemented for arbitrary layers.
"""
import copy
import pytest
import torch
from torch import nn
from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt.graph_policy import normalize_graph_policy
from mlbricks.bolt.presets import BoltPreset


def test_default_single_bolt():
    layer = Bolt(64, 4, device='cpu').eval()
    assert layer.backend == 'auto'
    assert layer.graph == 'auto'
    assert layer._modern.graph == 'auto'
    assert layer.graph_allowed()
    output = layer(torch.randn(1, 3, 64))
    assert output.shape == (1, 3, 64)


def test_explicit_reference_overrides_graph_auto():
    layer = Bolt(64, 4, backend='pytorch', device='cpu').eval()
    assert layer.graph == 'auto'
    assert not layer.graph_allowed()
    assert layer._modern.backend == 'pytorch'
    layer.set_backend('auto')
    assert layer.graph_allowed()
    layer.set_graph(False)
    assert layer.graph == layer._modern.graph == 'off'
    assert not layer.graph_allowed()


@pytest.mark.parametrize('value, expected', [('auto','auto'),('ON','on'),('off','off'),(True,'on'),(False,'off')])
def test_graph_values(value,expected):
    assert normalize_graph_policy(value)==expected


@pytest.mark.parametrize('value', ['cuda_graph', 'yes', None, 3])
def test_graph_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        Bolt(64,4,graph=value,device='cpu')


def test_training_cannot_capture():
    layer=Bolt(64,4,device='cpu')
    assert not layer.graph_allowed()
    layer.eval()
    assert layer.graph_allowed()


def test_direct_preset_default_and_explicit_pytorch():
    p=BoltPreset(64,4).eval()
    assert p.graph == p.backend == 'auto'
    p.set_backend('pytorch')
    assert not p.graph_allowed()


def test_hybrid_with_one_bolt_layer():
    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre=nn.Linear(64,64)
            self.bolt=Bolt(64,4,device='cpu')
            self.post=nn.Linear(64,64)
        def forward(self,x):
            return self.post(self.bolt(self.pre(x)))
    h=Hybrid().eval()
    with torch.no_grad():
        y=h(torch.randn(2, 5, 64))
    assert y.shape==(2,5,64)
    assert h.bolt.backend == h.bolt.graph == 'auto'


def test_model_configuration_propagates_to_layers():
    cfg = BoltConfig(vocab_size=128,context=32,layers=2,dim=64,heads=4)
    assert cfg.backend == cfg.graph == 'auto'
    model = BoltModel(config=cfg).eval()
    assert model.graph == 'auto'
    assert all(block.attn.graph=='auto' for block in model.blocks)
    before = set(model.state_dict().keys())
    model.set_graph(False)
    assert model.config.graph == 'off'
    assert all(block.attn.graph=='off' for block in model.blocks)
    assert set(model.state_dict().keys()) == before
    model.set_backend('pytorch')
    assert all(block.attn.backend=='pytorch' for block in model.blocks)
    assert all(not block.attn.graph_allowed() for block in model.blocks)
