"""BOLT owns its implementation without the obsolete Gaussian module."""
from pathlib import Path
import torch
import mlbricks
from mlbricks import BoltConfig, BoltModel
from mlbricks.bolt.model import BoltModelCore
from mlbricks.bolt.config import BoltModelSettings
from mlbricks.cache import BoltLegacyCache

def test_model_is_native_bolt_without_old_module():
    assert issubclass(BoltModel, BoltModelCore)
    assert BoltModel.__module__ == 'mlbricks.bolt.model'
    assert BoltConfig.__module__ == 'mlbricks.bolt.config'
    assert not hasattr(mlbricks, 'Gaussian')
    assert not hasattr(mlbricks, 'GaussianConfig')
    assert not hasattr(mlbricks.cache, 'GaussCache')
    assert not (Path(mlbricks.__file__).parent / 'gaussian.py').exists()

def test_matching_historical_bolt_math_is_available():
    c = BoltConfig(vocab_size=37, context=12, layers=1, dim=32, heads=4, backend='pytorch',
                   device='cpu', attention_scale='head', memory_init='legacy', projection='separate')
    m = BoltModel(c)
    assert m.blocks[0].attn._modern.attention_scale == 'head'
    assert m.blocks[0].attn._modern.projection == 'separate'
    assert m.blocks[0].attn._modern.avq.gamma.item() == 0.0
    assert m(torch.randint(0,37,(1,5))).shape == (1,5,37)

def test_avq_toggle_and_checkpoint_keys():
    with_memory = BoltModel(BoltConfig(vocab_size=37,context=12,layers=1,dim=32,heads=4, device='cpu',backend='pytorch',precision='fp32'))
    without_memory = BoltModel(BoltConfig(vocab_size=37,context=12,layers=1,dim=32,heads=4, device='cpu',backend='pytorch',precision='fp32',state_memory=False))
    assert with_memory.blocks[0].attn._modern.avq is not None
    assert without_memory.blocks[0].attn._modern.avq is None
    assert len(without_memory.state_dict()) < len(with_memory.state_dict())
    assert any(k.startswith('blocks.0.attn') for k in with_memory.state_dict())

def test_layer_precision_is_inherited_from_config():
    for policy in ('auto','fp16','fp32'):
        cfg = BoltConfig(vocab_size=37,context=12,layers=1,dim=32,heads=4,device='cpu',backend='pytorch',precision=policy)
        m = BoltModel(cfg)
        assert m.blocks[0].attn.precision == policy
