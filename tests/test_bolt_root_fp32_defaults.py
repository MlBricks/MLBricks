"""The documented public BOLT API and precision policy."""
from __future__ import annotations

import inspect

import torch

from mlbricks import Bolt, BoltConfig, BoltModel


def test_root_public_api_and_fp32_constructor_defaults():
    import mlbricks
    from mlbricks.bolt import Bolt as InternalBolt
    from mlbricks.bolt import BoltConfig as InternalConfig
    from mlbricks.bolt import BoltModel as InternalModel

    assert mlbricks.Bolt is Bolt is InternalBolt
    assert mlbricks.BoltConfig is BoltConfig is InternalConfig
    assert mlbricks.BoltModel is BoltModel is InternalModel
    assert inspect.signature(Bolt).parameters['precision'].default == 'fp32'
    assert BoltConfig.__dataclass_fields__['precision'].default == 'fp32'


def test_bolt16_and_bolt32_default_to_fp32():
    for profile in ('compact', 'standard'):
        model = Bolt(embed_dim=64, num_heads=4, profile=profile,
                     device='cpu', backend='pytorch').eval()
        assert model.precision == 'fp32'
        x = torch.randn(1, 3, 64)
        with torch.inference_mode():
            y = model(x)
        assert y.dtype == torch.float32
        assert all(p.dtype == torch.float32 for p in model.parameters())


def test_bolt_config_and_lm_default_to_fp32():
    config = BoltConfig(vocab_size=37, context=16, layers=1,
                        dim=64, heads=4, profile='compact',
                        device='cpu', backend='pytorch')
    assert config.precision == 'fp32'
    model = BoltModel(config).eval()
    assert model.config.precision == 'fp32'
    tokens = torch.randint(0, 37, (1, 3))
    with torch.inference_mode():
        logits = model(tokens)
    assert logits.dtype == torch.float32


def test_explicit_mixed_precision_modes_remain_available():
    for precision in ('auto', 'fp16', 'bf16', 'fp32'):
        cfg = BoltConfig(vocab_size=31, context=8, layers=1,
                         dim=64, heads=4, precision=precision)
        assert cfg.precision == precision
        layer = Bolt(embed_dim=64, num_heads=4, profile='compact',
                     device='cpu', precision=precision)
        assert layer.precision == precision
