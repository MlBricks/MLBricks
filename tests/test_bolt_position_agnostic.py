"""BOLT owns no positional algorithm; model users can supply Q/K transforms."""

import pytest
import torch
from torch import nn
from mlbricks import Bolt, BoltModel, Attention
from bolt_external_transform import ExternalQK


@pytest.mark.parametrize('mode', ['quantised', 'Keyless', 'base'])
def test_external_transform_prefill_decode_matches_full(mode):
    torch.manual_seed(9)
    layer = Bolt(128, 16, mode=mode, device='cpu', backend='pytorch',
                 position=ExternalQK()).eval()
    if mode == 'quantised':
        assert layer._modern.num_heads == 16
        assert layer._modern.kv_heads == 1
    x = torch.randn(1, 7, 128)
    with torch.no_grad():
        full = layer(x)
        prefix, cache = layer.prefill_with_cache(x[:, :5])
        step, _ = layer.decode_step(x[:, 5:6], cache)
    torch.testing.assert_close(prefix, full[:, :5], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(step, full[:, 5:6], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('cls', [Bolt, Attention])
def test_builtin_position_strings_are_explicitly_rejected(cls):
    with pytest.raises(ValueError, match='external Q/K transform'):
        cls(32, 4, position='rope')
    with pytest.raises(ValueError, match='external Q/K transform'):
        cls(32, 4, position='alibi')
    assert cls(32, 4, position='none').position is None


def test_model_owns_transform_and_no_rope_is_instantiated():
    model = BoltModel(vocab_size=45, context=16, layers=2, dim=64, heads=4,
                      position='none', backend='pytorch', device='cpu').eval()
    assert all(block.attn.position is None for block in model.blocks)
    external = ExternalQK()
    model.set_qk_transform(external)
    assert all(block.attn.position is external for block in model.blocks)
    seq = torch.randint(0, 45, (1, 7))
    with torch.no_grad():
        full = model(seq)
        pref, caches = model.prefill(seq[:, :6])
        dec, _ = model.decode_step(seq[:, 6:], caches, position=6)
    torch.testing.assert_close(pref, full[:, :6], atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(dec, full[:, 6:7], atol=3e-4, rtol=3e-4)


def test_model_rejects_old_rope_config_without_silent_change():
    with pytest.raises(ValueError, match='no longer constructs RoPE'):
        BoltModel(vocab_size=32, context=16, layers=1, dim=32, heads=4,
                  position='rope', device='cpu')
