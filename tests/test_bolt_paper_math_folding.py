"""Paper AVQ mathematics and direct-Q training parity (CPU-safe)."""
import copy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mlbricks.bolt.presets import BoltPreset


def compact_192_preset():
    return BoltPreset(384, 12, mode='quality', cache='tight', avq_dim=192,
                      projection='packed', attention_scale='head',
                      memory_init='legacy', position=None,
                      avq_return_map='identity',
                      avq_norm_eps=torch.finfo(torch.float32).eps,
                      backend='pytorch')


@pytest.mark.parametrize('projection', ['packed', 'separate'])
@pytest.mark.parametrize('bias', [False, True])
def test_direct_q_forward_decode_preserves_weights(projection, bias):
    torch.manual_seed(1234)
    m = BoltPreset(384, 12, mode='quality', cache='tight', avq_dim=192,
                   avq_return_map='identity', avq_norm_eps=torch.finfo(torch.float32).eps,
                   projection=projection, bias=bias, backend='pytorch').eval()
    with torch.no_grad():
        m.avq.gamma.fill_(0.2)
    ref = copy.deepcopy(m)
    before = {name: value.clone() for name, value in m.state_dict().items()}
    assert not hasattr(m, 'p')
    assert isinstance(m.avq.back, nn.Identity)
    inp = torch.randn(2, 6, 384)
    with torch.no_grad():
        torch.testing.assert_close(m(inp), ref(inp), atol=2e-5, rtol=2e-5)
        a, ca = m.prefill_with_cache(inp[:, :4])
        b, cb = ref.prefill_with_cache(inp[:, :4])
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        a, ca = m.decode_step(inp[:, 4:5], ca)
        b, cb = ref.decode_step(inp[:, 4:5], cb)
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_q_weight_gradient_and_retraining_after_eval():
    torch.manual_seed(42)
    m = compact_192_preset()
    assert not hasattr(m, 'p')
    assert m.avq.width == 192 and isinstance(m.avq.back, nn.Identity)
    inp = torch.randn(1, 5, 384)
    with torch.no_grad():
        m.avq.gamma.fill_(.2)
    m.train()
    m(inp).square().mean().backward()
    assert m.packed_qug_weight.grad is not None
    assert m.packed_qug_weight.grad[:192].abs().sum() > 0
    m.eval()
    with torch.no_grad():
        actual = m(inp)
        expected = copy.deepcopy(m)(inp)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_norm_and_no_avq_return_weights():
    m = compact_192_preset()
    assert not any(k.endswith('.back.weight') for k in m.state_dict())
    x = torch.randn(3, 4, 192)
    torch.testing.assert_close(m.avq.norm(x), F.rms_norm(x, (192,), m.avq.norm.weight),
                               rtol=0, atol=0)
    assert m.avq.norm.eps == torch.finfo(torch.float32).eps
    with pytest.raises(ValueError):
        BoltPreset(384, 6, mode='quality', cache='tight', avq_dim=192,
                   avq_return_map='identity')
    with pytest.raises(ValueError):
        BoltPreset(384, 12, mode='quality', cache='tight', avq_dim=198,
                   avq_return_map='identity')


def test_internal_preset_default_config_stays_configurable():
    m = BoltPreset(384, 12, backend='pytorch')
    assert m.avq_dim == 198
    assert isinstance(m.avq.back, nn.Linear)
    assert m.avq.norm.eps == 1e-6
