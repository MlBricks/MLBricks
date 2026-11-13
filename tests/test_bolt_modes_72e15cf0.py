"""BOLT public-mode checks for integration against Git commit 72e15cf0."""
import math
import pytest
import torch
from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt.enhanced_mqa import EnhancedMQA


@pytest.mark.parametrize('mode,bytes_per_token', [(None, 128), ('quantised', 128), ('Keyless', 768), ('base', 1536)])
@pytest.mark.parametrize('avq', [None, 'disable'])
def test_modes_memory_and_gradients(mode, bytes_per_token, avq):
    torch.manual_seed(12)
    layer = Bolt(384, 12, mode=mode, avq=avq, backend='pytorch', device='cpu', position='rope')
    expected = 'quantised' if mode is None else mode.lower()
    assert layer.mode == expected
    assert layer.head_dim == 32
    assert layer.cache_saving_percent == pytest.approx((1.0 - bytes_per_token / 1536) * 100)
    assert layer.state_memory == (avq != 'disable')
    assert layer._modern.head_gate is not None
    x = torch.randn(2, 4, 384, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer._modern.head_gate.weight.grad is not None
    if avq != 'disable':
        assert layer._modern.avq.project.weight.grad is not None
        forget_bias = layer._modern.avq.project.bias[:layer._modern.avq.width]
        assert torch.sigmoid(forget_bias).mean().item() == pytest.approx(0.95, abs=.01)
    else:
        assert layer._modern.avq is None


@pytest.mark.parametrize('mode', ['quantised', 'Keyless', 'base'])
@pytest.mark.parametrize('avq', [None, 'disable'])
def test_prefill_and_incremental_decode(mode, avq):
    torch.manual_seed(4)
    model = Bolt(64, 2, mode=mode, avq=avq, position='rope', backend='pytorch', device='cpu').eval()
    x = torch.randn(1, 5, 64)
    with torch.no_grad():
        full = model(x)
        prefix, cache = model.prefill_with_cache(x[:, :4])
        step, _ = model.decode_step(x[:, 4:], cache)
    torch.testing.assert_close(full[:, :4], prefix, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(full[:, 4:], step, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('mode', ['quantised', 'Keyless', 'base'])
@pytest.mark.parametrize('avq', [None, 'disable'])
def test_full_lm_prefill_and_cache(mode, avq):
    torch.manual_seed(3)
    cfg = BoltConfig(vocab_size=70, context=20, layers=2, dim=64, heads=2,
                     bolt_mode=mode, position='rope', backend='pytorch',
                     device='cpu', avq=avq)
    m = BoltModel(cfg).eval()
    ids = torch.randint(0, 70, (1, 6))
    with torch.no_grad():
        full = m(ids)
        prefix, caches = m.prefill(ids[:, :5])
        step, _ = m.decode_step(ids[:, 5:], caches, position=5)
        m.prepare_generation(batch_size=1, max_context=18, warmup_native=False)
        out = m.generate(ids[:, :3], max_new_tokens=2, temperature=0., top_k=1)
    assert out.shape == (1, 5)
    torch.testing.assert_close(full[:, :5], prefix, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(full[:, 5:], step, atol=3e-4, rtol=3e-4)


def test_three_layer_hybrid_and_per_layer_avq_widths():
    cfg = BoltConfig(vocab_size=73, context=24, layers=3,
                     dim=96, heads=2, position='rope', backend='pytorch',
                     layer_modes=('base', 'quantised', 'Keyless'), device='cpu')
    model = BoltModel(cfg).eval()
    assert [layer.attn.mode for layer in model.blocks] == ['base','quantised','keyless']
    assert [layer.attn.avq_dim for layer in model.blocks] == [96,96,64]
    ids = torch.randint(0, 73, (1, 6))
    with torch.no_grad():
        full = model(ids)
        prefix, caches = model.prefill(ids[:, :5])
        step, _ = model.decode_step(ids[:, 5:], caches, position=5)
    torch.testing.assert_close(full[:, :5], prefix, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(full[:, 5:], step, atol=3e-4, rtol=3e-4)


def test_avq_and_head_gate_independent():
    model = Bolt(64, 2, mode='base', avq='disable', head_output_gate=False,
                 backend='pytorch', device='cpu')
    assert model._modern.avq is None
    assert model._modern.head_gate is None
    assert model.cache_saving_percent == 0


def test_legacy_bolt32_state_loads_with_neutral_head_gate():
    old = Bolt(64, 2, mode='Keyless', head_output_gate=False,
               backend='pytorch', device='cpu')
    updated = Bolt(64, 2, mode='Keyless', head_output_gate=True,
                   backend='pytorch', device='cpu')
    updated.load_state_dict(old.state_dict(), strict=True)
    with torch.no_grad():
        x = torch.randn(1, 3, 64)
        torch.testing.assert_close(old(x), updated(x), atol=1e-6, rtol=1e-6)


def test_bolt16_not_selectable():
    with pytest.raises(ValueError, match='BOLT16'):
        Bolt(384, 12, mode='Keyless', cache='tight')
    with pytest.raises(ValueError, match='BOLT16'):
        BoltConfig(vocab_size=50, profile='compact')
    with pytest.raises(ValueError, match='BOLT16'):
        Bolt(384, 12, mode='quantised', profile='compact')

@pytest.mark.parametrize('mode', ['quantised', 'base'])
def test_custom_state_width_remains_supported(mode):
    layer = Bolt(64, 2, mode=mode, avq_dim=24, device='cpu', backend='pytorch')
    assert layer.avq_dim == 24
    assert layer._modern.avq.width == 24
    assert isinstance(layer._modern.avq.back, torch.nn.Linear)
    x = torch.randn(1, 4, 64, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    assert layer._modern.avq.back.weight.grad is not None
    layer.eval()
    with torch.no_grad():
        _, cache = layer.prefill_with_cache(x.detach()[:, :3])
        out, cache = layer.decode_step(x.detach()[:, 3:], cache)
    assert out.shape == (1, 1, 64)
