"""Modern BOLT exposes direct-Q mathematics with no manual inference fusion."""
import copy

import pytest
import torch
from torch import nn

from mlbricks import Bolt, BoltConfig, BoltModel


def new_layer(*, backend='pytorch', device='cpu'):
    return Bolt(d_model=48, num_heads=12, profile='compact', backend=backend,
                device=device, precision='fp32')


def test_public_constructor_chooses_compact_math_and_identity():
    m = new_layer()
    assert m.is_modern and m.avq_dim == 192
    assert isinstance(m._modern.avq.back, nn.Identity)
    assert m._modern.avq.norm.eps == torch.finfo(torch.float32).eps
    assert m._modern.attention_scale == 'head'
    assert m._modern.memory_init == 'legacy'
    assert not hasattr(m._modern, 'p')
    assert not any(k.endswith('.avq.back.weight') for k in m.state_dict())
    assert not any(k.endswith('.p') for k in m.state_dict())
    assert next(m.parameters()).device.type == 'cpu'


def test_training_updates_q_and_retraining_after_eval():
    torch.manual_seed(4)
    m = new_layer().train()
    x = torch.randn(2, 4, 48)
    q_weight = m._modern.packed_qug_weight
    before = q_weight.detach().clone()
    opt = torch.optim.SGD(m.parameters(), lr=.01)
    m(x).square().mean().backward()
    assert q_weight.grad is not None and q_weight.grad[:192].abs().sum() > 0
    opt.step()
    assert not torch.equal(before, q_weight.detach())
    m.eval()
    with torch.inference_mode():
        _ = m(x)
    m.train()
    m.zero_grad(set_to_none=True)
    m(x).square().mean().backward()
    assert q_weight.grad is not None and q_weight.grad[:192].abs().sum() > 0


@pytest.mark.parametrize('projection', ['packed', 'separate'])
def test_inference_matches_train_and_cached_decode(projection):
    torch.manual_seed(8)
    m = Bolt(d_model=48, num_heads=12, profile='compact', projection=projection,
             backend='pytorch', device='cpu', precision='fp32')
    with torch.no_grad():
        m._modern.avq.gamma.fill_(.1)
    checkpoint = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randn(1, 4, 48)
    m.train()
    with torch.no_grad():
        expected = m(x)
    m.eval()
    with torch.inference_mode():
        torch.testing.assert_close(m(x), expected, atol=3e-5, rtol=3e-5)
        y, cache = m.prefill_with_cache(x[:, :3])
        z, cache = m.decode_step(x[:, 3:], cache)
        torch.testing.assert_close(y, expected[:, :3], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(z, expected[:, 3:], atol=3e-5, rtol=3e-5)
    for name, value in checkpoint.items():
        torch.testing.assert_close(m.state_dict()[name], value, rtol=0, atol=0)


def test_explicit_custom_configuration_still_supported():
    standard = Bolt(48, 12, device='cpu')
    assert standard.avq_dim == 198
    assert isinstance(standard._modern.avq.back, nn.Linear)
    custom = Bolt(48, 12, profile='compact', avq_dim=192,
                  avq_return_map='linear', avq_norm_eps=1e-6, device='cpu')
    assert isinstance(custom._modern.avq.back, nn.Linear)
    assert custom._modern.avq.norm.eps == 1e-6
    assert not hasattr(custom._modern, 'p')
    off = Bolt(48, 12, profile='compact', state_memory=False, device='cpu')
    assert off._modern.avq is None


def test_model_compact_config_uses_192_without_projection():
    cfg = BoltConfig(vocab_size=37, context=8, layers=1, dim=48, heads=12,
                     profile='compact', device='cpu', backend='pytorch', precision='fp32')
    model = BoltModel(cfg)
    layer = model.blocks[0].attn
    assert cfg.avq_dim == 192
    assert cfg.attention_scale == 'head' and cfg.memory_init == 'legacy'
    assert isinstance(layer._modern.avq.back, nn.Identity)
    assert layer._modern.avq.norm.eps == torch.finfo(torch.float32).eps
    assert not hasattr(layer._modern, 'p')


def test_explicit_attention_and_avq_settings_still_supported():
    cfg = BoltConfig(vocab_size=37, context=8, layers=1, dim=48, heads=12,
                     profile='compact', state_dim=192, attention_scale='latent',
                     memory_init='balanced', avq_return_map='linear',
                     avq_norm_eps=1e-6, device='cpu', backend='pytorch', precision='fp32')
    a = BoltModel(cfg).blocks[0].attn._modern
    assert a.attention_scale == 'latent' and a.memory_init == 'balanced'
    assert isinstance(a.avq.back, nn.Linear)
    assert a.avq.norm.eps == 1e-6


def test_no_query_scale_option_in_professional_public_api():
    with pytest.raises(TypeError):
        Bolt(48, 12, profile='compact', query_scale='learned')
