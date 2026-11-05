"""No research-specific constructor or manual fusion needed by Bolt callers."""
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
    assert m._modern.p.requires_grad
    assert next(m.parameters()).device.type == 'cpu'
    assert not any(k.endswith('.avq.back.weight') for k in m.state_dict())
    assert m._modern._fold_query_inference  # automatic; no public fusion call


def test_training_keeps_original_trainable_p_and_retraining_after_eval():
    torch.manual_seed(4)
    m = new_layer().train()
    x = torch.randn(2, 4, 48)
    m(x).square().mean().backward()
    assert m._modern.p.grad is not None
    assert m._modern.p.grad.abs().sum() > 0
    before = m._modern.p.detach().clone()
    optimizer = torch.optim.SGD([m._modern.p], lr=0.1)
    optimizer.step()
    assert not torch.equal(before, m._modern.p)
    m.eval()
    with torch.inference_mode():
        _ = m(x)
    m.train()
    assert m._modern.p.requires_grad
    assert m._modern._folded_q_cache is None
    m.zero_grad(set_to_none=True)
    m(x).square().mean().backward()
    assert m._modern.p.grad is not None and m._modern.p.grad.abs().sum() > 0


@pytest.mark.parametrize('projection', ['packed', 'separate'])
def test_automatic_folded_inference_matches_unfused_and_cache_decode(projection):
    torch.manual_seed(8)
    m = Bolt(d_model=48, num_heads=12, profile='compact', projection=projection,
             backend='pytorch', device='cpu', precision='fp32').eval()
    with torch.no_grad():
        m._modern.p.uniform_(0.4, 1.6)
        m._modern.avq.gamma.fill_(0.1)
    unfused = copy.deepcopy(m)
    unfused._modern._disable_folded_query_inference()
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    x = torch.randn(1, 4, 48)
    with torch.inference_mode():
        torch.testing.assert_close(m(x), unfused(x), atol=3e-5, rtol=3e-5)
        y1, c1 = m.prefill_with_cache(x[:, :3]); y2, c2 = unfused.prefill_with_cache(x[:, :3])
        torch.testing.assert_close(y1, y2, atol=3e-5, rtol=3e-5)
        z1, c1 = m.decode_step(x[:, 3:], c1); z2, c2 = unfused.decode_step(x[:, 3:], c2)
        torch.testing.assert_close(z1, z2, atol=3e-5, rtol=3e-5)
        assert m._modern._folded_q_cache is not None
    assert list(saved) == list(m.state_dict())
    for k, old in saved.items():
        torch.testing.assert_close(m.state_dict()[k], old, rtol=0, atol=0)
    with torch.no_grad():
        m._modern.p.add_(0.01)
        _ = m(x)
        assert m._modern._folded_q_cache is not None
    # Explicitly resetting P also invalidates any folded weights by version.
    assert not any('folded' in k for k in m.state_dict())


def test_backward_compatible_explicit_linear_and_legacy_defaults():
    standard = Bolt(48, 12, device='cpu')
    assert standard.avq_dim == 198
    assert isinstance(standard._modern.avq.back, nn.Linear)
    old = Bolt(48, 12, profile='compact', avq_dim=192,
               avq_return_map='linear', avq_norm_eps=1e-6, device='cpu')
    assert isinstance(old._modern.avq.back, nn.Linear)
    assert old._modern.avq.norm.eps == 1e-6
    assert old._modern.p.requires_grad
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
    assert layer._modern._fold_query_inference


def test_explicit_config_overrides_for_old_checkpoints():
    cfg = BoltConfig(vocab_size=37, context=8, layers=1, dim=48, heads=12,
                     profile='compact', state_dim=192, attention_scale='latent',
                     memory_init='balanced', avq_return_map='linear',
                     avq_norm_eps=1e-6, device='cpu', backend='pytorch', precision='fp32')
    m = BoltModel(cfg)
    a = m.blocks[0].attn._modern
    assert a.attention_scale == 'latent' and a.memory_init == 'balanced'
    assert isinstance(a.avq.back, nn.Linear)
    assert a.avq.norm.eps == 1e-6
