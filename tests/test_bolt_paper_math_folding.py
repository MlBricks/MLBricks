"""BOLT paper math and inference-only P folding regression tests (CPU-safe)."""
import copy
import torch
import pytest
from torch import nn
import torch.nn.functional as F

from mlbricks.bolt.presets import BoltPreset


def compact_192_preset():
    # Internal preset used only to exercise the low-level folding implementation.
    return BoltPreset(384, 12, mode='quality', cache='tight', avq_dim=192,
                      projection='packed', attention_scale='head',
                      memory_init='legacy', position=None,
                      avq_return_map='identity',
                      avq_norm_eps=torch.finfo(torch.float32).eps,
                      backend='pytorch')


@pytest.mark.parametrize('projection', ['packed', 'separate'])
@pytest.mark.parametrize('bias', [False, True])
def test_folded_q_forward_and_decode_preserve_weights(projection, bias):
    torch.manual_seed(1234)
    m = BoltPreset(
        384, 12, mode='quality', cache='tight', avq_dim=192,
        avq_return_map='identity', avq_norm_eps=torch.finfo(torch.float32).eps,
        projection=projection, bias=bias, backend='pytorch',
    )
    m.eval()
    with torch.no_grad():
        m.p.uniform_(0.5, 1.5)
        m.avq.gamma.fill_(0.2)
    ref = copy.deepcopy(m)
    old_weights = {name: tensor.clone() for name, tensor in m.state_dict().items()}
    m._enable_folded_query_inference()
    assert m.p.requires_grad
    assert isinstance(m.avq.back, nn.Identity)
    assert m.avq.norm.eps == torch.finfo(torch.float32).eps
    inp = torch.randn(2, 6, 384)
    with torch.no_grad():
        torch.testing.assert_close(m(inp), ref(inp), atol=2e-5, rtol=2e-5)
        a, ca = m.prefill_with_cache(inp[:, :4])
        b, cb = ref.prefill_with_cache(inp[:, :4])
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(ca[0],cb[0],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(ca[1],cb[1],atol=2e-5,rtol=2e-5)
        a,ca=m.decode_step(inp[:,4:5],ca)
        b,cb=ref.decode_step(inp[:,4:5],cb)
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(ca[1],cb[1],atol=2e-5,rtol=2e-5)
    for name, w in m.state_dict().items():
        torch.testing.assert_close(w, old_weights[name], rtol=0, atol=0)
    assert not any('_folded' in name for name in m.state_dict())
    m._disable_folded_query_inference()
    with torch.no_grad():
        torch.testing.assert_close(m(inp), ref(inp), rtol=0, atol=0)


def test_p_gradient_and_cache_refresh_after_retraining():
    torch.manual_seed(42)
    m = compact_192_preset()
    assert m.p.requires_grad
    assert m.avq.width == 192 and isinstance(m.avq.back, nn.Identity)
    m._enable_folded_query_inference().eval()
    inp = torch.randn(1,5,384)
    with torch.no_grad():
        m.p.uniform_(0.4,1.6)
        before=m(inp)
        m.p.add_(.01)
        after=m(inp)
    assert not torch.equal(before,after)  # Cache rebuild detects P._version
    m.train()
    inp.requires_grad_(True)
    m(inp).square().mean().backward()
    assert m.p.grad is not None and bool(m.p.grad.abs().sum() > 0)
    assert m._folded_q_cache is None
    m.eval()
    with torch.no_grad():
        expected = copy.deepcopy(m)._disable_folded_query_inference()(inp.detach())
        torch.testing.assert_close(m(inp.detach()), expected, atol=2e-5, rtol=2e-5)


def test_norm_and_no_avq_back_weights():
    m=compact_192_preset()
    assert not any(k.endswith('.back.weight') for k in m.state_dict())
    x=torch.randn(3,4,192)
    torch.testing.assert_close(
        m.avq.norm(x), F.rms_norm(x, (192,), m.avq.norm.weight), rtol=0, atol=0
    )
    assert m.avq.norm.eps == torch.finfo(torch.float32).eps
    with pytest.raises(ValueError):
        BoltPreset(384, 6, mode='quality', cache='tight', avq_dim=192, avq_return_map='identity')
    with pytest.raises(ValueError):
        BoltPreset(384, 12, mode='quality', cache='tight', avq_dim=198, avq_return_map='identity')


def test_legacy_defaults_unchanged():
    m=BoltPreset(384,12,backend='pytorch')
    assert m.avq_dim==198
    assert isinstance(m.avq.back,nn.Linear)
    assert m.avq.norm.eps==1e-6
