"""Architecture and parity tests for optimised-bolt (CPU-safe)."""
import pytest
import torch

from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt.shared_c import SharedCCache


@pytest.mark.parametrize('heads,expected', [(12,32),(16,24),(32,12)])
def test_one_shared_c_and_dynamic_width(heads, expected):
    b = Bolt(384, heads, mode='Keyless', device='cpu', backend='pytorch')
    assert b._modern.u_proj.out_features == expected
    assert b._modern.g_proj.out_features == expected
    assert b._modern.cache_bytes_per_token == expected * 2
    assert b._modern.avq.project.out_features == 2 * 384
    assert b.retention == 'decay'
    cache = b._modern.allocate_cache(batch_size=2, capacity=9)
    assert cache.c.shape == (2, 1, 9, expected)
    assert cache.state.shape == (2, 384)
    assert cache.state.dtype == torch.float32


@pytest.mark.parametrize('mode,kv', [('quantised',1),('quantised',3),('quantised',12),('base',None),('Keyless',None)])
@pytest.mark.parametrize('retention', ['decay','learned'])
def test_training_gradients_and_decode_parity(mode, kv, retention):
    torch.manual_seed(123)
    layer = Bolt(96, 12, mode=mode, kv_heads=kv, avq=True,
                 retention=retention, backend='pytorch', device='cpu')
    x = torch.randn(2, 5, 96, requires_grad=True)
    layer.train()
    y = layer(x)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer._modern.avq.project.weight.grad is not None
    assert layer._modern.head_gate.weight.grad is not None
    assert layer._modern.avq.project.out_features == (2 if retention == 'decay' else 3) * 96
    if mode == 'Keyless':
        assert layer._modern.avq is not None
        assert layer._modern.u_proj.weight.grad is not None
        assert layer._modern.g_proj.weight.grad is not None
    layer.eval()
    with torch.no_grad():
        full = layer(x.detach())
        first, cache = layer.prefill_with_cache(x.detach()[:, :2])
        outputs = [first]
        for t in range(2, 5):
            last, cache = layer.decode_step(x.detach()[:, t:t+1], cache)
            outputs.append(last)
        torch.testing.assert_close(full, torch.cat(outputs, dim=1), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('mode', ['quantised','Keyless','base'])
def test_disable_avq(mode):
    model = Bolt(96, 12, mode=mode, avq=False, device='cpu',backend='pytorch')
    assert model._modern.avq is None
    assert model.state_memory is False


@pytest.mark.parametrize('mode', ['Keyless','quantised'])
def test_full_model_prefill_and_preallocated_decode(mode):
    torch.manual_seed(9)
    config = BoltConfig(vocab_size=71, context=32, layers=2, dim=96, heads=12,
                        bolt_mode=mode, backend='pytorch', device='cpu')
    model = BoltModel(config).eval()
    token_ids = torch.randint(71, (2, 6))
    with torch.no_grad():
        full_logits = model(token_ids)
        logits, caches = model.prefill(token_ids[:, :3])
        for i in range(3, 6):
            logits, caches = model.decode_step(token_ids[:, i:i+1], caches, position=i)
        torch.testing.assert_close(logits[:, -1], full_logits[:, -1], atol=5e-5, rtol=5e-5)
        model.prepare_generation(batch_size=2, max_context=8, warmup_native=False)
        bufs = model._prepared_generation['caches']
        if mode == 'Keyless':
            assert all(isinstance(x,SharedCCache) for x in bufs)
            assert all(x.c.shape[1] == 1 for x in bufs)
        logits, caches = model.prefill(token_ids[:, :3], cache_buffers=bufs)
        for i in range(3, 6):
            logits, caches = model.decode_step(token_ids[:, i:i+1], caches, position=i)
        torch.testing.assert_close(logits[:, -1], full_logits[:, -1], atol=5e-5, rtol=5e-5)


class ExternalTokenTransform:
    def __call__(self, x, *, start_pos):
        p = torch.arange(start_pos, start_pos + x.size(2), device=x.device, dtype=x.dtype)
        return x + p.view(1, 1, -1, 1) * 0.0001


@pytest.mark.parametrize('mode', ['Keyless', 'quantised'])
def test_external_transform_no_builtin_position(mode):
    torch.manual_seed(1)
    layer = Bolt(96, 12, mode=mode, backend='pytorch', device='cpu').eval()
    with pytest.raises(ValueError):
        layer.set_qk_transform('rope')
    layer.set_qk_transform(ExternalTokenTransform())
    x = torch.randn(1, 4, 96)
    with torch.no_grad():
        expected = layer(x)
        prefix, cache = layer.prefill_with_cache(x[:, :2])
        a, cache = layer.decode_step(x[:, 2:3], cache)
        b, cache = layer.decode_step(x[:, 3:4], cache)
        torch.testing.assert_close(expected, torch.cat([prefix, a, b], dim=1), atol=1e-5, rtol=1e-5)


def test_invalid_gqa_and_keyless_kvheads():
    with pytest.raises(ValueError, match='kv_heads'):
        Bolt(96, 12, mode='quantised', kv_heads=5, device='cpu')
    with pytest.raises(ValueError, match='kv_heads'):
        Bolt(96, 12, mode='Keyless', kv_heads=1, device='cpu')
    with pytest.raises(ValueError):
        Bolt(96, 12, position='rope', device='cpu')
