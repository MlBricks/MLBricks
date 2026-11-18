"""Regression checks for compact AVQ state and shared-KV inference."""
import pytest
import torch
import torch.nn.functional as F

from mlbricks.bolt.enhanced_mqa import EnhancedMQA


@pytest.mark.parametrize('heads', [48, 64])
@pytest.mark.parametrize('retention', ['learned', 'decay'])
@pytest.mark.parametrize('batch', [1, 4])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_compact_state_dynamic_static(heads, retention, batch, device):
    torch.manual_seed(17)
    # Small width keeps the regression inexpensive; full 3072D speed is
    # measured by the separate research benchmark.
    width = heads * 8
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = EnhancedMQA(width, heads, kv_heads=1, retention=retention,
                        backend='pytorch').to(device=device, dtype=dtype).eval()
    x = torch.randn(batch, 13, width, device=device, dtype=dtype)
    with torch.no_grad():
        full = model(x)
        prefix, cache = model.prefill_with_cache(x[:, :10])
        k, v, state = cache
        assert k.shape == v.shape == (batch, 1, 10, 8)
        assert state.shape == (batch, width)
        assert state.dtype == torch.float32
        assert state.storage_offset() == 0
        assert state.untyped_storage().nbytes() == state.numel() * state.element_size()
        static = model.allocate_cache(batch_size=batch, capacity=13)
        static.load_prefix(*cache)
        atol, rtol = (0.005, 0.005) if device == 'cuda' else (3e-5, 3e-5)
        torch.testing.assert_close(prefix, full[:, :10], atol=atol, rtol=rtol)
        for i in range(10, 13):
            out, cache = model.decode_step(x[:, i:i+1], cache)
            static_out = model.decode_with_cache(x[:, i:i+1], static, position=i)
            torch.testing.assert_close(out, static_out, atol=atol, rtol=rtol)
            torch.testing.assert_close(out, full[:, i:i+1], atol=atol, rtol=rtol)
            torch.testing.assert_close(cache[0], static.k[:, :, :i+1], atol=atol, rtol=rtol)
            torch.testing.assert_close(cache[1], static.v[:, :, :i+1], atol=atol, rtol=rtol)
            torch.testing.assert_close(cache[2], static.state, atol=atol, rtol=rtol)


@pytest.mark.parametrize('heads', [48, 64])
@pytest.mark.parametrize('kv_heads', [1, 4])
@pytest.mark.parametrize('training,grad,causal,qlen', [
    (False, False, False, 1), (True, True, False, 1),
    (False, True, False, 1), (False, False, True, 1),
    (False, False, True, 5), (False, False, False, 5),
])
def test_attention_matches_explicit_repeat(heads, kv_heads, training, grad, causal, qlen):
    torch.manual_seed(23)
    d = 8
    model = EnhancedMQA(heads*d, heads, kv_heads=kv_heads, state_memory=False,
                        dropout=0.).train(training)
    q = torch.randn(2, heads, qlen, d, requires_grad=grad)
    k = torch.randn(2, kv_heads, 7, d, requires_grad=grad)
    v = torch.randn_like(k, requires_grad=grad)
    with torch.set_grad_enabled(grad):
        actual = model._attend(q, k, v, causal=causal)
        expected = F.scaled_dot_product_attention(q,
            k.repeat_interleave(heads//kv_heads, 1),
            v.repeat_interleave(heads//kv_heads, 1),
            is_causal=causal, dropout_p=0., scale=d**-0.5)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        if grad:
            probe = torch.randn_like(actual)
            ga = torch.autograd.grad(actual, (q,k,v), probe)
            gb = torch.autograd.grad(expected, (q,k,v), probe)
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)


def test_disabled_avq_empty_state():
    model = EnhancedMQA(96, 12, state_memory=False).eval()
    with torch.no_grad():
        _, cache = model.prefill_with_cache(torch.randn(1, 8, 96))
        assert cache[2].shape == (1, 0)
        assert cache[2].untyped_storage().nbytes() == 0
