"""Regression gates for inference-only chunked AVQ (no training changes)."""
from unittest.mock import patch

import pytest
import torch
from mlbricks import Bolt
from mlbricks.bolt.enhanced_mqa import EnhancedMQA

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


def check_close(a, b, device):
    # Relative RMS matches the research benchmark; also bound elementwise error.
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if device == 'cuda':
        err = (a.float() - b.float()).square().mean().sqrt()
        scale = b.float().square().mean().sqrt().clamp_min(1e-8)
        assert (err / scale).item() <= 0.005
        torch.testing.assert_close(a, b, rtol=0.01, atol=0.005)
    else:
        torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('retention', ['learned', 'decay'])
@pytest.mark.parametrize('heads', [48, 64])
@pytest.mark.parametrize('batch', [1, 4])
def test_native_prefill_and_decode(device, retention, heads, batch):
    torch.manual_seed(17)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    # Public BOLT wrapper; small width and shorter chunks keep this inexpensive.
    layer = Bolt(heads * 8, heads, mode='quantised', kv_heads=1,
                 retention=retention, backend='pytorch', device=device)
    layer = layer.to(device=device, dtype=dtype).eval()
    a = layer._modern
    assert a.avq_prefill_chunk_size == 2048
    x = torch.randn(batch, 58, heads * 8, device=device, dtype=dtype)
    with torch.no_grad():
        a.avq_prefill_chunk_size = 0
        ref, ref_cache = layer.prefill_with_cache(x[:, :55])
        a.avq_prefill_chunk_size = 17  # 3 full chunks and remainder of 4
        with patch.object(a.avq, '_prefill_chunk', wraps=a.avq._prefill_chunk) as call:
            out, cache = layer.prefill_with_cache(x[:, :55])
            assert [c.args[0].size(1) for c in call.call_args_list] == [17, 17, 17, 4]
        check_close(out, ref, device)
        for got, expected in zip(cache, ref_cache):
            check_close(got, expected, device)
        assert cache[2].dtype == torch.float32
        assert cache[2].untyped_storage().nbytes() == cache[2].numel() * 4
        static = a.allocate_cache(batch_size=batch, capacity=58)
        static.load_prefix(*cache)
        for i in range(55, 58):
            expected, ref_cache = layer.decode_step(x[:, i:i+1], ref_cache)
            got, cache = layer.decode_step(x[:, i:i+1], cache)
            static_out = a.decode_with_cache(x[:, i:i+1], static, position=i)
            check_close(got, expected, device)
            check_close(static_out, got, device)
            check_close(static.state, cache[2], device)
            for actual, ref_tensor in zip(cache, ref_cache):
                check_close(actual, ref_tensor, device)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('retention', ['learned', 'decay'])
def test_real_2048_chunk_boundary(device, retention):
    torch.manual_seed(53)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    # AVQ primitive at the actual production chunk boundary. No quadratic attention.
    a = EnhancedMQA(48, 6, avq_dim=29, retention=retention,
                    backend='pytorch').to(device=device, dtype=dtype).eval().avq
    x = torch.randn(2, 4103, 48, device=device, dtype=dtype)
    with torch.no_grad():
        ref, state = a(x, return_state=True)
        with patch.object(a, '_prefill_chunk', wraps=a._prefill_chunk) as call:
            got, carried = a._prefill_chunked(x, 2048)
            assert [c.args[0].size(1) for c in call.call_args_list] == [2048, 2048, 7]
        check_close(got, ref, device)
        check_close(carried, state, device)
        assert carried.untyped_storage().nbytes() == carried.numel() * 4


@pytest.mark.parametrize('training,kv_heads,causal,seq,chunk', [
    (True, 1, True, 19, 8), (False, 1, True, 8, 8),
    (False, 1, True, 19, 0), (False, 2, True, 19, 8),
    (False, 1, False, 19, 8),
])
def test_dispatch_guards(training, kv_heads, causal, seq, chunk):
    a = EnhancedMQA(48, 6, kv_heads=kv_heads, causal=causal,
                    backend='pytorch').train(training)
    a.avq_prefill_chunk_size = chunk
    with patch.object(a.avq, '_prefill_chunked', side_effect=AssertionError('unexpected chunking')):
        a.prefill_with_cache(torch.randn(1, seq, 48))


@pytest.mark.parametrize('retention', ['learned', 'decay'])
def test_forward_and_training_not_chunked(retention):
    a = EnhancedMQA(48, 6, retention=retention, backend='pytorch')
    a.avq_prefill_chunk_size = 8
    keys = tuple(a.state_dict())
    x = torch.randn(2, 19, 48, requires_grad=True)
    with patch.object(a.avq, '_prefill_chunked', side_effect=AssertionError('unexpected chunking')):
        a.train()
        a(x).square().mean().backward()
        assert a.avq.project.weight.grad is not None
        assert torch.isfinite(a.avq.project.weight.grad).all()
        assert torch.isfinite(x.grad).all()
        a.eval()
        with torch.no_grad():
            a(x.detach())  # ordinary forward still uses the original path
    assert tuple(a.state_dict()) == keys
    fresh = EnhancedMQA(48, 6, retention=retention, backend='pytorch')
    fresh.load_state_dict(a.state_dict(), strict=True)


@pytest.mark.parametrize('retention', ['learned', 'decay'])
def test_default_automatic_chunk_dispatch(retention):
    # Exercise the shipped setting through an actual long cached prefill.
    torch.manual_seed(71)
    a = EnhancedMQA(16, 2, retention=retention, backend='pytorch').eval()
    x = torch.randn(1, 2051, 16)
    assert a.avq_prefill_chunk_size == 2048
    with torch.no_grad():
        with patch.object(a.avq, '_prefill_chunk', wraps=a.avq._prefill_chunk) as call:
            y, cache = a.prefill_with_cache(x)
            assert [c.args[0].size(1) for c in call.call_args_list] == [2048, 3]
        a.avq_prefill_chunk_size = 0
        ref, ref_cache = a.prefill_with_cache(x)
        check_close(y, ref, 'cpu')
        for got, expected in zip(cache, ref_cache):
            check_close(got, expected, 'cpu')
