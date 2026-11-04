"""BaseESA native decode: original retention math and API regression tests.

CUDA-specific tests are skipped without a CUDA toolkit.  All other tests
validate the inference-only dispatch policy without changing training math.
"""
import pytest
import torch

from mlbricks import ESA
from mlbricks.esa.fast_engine.core import BaseESA as BaseEngine


def test_base_has_native_decode_policy_and_no_extra_parameters():
    base = ESA(embd=24, head=6, variant="base", backend="auto", device="cpu")
    assert base.fused_decode == "auto"
    assert sorted(base.state_dict()) == sorted([
        "qgv.weight", "W_O.weight", "raw_alpha", "raw_beta"
    ])
    base.set_backend("pytorch")
    assert base.fused_decode == "off"
    base.set_backend("native")
    assert base.fused_decode == "on"
    with pytest.raises(RuntimeError, match="requires a CUDA input"):
        base.decode_step(torch.randn(2, 24))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_base_mixed_precision_preserves_fp32_bounds(dtype):
    model = ESA(embd=24, head=6, variant="base", backend="auto")
    model.to(dtype=dtype)
    assert model.qgv.weight.dtype == dtype
    assert model.W_O.weight.dtype == dtype
    assert model.raw_alpha.dtype == torch.float32
    assert model.raw_beta.dtype == torch.float32
    model.to(dtype=torch.float32)
    assert model.raw_alpha.dtype == torch.float32
    assert model.raw_beta.dtype == torch.float32


def test_base_cpu_prefill_decode_and_gradients_unchanged():
    torch.manual_seed(1001)
    base = ESA(embd=24, head=6, variant="base", backend="pytorch")
    reference = BaseEngine(dim=24, heads=6, scan="reference")
    reference.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(2, 11, 24)
    init = torch.randn(2, 6, 4, dtype=torch.float32)
    ya, sa = base(x, init)
    yb, sb = reference(x, init)
    torch.testing.assert_close(ya, yb, atol=0, rtol=0)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)

    ga = torch.autograd.grad(ya.square().mean(), tuple(base.parameters()))
    gb = torch.autograd.grad(yb.square().mean(), tuple(reference.parameters()))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    with torch.inference_mode():
        ya1, sa1 = base.decode_step(x[:, 0], init)
        yb1, sb1 = reference(x[:, :1], init)
        torch.testing.assert_close(ya1, yb1, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sa1, sb1, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/nvcc")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_base_native_fused_matches_reference_gpu(dtype):
    # FP32 parameters + FP32 state, regardless of projection precision.
    # This is the exact training-vs-fused recurrence comparison.
    from mlbricks.esa.fast_engine.base_decode import prepare_base_decode
    prepare_base_decode()
    torch.manual_seed(42)
    fast = ESA(embd=384, head=6, variant="base", backend="native", device="cuda").eval()
    slow = ESA(embd=384, head=6, variant="base", backend="pytorch", device="cuda").eval()
    fast.to(dtype=dtype)
    slow.to(dtype=dtype)
    slow.load_state_dict(fast.state_dict())
    x = torch.randn(2, 1, 384, device="cuda", dtype=dtype)
    state = torch.randn(2, 6, 64, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        yf, sf = fast.decode_step(x, state)
        yr, sr = slow.decode_step(x, state)
    tol = 5e-5 if dtype == torch.float32 else 3e-3
    torch.testing.assert_close(yf.float(), yr.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(sf, sr, atol=5e-5, rtol=5e-5)
