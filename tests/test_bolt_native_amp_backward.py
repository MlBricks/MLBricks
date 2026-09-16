from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mlbricks import Bolt
import mlbricks.bolt.attention as bolt_attention


def test_native_stage1_backward_accepts_mixed_master_dtype(monkeypatch):
    "Regression for b1: native adjoint dtype may differ from master weights."
    B, T, D = 2, 3, 4
    H, R = 1, 2
    packed = 3 * H * R

    x = torch.randn(B, T, D, dtype=torch.float32)
    weight = torch.randn(packed, D, dtype=torch.float32)
    c = torch.randn(B, H, T, R, dtype=torch.float64)
    rho = torch.rand(B, H, T, dtype=torch.float32)
    gate = torch.randn(B, H, T, R, dtype=torch.float64)
    dq = torch.randn_like(c)
    dc = torch.randn_like(c)
    drho = torch.randn_like(rho)
    dqcg = torch.randn(B, T, packed, dtype=torch.float64)

    class FakeExt:
        def gauss_stage1_backward(self, *args, **kwargs):
            return dqcg

    monkeypatch.setattr(bolt_attention, 'load_cuda_extension', lambda: FakeExt())
    ctx = SimpleNamespace(
        saved_tensors=(x, weight, c, rho, gate),
        has_bias=True,
        num_heads=H,
        latent_dim=R,
    )

    dx, dw, db, *rest = bolt_attention._BoltNativeStage1Fn.backward(ctx, dq, dc, drho)

    work = dqcg.reshape(B * T, packed)
    x_flat = x.reshape(B * T, D)
    expected_dx = work.matmul(weight.to(work.dtype)).reshape_as(x).to(x.dtype)
    expected_dw = work.transpose(0, 1).matmul(x_flat.to(work.dtype)).to(weight.dtype)
    expected_db = work.sum(dim=0).to(weight.dtype)

    torch.testing.assert_close(dx, expected_dx)
    torch.testing.assert_close(dw, expected_dw)
    torch.testing.assert_close(db, expected_db)
    assert dx.dtype == x.dtype
    assert dw.dtype == weight.dtype
    assert db.dtype == weight.dtype
    assert rest == [None, None, None]


@pytest.mark.parametrize('bias', [False, True])
def test_bolt_native_amp_fp16_gradient_parity_when_cuda_available(bias):
    "Exact b1 regression: FP16 AMP working tensors with FP32 master Parameters."
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')

    ext = bolt_attention.load_cuda_extension()
    if ext is None or not hasattr(ext, 'gauss_stage1_forward') or not hasattr(ext, 'gauss_stage1_backward'):
        pytest.skip('native Bolt Stage-1 extension unavailable')

    torch.manual_seed(2026)
    common = dict(
        d_model=128,
        num_heads=4,
        latent_dim=32,
        bias=bias,
        dropout=0.0,
        causal=True,
        use_sdpa=True,
        position=None,
        native_full_sequence=False,
    )

    ref = Bolt(**common, backend='pytorch').cuda().float().train()
    opt = Bolt(**common, backend='native').cuda().float().train()
    opt.load_state_dict(ref.state_dict())

    x_ref = torch.randn(2, 64, 128, device='cuda', dtype=torch.float16, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    with torch.autocast('cuda', dtype=torch.float16):
        y_ref = ref(x_ref)
        y_opt = opt(x_opt)

    torch.testing.assert_close(y_opt, y_ref, rtol=5e-3, atol=5e-3)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_opt.backward(grad)

    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-2, atol=1e-2)

    ref_params = dict(ref.named_parameters())
    opt_params = dict(opt.named_parameters())
    assert ref_params.keys() == opt_params.keys()

    for name in ref_params:
        rg = ref_params[name].grad
        og = opt_params[name].grad
        assert rg is not None, name
        assert og is not None, name
        assert torch.isfinite(og).all(), name
        assert og.dtype == opt_params[name].dtype == torch.float32, name
        torch.testing.assert_close(og, rg, rtol=2e-2, atol=2e-2)
