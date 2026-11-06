"""Opt-in CUDA qualification for BOLT's width-tiled AVQ scans.

Run on an NVIDIA GPU after compiling both MLBRICKS_BUILD_BOLT_ACCEL=1
extensions: MLBRICKS_VALIDATE_AVQ384_NATIVE=1 python -m pytest -q
 tests/test_bolt_avq384_native_scan_cuda.py
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("MLBRICKS_VALIDATE_AVQ384_NATIVE") != "1",
    reason="native AVQ GPU qualification is opt-in; set MLBRICKS_VALIDATE_AVQ384_NATIVE=1",
)


@pytest.fixture(scope="module")
def native():
    if not torch.cuda.is_available():
        pytest.fail("MLBRICKS_VALIDATE_AVQ384_NATIVE=1 requires CUDA")
    # Do NOT silently skip missing extensions in an explicit native-validation run.
    short = importlib.import_module("mlbricks._bolt_scan_short")
    segmented = importlib.import_module("mlbricks._bolt_scan_segmented")
    return short, segmented


@torch.no_grad()
def _raw_reference(za, zv, s0, de):
    """FP32 sequential recurrence and analytical reverse-mode reference."""
    B, T, D = za.shape
    a = za.float().sigmoid()
    u = zv.float().tanh()
    v = 0.1 * u
    state = s0
    rows = []
    for t in range(T):
        state = a[:, t] * state + v[:, t]
        rows.append(state)
    e = torch.stack(rows, dim=1)
    dza = torch.empty_like(za)
    dzv = torch.empty_like(zv)
    carry = torch.zeros(B, D, dtype=torch.float32, device=za.device)
    for t in range(T - 1, -1, -1):
        gt = de[:, t] + carry
        prev = s0 if t == 0 else e[:, t - 1]
        dza[:, t] = (gt * prev * a[:, t] * (1 - a[:, t])).to(za.dtype)
        dzv[:, t] = (gt * 0.1 * (1 - u[:, t].square())).to(zv.dtype)
        carry = gt * a[:, t]
    return e, dza, dzv, carry


@torch.no_grad()
def _affine_reference(a, v, s0, de):
    B, T, D = a.shape
    state = s0
    rows = []
    for t in range(T):
        state = a[:, t] * state + v[:, t]
        rows.append(state)
    e = torch.stack(rows, dim=1)
    da = torch.empty_like(a)
    dv = torch.empty_like(v)
    carry = torch.zeros(B, D, device=a.device)
    for t in range(T - 1, -1, -1):
        gt = de[:, t] + carry
        prev = s0 if t == 0 else e[:, t - 1]
        da[:, t] = gt * prev
        dv[:, t] = gt
        carry = gt * a[:, t]
    return e, da, dv, carry


def _close(actual, expected, *, label, atol=4e-4, rtol=2e-3):
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (label, actual.dtype, expected.dtype)
    assert torch.isfinite(actual).all(), label
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)


@pytest.mark.parametrize("width,length,dtype,path", [
    (192, 17, torch.float32, "short"),
    (256, 512, torch.float32, "short"),
    (257, 33, torch.float16, "short"),
    (384, 128, torch.float32, "short"),
    (384, 512, torch.float16, "short"),
    (384, 513, torch.float32, "segmented"),
    (384, 1025, torch.float16, "segmented"),
    (513, 769, torch.float32, "segmented"),
])
def test_raw_cuda_scan_forward_backward(native, width, length, dtype, path):
    short, _ = native
    torch.manual_seed(321 + width + length)
    B = 2
    # Mimic the packed A/V/R projection slices actually produced by BOLT.
    packed = torch.randn(B, length, 3 * width, device="cuda", dtype=dtype) * 0.7
    za = packed[..., :width]
    zv = packed[..., width:2*width]
    assert za.stride(-1) == 1 and not za.is_contiguous()
    s0 = torch.randn(B, width, device="cuda", dtype=torch.float32) * 0.05
    de = torch.randn(B, length, width, device="cuda") * 0.2
    use_short = path == "short"
    got = short.forward(za, zv, s0, 8, use_short)
    dza, dzv, d0 = short.backward(za, zv, got, s0, de, 8, use_short)
    want, wa, wv, w0 = _raw_reference(za, zv, s0, de)
    _close(got, want, label="raw forward")
    _close(dza, wa, label="raw za backward", atol=2e-3 if dtype == torch.float16 else 4e-4)
    _close(dzv, wv, label="raw zv backward", atol=2e-3 if dtype == torch.float16 else 4e-4)
    _close(d0, w0, label="raw state0 backward")


@pytest.mark.parametrize("width,length", [
    (192, 17), (256, 256), (257, 513),
    (384, 1), (384, 513), (384, 1025), (513, 769),
])
def test_segmented_cuda_scan_forward_backward(native, width, length):
    _, seg = native
    torch.manual_seed(712 + width + length)
    B = 2
    a = (torch.rand(B, length, width, device="cuda") * 0.5 + 0.25).contiguous()
    v = (torch.randn(B, length, width, device="cuda") * 0.07).contiguous()
    s0 = torch.randn(B, width, device="cuda") * 0.02
    de = torch.randn(B, length, width, device="cuda") * 0.2
    got = seg.forward(a, v, s0)
    da, dv, d0 = seg.backward(a, got, s0, de)
    want, wa, wv, w0 = _affine_reference(a, v, s0, de)
    _close(got, want, label="affine forward")
    _close(da, wa, label="affine da")
    _close(dv, wv, label="affine dv")
    _close(d0, w0, label="affine state0")


def test_actual_python_scan_dispatch_and_gradients(native):
    from mlbricks.bolt import accelerated
    for T in (128, 600):
        W = 384
        torch.manual_seed(30 + T)
        za = torch.randn(2, T, W, device="cuda", requires_grad=True)
        zv = torch.randn(2, T, W, device="cuda", requires_grad=True)
        result = accelerated.scan(za, zv, latent_dim=32)
        assert result is not None and result.shape == (2, T, W)
        assert result.dtype == torch.float32 and result.is_cuda
        loss = result.square().mean()
        gza, gzv = torch.autograd.grad(loss, (za, zv))
        assert gza.shape == za.shape and gzv.shape == zv.shape
        assert torch.isfinite(gza).all() and torch.isfinite(gzv).all()


def test_reject_cross_device_inputs(native):
    short, segmented = native
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    za = torch.zeros(1, 8, 384, device="cuda:0")
    zv = torch.zeros(1, 8, 384, device="cuda:1")
    state = torch.zeros(1, 384, device="cuda:0")
    with pytest.raises(RuntimeError, match="device mismatch"):
        short.forward(za, zv, state, 8, True)
    a = torch.full_like(za, 0.5)
    v = torch.zeros_like(za, device="cuda:1")
    with pytest.raises(RuntimeError, match="device mismatch"):
        segmented.forward(a, v, state)
