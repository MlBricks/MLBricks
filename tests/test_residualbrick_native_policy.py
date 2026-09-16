import torch

from mlbricks import ResController
from mlbricks.residualbrick import native as residual_native


def test_rescontroller_python_formula_matches_reference_structure():
    torch.manual_seed(7)
    controller = ResController(update_ratio=0.18, use_native=False)
    residual = torch.randn(2, 3, 16)
    update = torch.randn_like(residual)
    out = controller(residual, update)

    eps = controller.eps
    residual32 = residual.float()
    update32 = update.float()
    residual_rms = torch.sqrt(residual32.square().mean(-1, keepdim=True) + eps)
    update_rms = torch.sqrt(update32.square().mean(-1, keepdim=True) + eps)
    allowed_update = controller.update_ratio * residual_rms
    hard_update_scale = torch.clamp(allowed_update / (update_rms + eps), max=1.0)
    update_pressure = update_rms / (allowed_update + eps)
    update_gate = torch.sigmoid(controller.update_softness * (update_pressure - 1.0))
    update_scale = 1.0 - update_gate * (1.0 - hard_update_scale)
    bounded_update = update32 * update_scale
    candidate = residual32 + bounded_update
    candidate_rms = torch.sqrt(candidate.square().mean(-1, keepdim=True) + eps)
    allowed_stream = controller.stream_ratio * residual_rms
    hard_stream_scale = torch.clamp(allowed_stream / (candidate_rms + eps), max=1.0)
    stream_pressure = candidate_rms / (allowed_stream + eps)
    stream_gate = torch.sigmoid(controller.stream_softness * (stream_pressure - 1.0))
    stream_scale = 1.0 - stream_gate * (1.0 - hard_stream_scale)
    expected = (residual32 + bounded_update * stream_scale).to(residual.dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_native_policy_does_not_hide_training_graph(monkeypatch):
    controller = ResController(update_ratio=0.18, use_native=True)
    residual = torch.randn(2, 4, 8, requires_grad=True)
    update = torch.randn_like(residual, requires_grad=True)
    monkeypatch.setattr(residual_native, "_C", object())
    monkeypatch.setattr(residual_native, "is_compiling", lambda: False)

    called = False
    def fail_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native path must not run with autograd enabled")

    monkeypatch.setattr(residual_native, "residual_forward", fail_native)
    controller(residual, update).square().mean().backward()
    assert not called
    assert residual.grad is not None and update.grad is not None


def test_native_policy_does_not_hide_torch_compile_graph(monkeypatch):
    controller = ResController(update_ratio=0.18, use_native=True)
    residual = torch.randn(2, 4, 8)
    update = torch.randn_like(residual)
    monkeypatch.setattr(residual_native, "_C", object())
    monkeypatch.setattr(residual_native, "is_compiling", lambda: True)

    called = False
    def fail_native(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native path must not run while compiling")

    monkeypatch.setattr(residual_native, "residual_forward", fail_native)
    with torch.no_grad():
        out = controller(residual, update)
    assert not called
    assert out.shape == residual.shape


def test_cpu_inference_keeps_python_path(monkeypatch):
    controller = ResController(update_ratio=0.18, use_native=True)
    residual = torch.randn(2, 4, 8)
    update = torch.randn_like(residual)
    monkeypatch.setattr(residual_native, "_C", object())
    monkeypatch.setattr(residual_native, "is_compiling", lambda: False)
    with torch.no_grad():
        assert not residual_native.inference_native_allowed(controller, residual, update)
