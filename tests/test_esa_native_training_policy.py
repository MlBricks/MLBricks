from __future__ import annotations

from types import SimpleNamespace

import torch

from mlbricks.esa import native


class _FakeNative:
    def has_cuda(self):
        return True


def test_native_training_eligibility_requires_cuda_autograd_registration(monkeypatch):
    monkeypatch.delenv("MLBRICKS_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(native, "_C", _FakeNative())
    monkeypatch.setattr(native, "_CUSTOM_OPS_REGISTERED", True)
    tensor = SimpleNamespace(is_cuda=True, dtype=torch.float16)

    with torch.enable_grad():
        assert native.training_enabled_for(tensor) is True
        assert native.enabled_for(tensor) is True
        # Fused readout/full-forward kernels remain inference-only.
        assert native.fused_enabled_for(tensor) is False


def test_native_training_eligibility_rejects_missing_autograd_registration(monkeypatch):
    monkeypatch.delenv("MLBRICKS_DISABLE_NATIVE", raising=False)
    monkeypatch.setattr(native, "_C", _FakeNative())
    monkeypatch.setattr(native, "_CUSTOM_OPS_REGISTERED", False)
    tensor = SimpleNamespace(is_cuda=True, dtype=torch.float16)

    with torch.enable_grad():
        assert native.training_enabled_for(tensor) is False
        assert native.enabled_for(tensor) is False


def test_native_training_scan_uses_planned_autograd_route_without_env_opt_in(monkeypatch):
    monkeypatch.delenv("MLBRICKS_NATIVE_TRAINING", raising=False)
    monkeypatch.setattr(native, "_C", _FakeNative())
    monkeypatch.setattr(native, "_planner_supported", lambda A, B: True)

    sentinel = torch.tensor(123.0)
    calls = []

    def fake_planned(A, B, compass):
        calls.append(compass)
        return sentinel

    monkeypatch.setattr(native, "thunder_scan_planned", fake_planned)

    A = torch.ones(1, 8, 1, 1, requires_grad=True)
    B = torch.ones_like(A, requires_grad=True)
    out = native.thunder_scan(A, B, 16)

    assert out is sentinel
    assert calls == [16]
