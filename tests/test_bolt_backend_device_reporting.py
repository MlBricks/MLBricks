import pytest
import torch

from mlbricks import Bolt
from mlbricks.bolt import presets


@pytest.mark.parametrize("profile", ["standard", "compact"])
@pytest.mark.parametrize("state_memory", [True, False])
def test_cpu_backend_reporting(monkeypatch, profile, state_memory):
    monkeypatch.setattr(
        presets._accelerated, "available", lambda: True
    )
    monkeypatch.setattr(
        presets._native, "get_extension", lambda: object()
    )

    model = Bolt(
        embed_dim=64,
        num_heads=4,
        profile=profile,
        state_memory=state_memory,
        backend="auto",
        device="cpu",
        precision="fp32",
    )

    assert next(model.parameters()).device.type == "cpu"
    assert model.resolved_backend().startswith("pytorch")

    x = torch.randn(1, 5, 64, requires_grad=True)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_native_backend_requires_cuda(monkeypatch):
    monkeypatch.setattr(
        presets._accelerated, "available", lambda: True
    )

    model = Bolt(
        embed_dim=64,
        num_heads=4,
        backend="native",
        device="cpu",
        precision="fp32",
    )

    assert "unavailable" in model.resolved_backend()

    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        model(torch.randn(1, 3, 64))