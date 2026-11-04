"""CUDA device by default with automatic execution dispatch."""
import inspect

import pytest
import torch

from mlbricks import ESAAdapter as ESA, ESAConfig


def test_esa_auto_constructor_defaults():
    signature = inspect.signature(ESA)
    assert signature.parameters["backend"].default == "auto"
    assert signature.parameters["device"].default == "cuda"
    assert ESAConfig(embd=24, head=6).backend == "auto"


def test_auto_default_routes_to_fastesaq_auto():
    # No GPU is needed to check the dispatch configuration itself.
    esa = ESA(embd=24, head=6, device=None)
    assert esa.backend == "auto"
    assert esa.resolved_backend() == "auto"
    assert esa.layer.requested_backend == "auto"
    assert esa.layer.scan == "auto"
    assert esa.layer.fused_decode == "auto"
    esa.set_backend("pytorch")
    assert esa.backend == "pytorch"
    assert esa.layer.requested_backend == "pytorch"
    esa.set_backend("auto")
    assert esa.backend == "auto"
    assert esa.layer.requested_backend == "auto"


def test_explicit_cuda_backend_routes_to_strict_native():
    esa = ESA(embd=24, head=6, backend="cuda", device=None)
    assert esa.resolved_backend() == "cuda"
    assert esa.layer.requested_backend == "native"
    assert esa.layer.scan == "parallel"
    assert esa.layer.fused_decode == "on"


def test_default_cuda_device_requires_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        ESA(embd=24, head=6)


def test_explicit_cpu_remains_available():
    esa = ESA(embd=24, head=6, backend="pytorch", device="cpu")
    x = torch.randn(2, 4, 24)
    y, state = esa.prefill(x)
    assert y.shape == x.shape
    assert state.dtype == torch.float32


def test_explicit_cpu_cannot_accidentally_use_strict_cuda_backend():
    esa = ESA(embd=24, head=6, backend="cuda", device="cpu")
    with pytest.raises(RuntimeError, match="requires a CUDA input"):
        esa(torch.randn(1, 2, 24))


def test_default_auto_works_with_explicit_cpu_device():
    esa = ESA(embd=24, head=6, device="cpu")
    assert esa.backend == "auto"
    y = esa(torch.randn(1, 3, 24))
    assert y.shape == (1, 3, 24)
