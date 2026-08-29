import torch

from mlbricks.esa.auto_backend import select_esa_auto_backend
from mlbricks.esa.generation import _decode_runtime_backend
from mlbricks.esa.layer import ESA


def test_auto_selector_keeps_cpu_on_pytorch_when_cuda_extension_exists(monkeypatch):
    monkeypatch.delenv("MLBRICKS_NATIVE_CPU", raising=False)
    x = torch.randn(2, 1, 32)

    route = select_esa_auto_backend(
        x,
        workload="decode",
        training=False,
        native_available=True,
        native_cuda_available=True,
    )

    assert route == "pytorch"


def test_auto_selector_allows_cpu_native_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MLBRICKS_NATIVE_CPU", "1")
    x = torch.randn(2, 1, 32)

    route = select_esa_auto_backend(
        x,
        workload="decode",
        training=False,
        native_available=True,
        native_cuda_available=True,
    )

    assert route == "native"


def test_decode_auto_does_not_treat_cuda_build_as_cpu_native(monkeypatch):
    monkeypatch.delenv("MLBRICKS_NATIVE_CPU", raising=False)

    # Simulate the exact Kaggle condition that exposed the regression: the
    # extension exists and has CUDA kernels, while this ESA element/input is CPU.
    import importlib
    native = importlib.import_module("mlbricks.esa.native")

    monkeypatch.setattr(native, "available", lambda: True)
    monkeypatch.setattr(native, "cuda_available", lambda: True)

    layer = ESA(embd=32, head=4, backend="auto", device="cpu").eval()
    x = torch.randn(2, 32)

    with torch.no_grad():
        assert _decode_runtime_backend(layer, x) == "pytorch"
