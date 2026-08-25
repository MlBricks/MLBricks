import inspect

import pytest
import torch

from mlbricks import ESA


def test_default_backend_and_compass():
    signature = inspect.signature(ESA)
    assert signature.parameters["backend"].default == "auto"
    assert signature.parameters["compass"].default == "auto"


def test_thunder_forward_cpu_auto():
    layer = ESA(embd=32, head=4, precision="fp32", device=None)
    x = torch.randn(2, 16, 32)
    y = layer(x)
    assert y.shape == x.shape
    assert layer.backend == "auto"
    assert layer.compass == "auto"


def test_manual_compass_is_preserved():
    layer = ESA(embd=32, head=4, compass=16, precision="fp32", device=None)
    assert layer.compass == 16


def test_uniform_backend_policy_and_legacy_aliases():
    assert ESA(embd=32, head=4, backend="pytorch", device=None).backend == "pytorch"
    assert ESA(embd=32, head=4, backend="native", device=None).backend == "native"
    # Historical execution names are compatibility aliases only.
    assert ESA(embd=32, head=4, backend="thunder", device=None).backend == "auto"
    assert ESA(embd=32, head=4, backend="pulse", device=None).backend == "pytorch"
    with pytest.raises(ValueError):
        ESA(embd=32, head=4, backend="flair", device=None)


def test_old_dimension_names_are_not_public_api():
    with pytest.raises(TypeError):
        ESA(n_embd=32, n_head=4)
