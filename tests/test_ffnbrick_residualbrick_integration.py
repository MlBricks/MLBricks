import torch

import mlbricks
from mlbricks import (
    MicroVirtualFFN,
    ResController,
    StateAwareFFN,
    VirtualStateAwareFFN,
)


def _ffn_sample():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 32)
    esa = torch.randn_like(x)
    prev_esa = torch.randn_like(x)
    state = torch.randn(2, 5, 8)
    return x, esa, prev_esa, state


def test_component_namespaces_and_top_level_exports():
    assert hasattr(mlbricks, "ffnbrick")
    assert hasattr(mlbricks, "residualbrick")
    assert mlbricks.ffnbrick.StateAwareFFN is StateAwareFFN
    assert mlbricks.residualbrick.ResController is ResController


def test_micro_virtual_ffn_identity_startup():
    x, *_ = _ffn_sample()
    layer = MicroVirtualFFN(32, hidden_dim=4, refinements=2, use_native=False)
    assert torch.equal(layer(x, 0), torch.zeros_like(x))
    assert torch.equal(layer.refine(x), x)


def test_state_aware_ffn_shapes_and_gradients():
    x, esa, prev_esa, state = _ffn_sample()
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=False)
    update, next_state = layer(x, esa, prev_esa, state)
    assert update.shape == x.shape
    assert next_state.shape == state.shape
    (update.square().mean() + next_state.square().mean()).backward()
    assert layer.output.weight.grad is not None


def test_virtual_state_aware_ffn_runs():
    x, esa, prev_esa, state = _ffn_sample()
    layer = VirtualStateAwareFFN(32, 8, 4, 1, 3, 2, 6, use_native=False)
    update, next_state = layer(x, esa, prev_esa, state)
    assert update.shape == x.shape
    assert next_state.shape == state.shape


def test_res_controller_shapes_gradients_and_identity():
    controller = ResController(update_ratio=0.18, use_native=False)
    residual = torch.randn(2, 4, 16, requires_grad=True)
    update = torch.randn_like(residual, requires_grad=True)
    output = controller(residual, update)
    assert output.shape == residual.shape
    assert output.dtype == residual.dtype
    output.square().mean().backward()
    assert residual.grad is not None
    assert update.grad is not None

    with torch.no_grad():
        x = torch.randn(2, 4, 16)
        y = controller(x, torch.zeros_like(x))
        torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)


def test_component_backend_diagnostics_do_not_collide():
    assert isinstance(mlbricks.ffnbrick_native_backend_available(), bool)
    assert mlbricks.ffnbrick_native_backend_name() in {"python", "cpp", "cpp/cuda"}
    assert isinstance(mlbricks.residualbrick_native_backend_available(), bool)
    assert mlbricks.residualbrick_native_backend_name() in {"python", "cpp", "cpp/cuda"}
