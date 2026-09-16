from __future__ import annotations

import torch
import torch.nn.functional as F

from mlbricks import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from mlbricks.ffnbrick import native


def _sample(d_model=32, state_dim=8):
    torch.manual_seed(123)
    x = torch.randn(2, 5, d_model)
    esa = torch.randn_like(x)
    prev_esa = torch.randn_like(x)
    state = torch.randn(2, 5, state_dim)
    return x, esa, prev_esa, state


def test_state_aware_python_path_matches_original_equations():
    x, esa, prev_esa, state = _sample()
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=False)

    update, next_state = layer.forward_python(x, esa, prev_esa, state)

    depth_candidate = layer.depth_to_candidate(layer.depth_embedding)
    depth_write = layer.depth_to_write(layer.depth_embedding)
    depth_value = layer.depth_to_value(layer.depth_embedding)
    esa_delta = esa - prev_esa
    delta_magnitude = torch.sqrt(
        esa_delta.float().square().mean(dim=-1, keepdim=True) + 1e-6
    ).to(esa.dtype)
    scaled_delta = torch.exp(layer.delta_magnitude_log_scale) * delta_magnitude

    candidate_esa = esa + torch.sigmoid(layer.candidate_transition_logit) * esa_delta
    write_esa = esa + torch.sigmoid(layer.write_transition_logit) * esa_delta
    candidate = torch.tanh(
        layer.x_candidate(x)
        + layer.esa_candidate(candidate_esa)
        + layer.state_candidate(state)
        + depth_candidate
    )
    write_gate = torch.sigmoid(
        layer.x_write(x)
        + layer.esa_write(write_esa)
        + layer.state_write(state)
        + depth_write
    )
    retain_gate = torch.sigmoid(
        layer.retain_logit - scaled_delta * layer.retain_delta_scale
    )
    expected_state = (
        (1.0 - write_gate) * (retain_gate * state) + write_gate * candidate
    )
    value = F.silu(layer.value(x) + depth_value)
    read_gate = torch.sigmoid(layer.read_logit + scaled_delta * layer.read_delta_scale)
    expected_update = layer.output(expected_state * value * read_gate)

    torch.testing.assert_close(next_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(update, expected_update, atol=0, rtol=0)


def test_state_packing_replaces_exact_projection_groups():
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=False)
    x, _, _, state = _sample()
    p = native._state_packed_params(layer)

    x_proj = F.linear(x, p[0], p[1])
    torch.testing.assert_close(x_proj[..., :8], layer.x_candidate(x))
    torch.testing.assert_close(x_proj[..., 8:16], layer.x_write(x))
    torch.testing.assert_close(x_proj[..., 16:24], layer.value(x))

    state_proj = F.linear(state, p[2])
    torch.testing.assert_close(state_proj[..., :8], layer.state_candidate(state))
    torch.testing.assert_close(state_proj[..., 8:16], layer.state_write(state))

    torch.testing.assert_close(p[7], layer.depth_to_candidate(layer.depth_embedding))
    torch.testing.assert_close(p[8], layer.depth_to_write(layer.depth_embedding))
    torch.testing.assert_close(p[9], layer.depth_to_value(layer.depth_embedding))
    torch.testing.assert_close(p[14], torch.sigmoid(layer.candidate_transition_logit))
    torch.testing.assert_close(p[15], torch.sigmoid(layer.write_transition_logit))
    torch.testing.assert_close(p[16], torch.exp(layer.delta_magnitude_log_scale))


def test_state_packed_cache_invalidates_after_weight_update():
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=False)
    p1 = native._state_packed_params(layer)
    with torch.no_grad():
        layer.x_candidate.weight.add_(0.25)
    p2 = native._state_packed_params(layer)
    assert p1[0] is not p2[0]
    torch.testing.assert_close(p2[0][:8], layer.x_candidate.weight)


def test_micro_packing_is_exact():
    torch.manual_seed(5)
    layer = MicroVirtualFFN(32, hidden_dim=6, refinements=3, use_native=False)
    packed, down = native._micro_packed_params(layer)
    assert packed.shape == (3, 12, 32)
    assert down.shape == (3, 32, 6)
    torch.testing.assert_close(packed[:, :6], layer.gate)
    torch.testing.assert_close(packed[:, 6:], layer.up)
    torch.testing.assert_close(down, layer.down)


def test_micro_refine_preserves_original_sequential_semantics():
    torch.manual_seed(9)
    layer = MicroVirtualFFN(32, hidden_dim=6, refinements=3, use_native=False)
    with torch.no_grad():
        layer.down.normal_(0.0, 0.03)
    x = torch.randn(2, 4, 32)

    expected = x
    for i in range(layer.refinements):
        expected = expected + layer.forward_python(expected, i)

    actual = layer.refine(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_native_dispatch_is_never_used_for_autograd(monkeypatch):
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=True)
    x = torch.randn(1, 1, 32, requires_grad=True)
    monkeypatch.setattr(native, "_C", object())
    assert native.inference_native_allowed(layer, x) is False


def test_native_dispatch_is_allowed_only_in_eager_no_grad(monkeypatch):
    layer = StateAwareFFN(32, 8, 4, 1, 3, use_native=True)
    x = torch.randn(1, 1, 32)
    monkeypatch.setattr(native, "_C", object())
    monkeypatch.setattr(native, "is_compiling", lambda: False)
    with torch.no_grad():
        assert native.inference_native_allowed(layer, x) is True

    monkeypatch.setattr(native, "is_compiling", lambda: True)
    with torch.no_grad():
        assert native.inference_native_allowed(layer, x) is False


def test_virtual_python_branch_still_learns():
    x, esa, prev_esa, state = _sample()
    layer = VirtualStateAwareFFN(32, 8, 4, 1, 3, 2, 6, use_native=False)
    update, next_state = layer(x, esa, prev_esa, state)
    (update.square().mean() + next_state.square().mean()).backward()
    assert layer.virtual_refiner.down.weight.grad is not None
