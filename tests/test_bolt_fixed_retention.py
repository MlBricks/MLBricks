"""Correctness and compatibility checks for fixed per-head AVQ retention."""

import torch

from mlbricks.bolt.enhanced_mqa import EnhancedMQA
from mlbricks.bolt.presets import BoltStateMemory, _head_retention_rates


def _expanded_coefficients(width, heads):
    rates = torch.tensor(_head_retention_rates(heads), dtype=torch.float32)
    channel_heads = torch.div(
        torch.arange(width, dtype=torch.long) * heads,
        width,
        rounding_mode="floor",
    ).clamp_(max=heads - 1)
    return torch.exp(-rates[channel_heads])


def test_enhanced_mqa_uses_two_branch_projection_and_configured_heads():
    layer = EnhancedMQA(
        d_model=60,
        num_heads=5,
        avq_dim=23,
        state_memory=True,
        backend="pytorch",
    )
    memory = layer.avq

    assert memory._fixed_retention_enabled
    assert memory._fixed_retention_heads == 5
    assert memory.project.out_features == 2 * memory.width
    assert memory._fixed_retention_coefficients.shape == (23,)
    torch.testing.assert_close(
        memory._fixed_retention_coefficients,
        _expanded_coefficients(23, 5),
        rtol=0,
        atol=0,
    )


def test_fixed_projection_copies_only_active_write_and_read_rows_without_rng_shift():
    torch.manual_seed(41)
    memory = BoltStateMemory(
        input_dim=60,
        width=23,
        latent_dim=12,
        device="cpu",
        backend="pytorch",
    )
    old_weight = memory.project.weight.detach().clone()
    old_bias = memory.project.bias.detach().clone()
    rng_before = torch.get_rng_state().clone()

    memory._enable_fixed_retention(5)

    expected_weight = torch.cat((old_weight[23:46], old_weight[46:69]), dim=0)
    expected_bias = torch.cat((old_bias[23:46], old_bias[46:69]), dim=0)
    torch.testing.assert_close(memory.project.weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(memory.project.bias, expected_bias, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    assert memory.project.out_features == 46


def test_fixed_recurrence_matches_manual_reference_and_incremental_step():
    torch.manual_seed(53)
    layer = EnhancedMQA(
        d_model=60,
        num_heads=5,
        avq_dim=23,
        state_memory=True,
        backend="pytorch",
    )
    memory = layer.avq
    with torch.no_grad():
        memory.gamma.fill_(0.4)

    x = torch.randn(2, 7, 60)
    raw = memory.project(x)
    v_raw, r_raw = raw.split(memory.width, dim=-1)
    a = memory._fixed_retention_coefficients.view(1, 1, -1).expand(2, 7, -1)
    v = (0.1 * torch.tanh(v_raw)).float()
    r = torch.tanh(r_raw).float()

    states = []
    state = torch.zeros(2, memory.width)
    for index in range(x.shape[1]):
        state = a[:, index] * state + v[:, index]
        states.append(state)
    reference_state = torch.stack(states, dim=1)
    reference = x + memory.gamma * memory.back(memory.norm(reference_state) * (1.0 + r))

    actual, final_state = memory(x, return_state=True)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(final_state, reference_state[:, -1], rtol=1e-6, atol=1e-6)

    step_state = torch.zeros(2, memory.width)
    step_outputs = []
    for index in range(x.shape[1]):
        step_output, step_state = memory.step(x[:, index:index + 1], step_state)
        step_outputs.append(step_output)
    incremental = torch.cat(step_outputs, dim=1)
    torch.testing.assert_close(incremental, actual, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(step_state, final_state, rtol=1e-6, atol=1e-6)


def test_legacy_three_branch_checkpoint_loads_into_fixed_memory():
    torch.manual_seed(67)
    legacy = BoltStateMemory(
        input_dim=60,
        width=23,
        latent_dim=12,
        device="cpu",
        backend="pytorch",
    )
    legacy_state = legacy.state_dict()
    expected_weight = torch.cat((
        legacy_state["project.weight"][23:46],
        legacy_state["project.weight"][46:69],
    ), dim=0)
    expected_bias = torch.cat((
        legacy_state["project.bias"][23:46],
        legacy_state["project.bias"][46:69],
    ), dim=0)

    target = EnhancedMQA(
        d_model=60,
        num_heads=5,
        avq_dim=23,
        state_memory=True,
        backend="pytorch",
    ).avq
    result = target.load_state_dict(legacy_state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    torch.testing.assert_close(target.project.weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(target.project.bias, expected_bias, rtol=0, atol=0)


def test_benchmark_head_layout_assigns_equal_contiguous_channel_groups():
    layer = EnhancedMQA(
        d_model=384,
        num_heads=12,
        avq_dim=384,
        state_memory=True,
        backend="pytorch",
    )
    grouped = layer.avq._fixed_retention_coefficients.view(12, 32)
    for head_values in grouped:
        torch.testing.assert_close(
            head_values,
            head_values[:1].expand_as(head_values),
            rtol=0,
            atol=0,
        )
    assert torch.unique(grouped[:, 0]).numel() == 12
