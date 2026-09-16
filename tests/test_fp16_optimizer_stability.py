from __future__ import annotations

import warnings

import torch

from mlbricks import Adam, AdamW, StateAwareFFN, Trainer
from mlbricks.optim import FP16_ADAM_MIN_EPS, stabilize_optimizer


def _half_stateaware() -> StateAwareFFN:
    torch.manual_seed(91)
    return StateAwareFFN(
        d_model=32,
        state_dim=8,
        depth_embedding_dim=4,
        layer_index=0,
        total_layers=2,
        use_native=False,
    ).to(dtype=torch.float16).train()


def _batch():
    torch.manual_seed(92)
    x = torch.randn(2, 5, 32, dtype=torch.float16)
    esa = torch.randn_like(x)
    prev = torch.randn_like(x)
    state = torch.randn(2, 5, 8, dtype=torch.float16)
    return x, esa, prev, state


def _step(layer, optimizer):
    x, esa, prev, state = _batch()
    optimizer.zero_grad(set_to_none=True)
    update, next_state = layer(x, esa, prev, state)
    loss = update.float().square().mean() + next_state.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all()
        for p in layer.parameters()
        if p.grad is not None
    )
    optimizer.step()
    return loss.detach()


def test_mlbricks_adamw_raises_epsilon_only_for_fp16_groups():
    fp16 = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    fp32 = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = AdamW([
        {"params": [fp16]},
        {"params": [fp32]},
    ], lr=1e-4)
    assert opt.param_groups[0]["eps"] == FP16_ADAM_MIN_EPS
    assert opt.param_groups[1]["eps"] == 1e-8


def test_mlbricks_adam_uses_fp16_safe_epsilon():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    opt = Adam([p], lr=1e-4)
    assert opt.param_groups[0]["eps"] == FP16_ADAM_MIN_EPS


def test_stateaware_fp16_adamw_stays_finite_for_multiple_steps():
    layer = _half_stateaware()
    opt = AdamW(layer.parameters(), lr=1e-4)
    losses = []
    for _ in range(20):
        losses.append(float(_step(layer, opt)))
    assert all(torch.isfinite(p).all() for p in layer.parameters())
    assert all(torch.isfinite(torch.tensor(losses)))


def test_stabilize_external_torch_adamw():
    layer = _half_stateaware()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    assert opt.param_groups[0]["eps"] == 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        changed = stabilize_optimizer(opt)
    assert changed is True
    assert opt.param_groups[0]["eps"] == FP16_ADAM_MIN_EPS
    _step(layer, opt)
    assert all(torch.isfinite(p).all() for p in layer.parameters())


def test_stabilize_does_not_touch_fp32_adamw():
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = torch.optim.AdamW([p], lr=1e-4)
    assert stabilize_optimizer(opt, warn=False) is False
    assert opt.param_groups[0]["eps"] == 1e-8


def test_trainer_stabilizes_external_adamw_without_needing_fit_loop(tmp_path):
    # The constructor does not require an ESAModel-specific method until a
    # checkpoint operation is requested, so a small FFNBrick module is enough
    # to verify the optimizer policy itself.
    layer = _half_stateaware()
    opt = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(layer, optimizer=opt, checkpoint_dir=tmp_path)
    assert trainer.optimizer is opt
    assert opt.param_groups[0]["eps"] == FP16_ADAM_MIN_EPS
    _step(layer, opt)
    assert all(torch.isfinite(p).all() for p in layer.parameters())
