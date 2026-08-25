# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

import torch

from mlbricks import (
    ESA,
    ESAModel,
    ESAModelConfig,
    Trainer,
)
from mlbricks.backends.lightning import lightning_scan



def _reference_scan(A, B):
    state = torch.zeros_like(A[:, 0])
    rows = []
    for t in range(A.size(1)):
        state = A[:, t] * state + B[:, t]
        rows.append(state)
    return torch.stack(rows, dim=1)


def test_lightning_scan_matches_reference():
    torch.manual_seed(1)

    A = (
        0.8
        + 0.19
        * torch.rand(
            2,
            17,
            4,
            8,
        )
    )

    B = (
        torch.randn_like(A)
        * 0.1
    )

    ref = _reference_scan(A, B)

    got = lightning_scan(
        A,
        B,
        compass=4,
    )

    assert torch.allclose(
        got,
        ref,
        atol=1e-5,
        rtol=1e-5,
    )


def test_layer_prefill_and_step_match_forward_cpu_thunder():
    torch.manual_seed(1)

    layer = ESA(
        embd=32,
        head=4,
        backend="thunder",
        compass=1,
        precision="fp32",
        device=None,
    )

    layer.eval()

    x = torch.randn(
        2,
        7,
        32,
    )

    y_ref = layer(x)

    y_prefill, state = layer.prefill(
        x
    )

    assert torch.allclose(
        y_ref,
        y_prefill,
        atol=1e-6,
        rtol=1e-6,
    )

    x_next = torch.randn(
        2,
        32,
    )

    y_step, new_state = (
        layer.decode_step(
            x_next,
            state,
        )
    )

    assert y_step.shape == (
        2,
        32,
    )

    assert (
        new_state.shape
        == state.shape
    )


def test_model_save_load_roundtrip(
    tmp_path,
):
    cfg = ESAModelConfig(
        vocab_size=128,
        block=16,
        n_layer=2,
        head=4,
        embd=32,
        dropout=0.0,
        backend="thunder",
        compass=1,
        precision="fp32",
    )

    model = ESAModel(
    cfg,
    device="cpu",
    )

    model.eval()

    x = torch.randint(
        0,
        128,
        (
            2,
            8,
        ),
    )

    ref, _ = model(
        x
    )

    path = (
        tmp_path
        / "model"
    )

    model.save(
        path
    )

    loaded = ESAModel.load(
        path
    )

    loaded.eval()

    got, _ = loaded(
        x
    )

    assert torch.allclose(
        ref,
        got,
        atol=0,
        rtol=0,
    )


def test_exact_checkpoint_is_protected(
    tmp_path,
):
    cfg = ESAModelConfig(
        vocab_size=64,
        block=8,
        n_layer=1,
        head=2,
        embd=16,
        dropout=0.0,
        backend="thunder",
        compass=1,
        precision="fp32",
    )

    model = ESAModel(
        cfg
    )

    trainer = Trainer(
        model,
        checkpoint_dir=(
            tmp_path
            / "checkpoints"
        ),
        save_every=1,
        save_at=[2],
        keep_last_n=1,
    )

    trainer.maybe_save(
        step=1
    )

    trainer.maybe_save(
        step=2
    )

    trainer.maybe_save(
        step=3
    )

    assert (
        tmp_path
        / "checkpoints"
        / "step_000002"
    ).exists()

    assert (
        tmp_path
        / "checkpoints"
        / "step_000003"
    ).exists()
