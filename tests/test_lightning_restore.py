from __future__ import annotations

import inspect
import os

import torch

from mlbricks import ESAModel, ESAModelConfig


def _tiny() -> ESAModel:
    return ESAModel(
        ESAModelConfig(
            vocab_size=32,
            block=16,
            n_layer=2,
            head=2,
            embd=8,
            dropout=0.0,
            precision="fp32",
            training_compile=False,
            ffn="standard",
            residual="standard",
        ),
        device="cpu",
    ).eval()


def test_generation_defaults_match_proven_public_lightning_path() -> None:
    generate = inspect.signature(ESAModel.generate)
    compile_generation = inspect.signature(ESAModel.compile_generation)
    assert generate.parameters["compile"].default is True
    assert generate.parameters["compile_mode"].default == "default"
    assert compile_generation.parameters["mode"].default == "default"


def test_standard_block_fast_step_matches_general_step() -> None:
    torch.manual_seed(0)
    model = _tiny()
    block = model.blocks[0]
    x = torch.randn(2, 8)
    state = block.esa.init_state(2)

    y_fast, state_fast = block.lightning_step_standard(x, state)
    y_general, state_general, _, _ = block.lightning_step(x, state)

    torch.testing.assert_close(y_fast, y_general, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state_fast, state_general, atol=1e-6, rtol=1e-6)


def test_model_standard_lightning_uses_fast_block_path(monkeypatch) -> None:
    model = _tiny()
    token = torch.tensor([1], dtype=torch.long)
    pos = torch.tensor(0, dtype=torch.long)
    states = torch.stack([block.esa.init_state(1) for block in model.blocks], dim=0)

    called = [0 for _ in model.blocks]
    for i, block in enumerate(model.blocks):
        original = block.lightning_step_standard

        def wrapped(x, state, *, _i=i, _original=original):
            called[_i] += 1
            return _original(x, state)

        monkeypatch.setattr(block, "lightning_step_standard", wrapped)

    logits, states_out = model.lightning_step(token, states, pos)
    assert logits.shape == (1, 32)
    assert states_out.shape == states.shape
    assert called == [1, 1]


def test_fused_lightning_is_opt_in_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MLBRICKS_LIGHTNING_FUSED", raising=False)
    assert os.getenv("MLBRICKS_LIGHTNING_FUSED", "0") == "0"
