from __future__ import annotations

import json

import torch

from mlbricks import ESA, ESAModel, ESAModelConfig, thunderBoost
from mlbricks.planner import sequence_bucket


def _tiny_model() -> ESAModel:
    return ESAModel(
        ESAModelConfig(
            vocab_size=32,
            block=16,
            n_layer=1,
            head=2,
            embd=8,
            dropout=0.0,
            precision="fp32",
            training_compile=False,
        ),
        device="cpu",
    )


def test_prefill_compiled_alias_uses_direct_native_path(monkeypatch):
    model = _tiny_model().eval()
    ids = torch.randint(0, 32, (1, 4))

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("prefill must not call torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden_compile)

    # Legacy 1.0.x keyword arguments and engine aliases remain accepted, but
    # all prefill execution is direct backend execution.
    alias = model.prefill(
        ids,
        engine="thunder_compiled_1",
        compile_mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    native = model.prefill(ids, engine="thunder_1")

    torch.testing.assert_close(alias[0], native[0], atol=0, rtol=0)
    torch.testing.assert_close(alias[1], native[1], atol=0, rtol=0)
    assert alias[2] == native[2]
    assert not hasattr(model, "_compiled_prefill_cache")
    assert not hasattr(model, "_prefill_compile_failures")


def test_batch_eos_is_persistent(monkeypatch):
    model = _tiny_model().eval()
    ids = torch.randint(0, 32, (2, 3))
    samples = [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[2], [0]], dtype=torch.long),
    ]

    def fake_sample(*args, **kwargs):
        return samples.pop(0).to(model.device)

    import importlib
    esa_model_module = importlib.import_module("mlbricks.esa.model")
    monkeypatch.setattr(esa_model_module, "sample_next_token", fake_sample)
    result = model.generate_ids(
        ids,
        seek=5,
        compile=False,
        temperature=0.0,
        eos_token_id=0,
        stats=True,
    )
    assert result.generated_ids.shape == (2, 2)
    assert result.generated_ids.tolist() == [[0, 0], [1, 0]]
    assert samples == []


def test_auto_compute_dtype_stays_fp32_on_cpu():
    model = ESAModel(ESAModelConfig(vocab_size=32, embd=8, head=2, n_layer=1), device="cpu")
    assert model.compute_dtype == torch.float32


def test_compass_sequence_lengths_are_bucketed():
    assert sequence_bucket(487) == sequence_bucket(493) == sequence_bucket(510) == 512
    assert sequence_bucket(513) == 1024


def test_thunderboost_restores_eval_mode():
    layer = ESA(
        embd=8,
        head=2,
        batch=1,
        block=2,
        precision="fp32",
        device="cpu",
    ).eval()
    boosted = thunderBoost(
        layer,
        compile=False,
        backward=False,
        amp=False,
        steps=1,
        device="cpu",
    )
    assert boosted.training is False


