from __future__ import annotations

import json

import torch

from mlbricks import ESA, ElasticBit, ElasticEmbedding, ElasticLinear, ESAModel, ESAModelConfig, thunderBoost
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


def test_elasticbit_init_state_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    ElasticBit(bits=4, group_size=8, scale_dtype=torch.float32).quantize_module(model)

    assert isinstance(model.blocks[0].esa.layer.qgv, ElasticLinear)
    state = model.blocks[0].esa.init_state(2)
    assert state.shape == (2, 2, 4)

    ids = torch.randint(0, 32, (1, 5))
    reference, _ = model(ids)
    path = tmp_path / "packed"
    model.save(path)

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["quantization"]["type"] == "elasticbit"

    loaded = ESAModel.load(path, device="cpu").eval()
    assert isinstance(loaded.blocks[0].esa.layer.qgv, ElasticLinear)
    output, _ = loaded(ids)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


def test_elasticbit_caches_materialized_weight():
    layer = torch.nn.Linear(8, 4, bias=False)
    packed = ElasticBit(bits=4, group_size=8).linear(layer)
    x = torch.randn(2, 8)
    packed(x)
    assert len(packed._dequant_cache) == 1
    first = next(iter(packed._dequant_cache.values()))
    packed(x)
    second = next(iter(packed._dequant_cache.values()))
    assert first.data_ptr() == second.data_ptr()


def test_elasticbit_preserves_tied_storage_when_embeddings_are_included():
    model = _tiny_model().eval()
    ElasticBit(bits=4, group_size=8).quantize_module(model, include_embeddings=True)
    assert isinstance(model.wte, ElasticEmbedding)
    assert isinstance(model.lm_head, ElasticLinear)
    assert model.wte.packed_weight.data_ptr() == model.lm_head.packed_weight.data_ptr()
    assert model.wte.scales.data_ptr() == model.lm_head.scales.data_ptr()


def test_elasticbit_does_not_duplicate_tied_lm_head_when_embeddings_excluded():
    model = _tiny_model().eval()
    ElasticBit(bits=4, group_size=8).quantize_module(model, include_embeddings=False)
    assert not isinstance(model.wte, ElasticEmbedding)
    assert not isinstance(model.lm_head, ElasticLinear)


def test_batch_eos_is_persistent(monkeypatch):
    model = _tiny_model().eval()
    ids = torch.randint(0, 32, (2, 3))
    samples = [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[2], [0]], dtype=torch.long),
    ]

    def fake_sample(*args, **kwargs):
        return samples.pop(0).to(model.device)

    monkeypatch.setattr("mlbricks.model.sample_next_token", fake_sample)
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


def test_elasticbit_tied_storage_survives_checkpoint_load(tmp_path):
    model = _tiny_model().eval()
    ElasticBit(bits=4, group_size=8).quantize_module(model, include_embeddings=True)
    path = tmp_path / "packed_tied"
    model.save(path)
    loaded = ESAModel.load(path, device="cpu").eval()
    assert isinstance(loaded.wte, ElasticEmbedding)
    assert isinstance(loaded.lm_head, ElasticLinear)
    assert loaded.wte.packed_weight.data_ptr() == loaded.lm_head.packed_weight.data_ptr()
    ids = torch.randint(0, 32, (1, 3))
    loaded(ids)
    assert loaded.wte._dequant_cache is loaded.lm_head._dequant_cache


def test_elasticbit_bitstream_roundtrip_for_all_supported_widths():
    from mlbricks.elasticbit import _pack_unsigned, _unpack_unsigned

    torch.manual_seed(4)
    for bits in range(2, 9):
        values = torch.randint(0, 1 << bits, (257,), dtype=torch.int16)
        restored = _unpack_unsigned(_pack_unsigned(values, bits), values.numel(), bits)
        assert torch.equal(values, restored)
