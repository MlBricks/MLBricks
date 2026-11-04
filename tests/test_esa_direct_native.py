"""Direct Fast-EsaQ implementation: no decode adapter on public import."""
import inspect
import pytest
import torch
from mlbricks import ESA, ESAAdapter, ESAModel, ESAModelConfig, Brick, Bricks
from mlbricks.esa.fast_engine.public_api import ESA as UpstreamESA
from mlbricks.esa.fast_engine.public_api import CurrentESA, BaseESA


def test_public_is_original_class():
    assert ESA is UpstreamESA
    assert not hasattr(ESA, "_fast_run")
    assert "variant" in inspect.signature(ESA).parameters


@pytest.mark.parametrize("variant,concrete", [("current", CurrentESA), ("base", BaseESA)])
def test_native_output_and_state(variant, concrete):
    torch.manual_seed(2026)
    native = ESA(embd=24, head=6, variant=variant, backend="pytorch", device="cpu").eval()
    assert isinstance(native, concrete)
    assert not hasattr(native, "layer")
    x = torch.randn(2, 7, 24)
    with torch.inference_mode():
        y, state = native(x)
        assert y.shape == x.shape
        assert state.shape == (2, 6, 4) and state.dtype == torch.float32
        step, next_state = native.decode_step(x[:, 0], state)
        assert step.shape == (2, 1, 24)
        torch.testing.assert_close(step, native(x[:, :1], state)[0], atol=0, rtol=0)
    keys = native.state_dict()
    assert "qgv.weight" in keys and not any(k.startswith("layer.") for k in keys)


@pytest.mark.parametrize("variant", ["current", "base"])
def test_native_vs_same_engine_parity(variant):
    torch.manual_seed(87)
    native = ESA(embd=24, head=6, backend="pytorch", variant=variant, device="cpu").eval()
    adapter = ESAAdapter(embd=24, head=6, backend="pytorch", esa_variant=variant, device="cpu").eval()
    adapter.layer.load_state_dict(native.state_dict())
    seq = torch.randn(2, 5, 24)
    state = torch.zeros(2, 6, 4, dtype=torch.float32)
    with torch.inference_mode():
        native_y, native_state = native.decode_step(seq[:, 0], state)
        adapter_y, adapter_state = adapter.decode_step(seq[:, 0], state)
    torch.testing.assert_close(native_y[:, 0], adapter_y, atol=0, rtol=0)
    torch.testing.assert_close(native_state, adapter_state, atol=0, rtol=0)


def test_base_and_half_life_keys_distinct():
    c = ESA(embd=24, head=6, backend="pytorch", variant="current", device="cpu")
    b = ESA(embd=24, head=6, backend="pytorch", variant="base", device="cpu")
    assert "log_half_life" in c.state_dict() and "raw_alpha" not in c.state_dict()
    assert "raw_alpha" in b.state_dict() and "log_half_life" not in b.state_dict()


def test_existing_model_still_uses_compat_without_checkpoint_change():
    cfg = ESAModelConfig(vocab_size=64, block=16, n_layer=2, head=6, embd=24,
                         dropout=0.0, precision="fp32", training_compile=False)
    model = ESAModel(cfg, device="cpu")
    assert isinstance(model.blocks[0].esa, ESAAdapter)
    tokens = torch.randint(0, 64, (2, 5))
    y, loss = model(tokens, targets=tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].esa.layer.qgv.weight.grad is not None


def test_bricks_support_native_direct_mixer():
    mixer = ESA(embd=24, head=6, backend="pytorch", device="cpu")
    model = Bricks(vocab_size=64, dim=24, context=16,
                   layers=[Brick(mixer=mixer)], backend="pytorch").eval()
    ids = torch.randint(0, 64, (2, 4))
    with torch.inference_mode():
        logits = model(ids)
    assert logits.shape == (2, 4, 64)
