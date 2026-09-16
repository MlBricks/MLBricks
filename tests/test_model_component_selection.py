import torch
import pytest

from mlbricks import (
    ESAModel,
    ESAModelConfig,
    Vesa,
    StateAwareFFN,
    VirtualStateAwareFFN,
    MicroVirtualFFN,
    ResController,

    save,
    load,)


def _esa_model(**overrides):
    kwargs = dict(
        vocab_size=64,
        block=8,
        n_layer=2,
        head=2,
        embd=16,
        dropout=0.0,
        precision="fp32",
        compass=1,
        device="cpu",
        ffn_state_dim=8,
        ffn_depth_embedding_dim=4,
        ffn_virtual_hidden_dim=8,
        ffn_micro_hidden_dim=8,
    )
    kwargs.update(overrides)
    return ESAModel(**kwargs)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ffnbrick", StateAwareFFN),
        ("virtual_ffnbrick", VirtualStateAwareFFN),
        ("micro_ffnbrick", MicroVirtualFFN),
    ],
)
def test_esa_model_selects_ffnbrick(name, expected):
    model = _esa_model(ffn=name)
    assert model.config.ffn == name
    assert all(isinstance(block.mlp, expected) for block in model.blocks)
    logits, loss = model(torch.randint(0, 64, (2, 6)), torch.randint(0, 64, (2, 6)))
    assert logits.shape == (2, 6, 64)
    assert loss is not None


def test_esa_model_selects_rescontroller():
    model = _esa_model(ffn="ffnbrick", residual="rescontroller")
    assert model.config.residual == "rescontroller"
    assert all(isinstance(block.esa_residual, ResController) for block in model.blocks)
    assert all(isinstance(block.ffn_residual, ResController) for block in model.blocks)

    ids = torch.randint(0, 64, (2, 6))
    logits, states, length = model.lightning_prefill(ids)
    assert logits.shape == (2, 64)
    assert states.shape[0] == 2
    assert length == 6

    step_logits, next_states = model.lightning_step(
        torch.randint(0, 64, (2,)),
        states,
        torch.tensor(6),
    )
    assert step_logits.shape == (2, 64)
    assert next_states.shape == states.shape


def test_esamodel_default_init_std_is_point_zero_two():
    model = _esa_model(n_layer=1)
    assert model.config.init_std == 0.02
    measured = float(model.wte.weight.detach().float().std())
    assert 0.012 < measured < 0.028
    assert model.wte.weight is model.lm_head.weight


def test_component_aliases_are_normalized():
    cfg = ESAModelConfig(vocab_size=64, ffn="state_aware", residual="ResController")
    assert cfg.ffn == "ffnbrick"
    assert cfg.residual == "rescontroller"


def test_invalid_model_component_names_are_rejected():
    with pytest.raises(ValueError, match="ffn must be"):
        ESAModelConfig(vocab_size=64, ffn="unknown")
    with pytest.raises(ValueError, match="residual must be"):
        ESAModelConfig(vocab_size=64, residual="unknown")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ffnbrick", StateAwareFFN),
        ("virtual_ffnbrick", VirtualStateAwareFFN),
        ("micro_ffnbrick", MicroVirtualFFN),
    ],
)
def test_vesa_selects_ffnbrick(name, expected):
    model = Vesa(
        image_size=8,
        patch_size=4,
        num_classes=3,
        dim=16,
        depth=2,
        perspective_groups=4,
        backend="thunder",
        ffn=name,
        ffn_state_dim=8,
        ffn_depth_embedding_dim=4,
        ffn_virtual_hidden_dim=8,
        ffn_micro_hidden_dim=8,
    )
    assert model.config.ffn == name
    assert all(isinstance(block.mlp, expected) for block in model.blocks)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 3)


def test_vesa_selects_rescontroller():
    model = Vesa(
        image_size=8,
        patch_size=4,
        num_classes=3,
        dim=16,
        depth=2,
        perspective_groups=4,
        backend="thunder",
        ffn="ffnbrick",
        residual="rescontroller",
        ffn_state_dim=8,
        ffn_depth_embedding_dim=4,
    )
    assert model.config.residual == "rescontroller"
    assert all(isinstance(block.local_residual, ResController) for block in model.blocks)
    assert all(isinstance(block.esa_residual, ResController) for block in model.blocks)
    assert all(isinstance(block.ffn_residual, ResController) for block in model.blocks)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 3)


def test_ffnbrick_rescontroller_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = _esa_model(ffn="ffnbrick", residual="rescontroller")
    model.eval()
    ids = torch.randint(0, 64, (2, 6))
    reference, _ = model(ids)
    path = tmp_path / "component-model"
    save(model, path)
    loaded = load(path, device="cpu")
    loaded.eval()
    result, _ = loaded(ids)
    assert loaded.config.ffn == "ffnbrick"
    assert loaded.config.residual == "rescontroller"
    assert torch.allclose(reference, result, atol=0, rtol=0)


def test_virtual_ffnbrick_keeps_zero_update_initialization():
    model = _esa_model(ffn="virtual_ffnbrick")
    for block in model.blocks:
        assert torch.count_nonzero(block.mlp.virtual_refiner.down.weight) == 0
        assert torch.count_nonzero(block.mlp.virtual_refiner.down.bias) == 0


def test_micro_ffnbrick_keeps_zero_update_initialization():
    model = _esa_model(ffn="micro_ffnbrick")
    for block in model.blocks:
        assert torch.count_nonzero(block.mlp.down) == 0
