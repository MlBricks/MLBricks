import inspect
import tempfile

import pytest
import torch
from torch import nn

from mlbricks import (ESA, LegacyESA, RetentionESA, CurrentESA,
                       DEFAULT_CURRENT_HALF_LIVES, make_esa,
                       save_current_esa_checkpoint, load_current_esa_checkpoint,
                       ESAModel, ESAModelConfig)


def test_default_mode_uses_retention_with_direct_half_lives():
    assert inspect.signature(ESA).parameters['mode'].default == 'retention'
    current = ESA(embd=24, head=6, backend='pytorch', device='cpu')
    assert isinstance(current, RetentionESA)
    assert isinstance(current, CurrentESA)
    assert isinstance(current, ESA)
    assert isinstance(current, nn.Module)
    torch.testing.assert_close(current.log_half_life.detach().exp(),
        torch.tensor(DEFAULT_CURRENT_HALF_LIVES), rtol=1e-6, atol=1e-6)
    assert current.log_half_life.dtype == torch.float32
    assert current.log_half_life.requires_grad
    # Original engine's key paths (no wrapper field in the state dict).
    assert 'log_half_life' in current.state_dict()
    assert not any(k.startswith(('impl.', 'engine.')) for k in current.state_dict())


def test_explicit_retention_and_default_are_identical_at_same_seed():
    torch.manual_seed(173)
    a = ESA(embd=24, head=6, backend='pytorch', device='cpu')
    torch.manual_seed(173)
    b = ESA(embd=24, head=6, mode='retention', backend='pytorch', device='cpu')
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
    x = torch.randn(2, 7, 24, requires_grad=True)
    ay, as_ = a(x)
    by, bs = b(x)
    torch.testing.assert_close(ay, by, atol=0, rtol=0)
    torch.testing.assert_close(as_, bs, atol=0, rtol=0)
    (ay.square().mean() + as_.square().mean()).backward()
    assert a.log_half_life.grad is not None
    assert torch.isfinite(a.log_half_life.grad).all()


def test_legacy_explicit_retains_old_class_and_state_dict():
    torch.manual_seed(92)
    old = ESA(embd=24, head=4, mode='legacy', backend='pytorch',
              precision='fp32', device='cpu')
    torch.manual_seed(92)
    historical = LegacyESA(embd=24, head=4, backend='pytorch',
                           precision='fp32', device='cpu')
    assert isinstance(old, LegacyESA) and isinstance(old, ESA)
    assert not isinstance(old, RetentionESA)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, historical.state_dict()[key], atol=0, rtol=0)
    x = torch.randn(2, 7, 24)
    torch.testing.assert_close(old(x), historical(x), atol=0, rtol=0)
    assert isinstance(old(x), torch.Tensor)
    assert 'log_half_life' not in old.state_dict()


def test_mode_dependent_default_heads_and_invalid_modes():
    assert ESA(embd=24, device='cpu', backend='pytorch').head == 6
    assert ESA(embd=24, mode='legacy', device='cpu', backend='pytorch').head == 4
    for mode in ('fast', '', 'base', 123, None):
        with pytest.raises(ValueError, match='mode'):
            ESA(embd=24, mode=mode, backend='pytorch', device='cpu')
    with pytest.raises(TypeError):
        ESA(embd=24, head=6, backend='pytorch', compass='auto')


def test_public_make_esa_factory_remains_compatible():
    assert isinstance(make_esa(architecture='retention', embd=24, head=6, device='cpu'), RetentionESA)
    assert isinstance(make_esa(architecture='current', embd=24, head=6, device='cpu'), RetentionESA)
    assert isinstance(make_esa(architecture='legacy', embd=24, head=4, device='cpu'), LegacyESA)


def test_mode_specific_forward_and_decode_contract():
    m = ESA(embd=24, head=6, backend='pytorch', device='cpu').eval()
    x = torch.randn(2, 8, 24)
    y, state = m.prefill(x)
    next_y, final_state = m.decode_step(torch.randn(2, 24), state)
    assert y.shape == (2, 8, 24)
    assert state.shape == final_state.shape == (2, 6, 4)
    assert next_y.shape == (2, 1, 24)
    legacy = ESA(embd=24, head=4, mode='legacy', precision='fp32', device='cpu')
    assert legacy(x).shape == (2, 8, 24)


def test_retention_checkpoint_roundtrip_unchanged():
    m = ESA(embd=24, head=6, device='cpu', backend='pytorch')
    with tempfile.TemporaryDirectory() as td:
        path = td + '/esa.pt'
        save_current_esa_checkpoint(m, path)
        other = load_current_esa_checkpoint(path, backend='pytorch', device='cpu')
        assert isinstance(other, RetentionESA)
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, other.state_dict()[k], atol=0, rtol=0)


def test_existing_esa_language_model_keeps_legacy_engine():
    config = ESAModelConfig(vocab_size=64, embd=24, head=4, n_layer=1)
    lm = ESAModel(config, device='cpu')
    assert isinstance(lm.blocks[0].esa, LegacyESA)


def test_bricks_can_recurrently_use_retention_esa():
    from mlbricks import Brick
    mixer = ESA(embd=24, head=6, backend="pytorch", device="cpu")
    block = Brick(mixer=mixer, dim=24)
    block.eval()
    x = torch.randn(2, 6, 24)
    with torch.no_grad():
        full, _, _ = block(x)
        prefill, state, _, _ = block.prefill(x)
    torch.testing.assert_close(full, prefill, atol=1e-5, rtol=1e-5)
    assert state[0] == 'esa'
    assert state[1].shape == (2, 6, 4)
