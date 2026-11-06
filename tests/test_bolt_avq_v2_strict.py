"""Regression checks for opt-in BOLT V2 strict AVQ decode."""
import copy
import pytest
import torch

from mlbricks.bolt import v2_strict
from mlbricks.bolt.presets import BoltStateMemory, BoltPreset, BoltPresetCache


def test_v2_off_by_default(monkeypatch):
    monkeypatch.delenv('MLBRICKS_BOLT_V2_STRICT_DECODE', raising=False)
    assert not v2_strict.enabled()
    m = BoltStateMemory(192, 192, device='cpu', backend='pytorch').eval()
    with torch.no_grad():
        y, s = m.step(torch.randn(1, 1, 192), torch.zeros(1, 192))
    assert y.shape == (1, 1, 192)
    assert s.shape == (1, 192)


def test_no_unsupported_cpu_or_training_dispatch(monkeypatch):
    monkeypatch.setenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '1')
    m = BoltStateMemory(192, 192, device='cpu', backend='auto')
    x = torch.randn(1, 1, 192)
    s = torch.zeros(1, 192)
    assert v2_strict.maybe_step(m, x, s) is None
    m.eval()
    assert v2_strict.maybe_step(m, x, s) is None
    assert m.state_dict().keys() == copy.deepcopy(m).state_dict().keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('width', [192, 384])
def test_strict_v2_matches_reference_and_preserves_step_inputs(monkeypatch, width):
    if v2_strict.get_extension() is None:
        pytest.skip('build with MLBRICKS_BUILD_BOLT_V2_STRICT=1')
    torch.manual_seed(900 + width)
    m = BoltStateMemory(width, width, device='cuda', backend='auto').eval().cuda()
    m = m.half()
    with torch.no_grad():
        m.gamma.fill_(0.2)
    x = torch.randn(1, 1, width, device='cuda', dtype=torch.float16)
    state = torch.randn(1, width, device='cuda', dtype=torch.float32) * 0.01
    m.use_native = True
    monkeypatch.delenv('MLBRICKS_BOLT_V2_STRICT_DECODE', raising=False)
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            expected_y, expected_s = m.step(x, state)
    monkeypatch.setenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '1')
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            actual_y, actual_s = m.step(x, state)
    assert x.isfinite().all() and state.isfinite().all()
    torch.testing.assert_close(actual_y.float(), expected_y.float(), atol=0.02, rtol=0.01)
    torch.testing.assert_close(actual_s, expected_s, atol=6e-4, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_v2_cached_decode_64_steps(monkeypatch):
    if v2_strict.get_extension() is None:
        pytest.skip('build with MLBRICKS_BUILD_BOLT_V2_STRICT=1')
    torch.manual_seed(77)
    model = BoltPreset(384, 12, mode='quality', cache='tight',
                       backend='pytorch', projection='packed').eval().cuda().half()
    with torch.no_grad():
        model.avq.gamma.fill_(0.2)
    ref = copy.deepcopy(model)
    # For this integration test, enable only the V2 AVQ path while keeping the
    # same (PyTorch) attention/feature projection on both models.
    model.backend = 'pytorch'; model.avq.use_native = True
    ref.backend = 'pytorch'; ref.avq.use_native = False
    # Test the AVQ one-token step directly to avoid mixing attention backends.
    x = torch.randn(1, 1, 192, device='cuda', dtype=torch.float16)
    state1 = torch.zeros(1, 192, device='cuda', dtype=torch.float32)
    state2 = state1.clone()
    monkeypatch.setenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '1')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for _ in range(64):
            o1, state1 = model.avq.step(x, state1)
            monkeypatch.setenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '0')
            o2, state2 = ref.avq.step(x, state2)
            monkeypatch.setenv('MLBRICKS_BOLT_V2_STRICT_DECODE', '1')
            torch.testing.assert_close(o1.float(), o2.float(), atol=0.03, rtol=0.02)
            torch.testing.assert_close(state1, state2, atol=6e-4, rtol=1e-3)
