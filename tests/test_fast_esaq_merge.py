"""Fast-EsaQ replacement regression suite; no CUDA needed."""
import math
import torch
import pytest

from mlbricks import ESA, ESAModel, ESAModelConfig
from mlbricks.esa.fast_engine import CurrentESA, BaseESA

@pytest.mark.parametrize("heads", [1, 2, 4, 6])
def test_mixer_runs_all_supported_heads_and_differentiates(heads):
    torch.manual_seed(10)
    m = ESA(embd=24, head=heads, backend="pytorch", precision="fp32", device="cpu")
    x = torch.randn(2, 7, 24, requires_grad=True)
    y = m(x)
    assert isinstance(m.layer, CurrentESA)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert m.layer.qgv.weight.grad is not None
    assert m.layer.log_half_life.grad is not None
    assert m.layer.raw_decay_bias.grad is not None

@pytest.mark.parametrize("variant", ["current", "base"])
def test_fast_esa_cached_continuation_matches_full(variant):
    torch.manual_seed(77)
    m = ESA(embd=24, head=6, backend="pytorch", esa_variant=variant, device="cpu").eval()
    assert (isinstance(m.layer, BaseESA) if variant == "base" else isinstance(m.layer, CurrentESA))
    seq = torch.randn(2, 10, 24)
    with torch.no_grad():
        full = m(seq)
        prefix, state = m.prefill(seq[:, :6])
        continued = []
        for i in range(6, 10):
            yy, state = m.decode_step(seq[:, i], state)
            continued.append(yy.unsqueeze(1))
        torch.testing.assert_close(torch.cat((prefix, torch.cat(continued, dim=1)), dim=1), full, rtol=1e-5, atol=1e-5)
        assert state.dtype == torch.float32

def test_state_layout_reverse_and_checkpoint_markers():
    m = ESA(embd=24, head=6, backend="pytorch", device="cpu").eval()
    x = torch.randn(1, 5, 24)
    y, st = m.forward_with_state(x, reverse=True)
    y2, flat_state = m.forward_with_state(x, state=m.init_state(1, layout="flat"), reverse=True)
    torch.testing.assert_close(y, y2)
    torch.testing.assert_close(st, flat_state.reshape(st.shape))
    assert 'layer.log_half_life' in m.state_dict()
    assert 'layer.out_proj.weight' not in m.state_dict()
    torch.testing.assert_close(m.layer.log_half_life, torch.tensor([16,64,256,1024,4096,8192], dtype=torch.float32).log())

def test_half_precision_storage_fp32_recurrence():
    m = ESA(embd=24, head=6, backend="pytorch", device="cpu").half()
    with torch.no_grad():
        out, state = m.prefill(torch.randn(2, 5, 24).half())
    assert out.dtype == torch.float16 and state.dtype == torch.float32
    assert m.layer.qgv.weight.dtype == torch.float16
    assert torch.isfinite(out).all() and torch.isfinite(state).all()

def test_single_fast_esa_no_architecture_switch_and_rejects_old_weights():
    import inspect
    from dataclasses import fields
    assert 'architecture' not in inspect.signature(ESA).parameters
    assert 'esa_architecture' not in {field.name for field in fields(ESAModelConfig)}
    with pytest.raises(TypeError, match='architecture'):
        ESA(embd=24, head=6, architecture='legacy')
    with pytest.raises(TypeError, match='esa_architecture'):
        ESAModelConfig(vocab_size=24, esa_architecture='legacy')
    with pytest.raises(ValueError, match='Fast-EsaQ only'):
        ESAModelConfig.from_dict({'vocab_size':24, 'esa_architecture':'legacy'})
    cfg = ESAModelConfig.from_dict({'vocab_size':24, 'embd':24, 'head':6, 'block':16})
    assert cfg.embd == 24

    fast = ESA(embd=24, head=6, backend='pytorch', device='cpu')
    with pytest.raises(ValueError, match='Old Thunder ESA weights'):
        fast.load_state_dict({'layer.out_proj.weight': torch.zeros(24,24)}, strict=False)
    model = ESAModel(ESAModelConfig(
        vocab_size=24, embd=24, head=6, block=16, n_layer=1,
        precision='fp32', training_compile=False,
    ), device='cpu')
    with pytest.raises(ValueError, match='Old Thunder ESA checkpoint'):
        model.load_state_dict({'blocks.0.esa.layer.out_proj.weight': torch.zeros(24,24)}, strict=False)

def test_fast_engine_and_mlbricks_wrapper_identical_mixer_weights_and_output():
    torch.manual_seed(42)
    wrapped = ESA(embd=24, head=6, backend="pytorch", device="cpu")
    original = CurrentESA(embd=24, head=6, backend="pytorch")
    original.load_state_dict(wrapped.layer.state_dict(), strict=True)
    x = torch.randn(2, 9, 24)
    with torch.no_grad():
        y0, st0 = original(x)
        y1, st1 = wrapped.prefill(x)
    torch.testing.assert_close(y0, y1, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(st0, st1)
