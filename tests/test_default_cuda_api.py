"""Minimal public API and safe automatic inference routing regression tests."""
import torch
import pytest
from mlbricks import BoltModel, Bolt, Attention


def test_minimal_api_cpu_train_and_infer(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '0')
    model = BoltModel(vocab_size=127, layers=2, dim=96, heads=6,
                      context=64, bolt_cache='tight', avq_dim=30)
    assert next(model.parameters()).device.type == 'cpu'
    ids = torch.randint(0, 127, (1, 12))
    model.train()
    logits, loss = model(ids, targets=ids)
    assert logits.shape == (1, 12, 127)
    loss.backward()
    assert next(model.parameters()).grad is not None
    before = tuple(model.state_dict())
    model.eval()
    result = model.generate(ids, 5, temperature=0.0)
    assert result.shape == (1, 17)
    assert model.inference_info()['last_execution'] == 'pytorch_eager'
    assert before == tuple(model.state_dict())


def test_single_bolt_and_qkv_are_standalone(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '0')
    hidden = torch.randn(2, 8, 96)
    bolt = Bolt(d_model=96, num_heads=6, cache='tight').eval()
    attention = Attention(d_model=96, num_heads=6).eval()
    assert bolt(hidden).shape == hidden.shape
    assert attention(hidden).shape == hidden.shape


def test_sampling_and_unverified_gpu_path_do_not_capture(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '0')
    model = BoltModel(vocab_size=127, layers=1, dim=96, heads=6,
                      context=64, bolt_cache='tight', avq_dim=30).eval()
    ids = torch.randint(0, 127, (1, 8))
    assert not model._auto_strict_graph_eligible(ids, 4, 0.0, None)
    _ = model.generate(ids, 2, temperature=1.0)
    assert model.inference_info()['last_execution'] == 'pytorch_eager'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_first_constructors_and_input_alignment(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '1')
    bolt = Bolt(d_model=96, num_heads=6, cache='tight')
    attention = Attention(d_model=96, num_heads=6)
    assert next(bolt.parameters()).is_cuda
    assert next(attention.parameters()).is_cuda
    cpu_x = torch.randn(1, 4, 96)
    assert bolt(cpu_x).is_cuda
    assert attention(cpu_x).is_cuda


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7,5),
                    reason='T4 required for strict-graph integration gate')
def test_bolt_graph_automatically_selected_on_fp16_t4(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '1')
    torch.manual_seed(2026)
    model = BoltModel(vocab_size=127, layers=2, dim=96, heads=6,
                      context=64, bolt_cache='tight', avq_dim=30).cuda().half().eval()
    prompt = torch.randint(0,127,(1,12), device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        # Independent eager baseline, preserving original weights.
        import copy
        reference = copy.deepcopy(model)
        eager = reference.generate(prompt, 8, temperature=0)
        automatic = model.generate(prompt, 8, temperature=0)
    assert torch.equal(eager, automatic)
    assert model.inference_info()['last_execution'] == 'strict_cuda_graph'
