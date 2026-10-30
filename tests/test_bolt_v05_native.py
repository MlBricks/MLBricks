"""BOLT v0.5 opt-in regression tests.

CPU tests exercise the real MLBricks public API; optional GPU tests compile
native kernels only when explicitly enabled and a CUDA GPU is available.

Kaggle GPU invocation:
    MLBRICKS_BOLT_V05_ALLOW_JIT=1 pytest -q tests/test_bolt_v05_native.py
"""
import copy
import os

import pytest
import torch

from mlbricks import Bolt, BoltModel, BoltPresetCache


def _model(*, layers=2, cache='tight', mode='quality', context=48):
    return BoltModel(vocab_size=127, dim=48, heads=3, layers=layers, context=context,
                     bolt_mode=mode, bolt_cache=cache, position='none', backend='pytorch',
                     dropout=0).eval()


def _stimulate(m):
    with torch.no_grad():
        for block in m.blocks:
            preset = block.attn._modern
            preset.p.uniform_(.63, 1.4)
            if preset.avq is not None:
                preset.avq.gamma.fill_(.21)
                preset.avq.norm.weight.uniform_(.75, 1.25)


def test_v05_model_api_opt_in_cpu_same_logits_tokens_and_keys():
    torch.manual_seed(5501)
    m = _model()
    _stimulate(m)
    original = copy.deepcopy(m)
    keys = tuple(m.state_dict())
    m.enable_bolt_optimized_inference(True)
    m.enable_bolt_v05_native(gate=True, avq=True)
    ids = torch.randint(0, 127, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(original(ids), m(ids), rtol=3e-5, atol=3e-5)
        assert torch.equal(original.generate(ids, 4, temperature=0),
                           m.generate(ids, 4, temperature=0))
    assert tuple(m.state_dict()) == keys
    assert all(block.attn._modern.v05_native_gate for block in m.blocks)
    assert all(block.attn._modern.avq.v05_native_avq for block in m.blocks)
    m.enable_bolt_v05_native(gate=False, avq=False)
    assert not any(block.attn._modern.v05_native_gate for block in m.blocks)
    assert not any(block.attn._modern.avq.v05_native_avq for block in m.blocks)


def test_v05_requires_opt_in_and_pytorch_backend():
    b = Bolt(48, 3, mode='quality', cache='tight', backend='pytorch').eval()
    with pytest.raises(ValueError, match='enable_optimized_inference'):
        b.enable_v05_native_fusions()
    b.enable_optimized_inference(True)
    b.enable_v05_native_fusions()
    b.set_backend('native')
    with pytest.raises(ValueError, match='backend="pytorch"'):
        b.enable_v05_native_fusions()


def test_v05_fast_gate_only_cpu_and_reject_avq():
    torch.manual_seed(5502)
    b = Bolt(48, 3, mode='fast', cache='normal', backend='pytorch').eval()
    b.enable_optimized_inference(True)
    with pytest.raises(ValueError, match='quality'):
        b.enable_v05_native_fusions(gate=True, avq=True)
    b.enable_v05_native_fusions(gate=True, avq=False)
    assert b._modern.v05_native_gate
    x = torch.randn(2, 4, 48)
    with torch.no_grad():
        ref = Bolt(48, 3, mode='fast', cache='normal', backend='pytorch').eval()
        ref.load_state_dict(b.state_dict())
        torch.testing.assert_close(b(x), ref(x), atol=1e-5, rtol=1e-5)


def test_v05_opt_in_training_keeps_existing_gradients():
    torch.manual_seed(5503)
    base = Bolt(48, 3, mode='quality', cache='tight', backend='pytorch').train()
    base._modern.avq.gamma.data.fill_(.21)
    candidate = copy.deepcopy(base).eval().enable_optimized_inference(True)
    candidate.enable_v05_native_fusions()
    candidate.train()
    x = torch.randn(2, 5, 48)
    base(x).square().mean().backward()
    candidate(x).square().mean().backward()
    for (an, ap), (bn, bp) in zip(base.named_parameters(), candidate.named_parameters()):
        assert an == bn
        torch.testing.assert_close(ap.grad, bp.grad, atol=1e-7, rtol=1e-6)


def test_v05_cpu_fixed_cache_matches_reference():
    torch.manual_seed(5504)
    b = Bolt(48, 3, mode='quality', cache='tight', backend='pytorch').eval()
    b.enable_optimized_inference(True).enable_v05_native_fusions()
    x = torch.randn(1, 9, 48)
    with torch.inference_mode():
        _, prefix = b.prefill(x[:, :5])
        c1 = BoltPresetCache.allocate(1, 3, 10, 16, 198, device='cpu', dtype=torch.float32).load_prefix(*prefix)
        c2 = BoltPresetCache.allocate(1, 3, 10, 16, 198, device='cpu', dtype=torch.float32).load_prefix(*prefix)
        for pos in range(5, 9):
            eager = b._modern.decode_with_cache(x[:, pos:pos+1], c1, position=pos)
            graph_cpu = b._modern.graph_decode_with_cache(
                x[:, pos:pos+1], c2, position=torch.tensor(pos), positions=torch.arange(10))
            torch.testing.assert_close(eager, graph_cpu, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(c1.state, c2.state, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
@pytest.mark.parametrize('cache', ['tight', 'normal'])
@pytest.mark.parametrize('seed', [2026, 5505])
def test_v05_native_closed_loop_eager_and_graph_parity(cache, seed):
    """Run on Kaggle. Reference and graph MUST use independent model instances."""
    if os.environ.get('MLBRICKS_BOLT_V05_ALLOW_JIT') != '1':
        pytest.skip('set MLBRICKS_BOLT_V05_ALLOW_JIT=1 to build v0.5 CUDA extension')
    torch.manual_seed(seed)
    ref = _model(cache=cache, layers=2, context=96).to('cuda', dtype=torch.float16)
    _stimulate(ref)
    fast = copy.deepcopy(ref)
    ref.enable_bolt_optimized_inference(True)
    fast.enable_bolt_optimized_inference(True)
    fast.enable_bolt_v05_native(gate=True, avq=True)
    fast.prepare_bolt_v05_native()
    prompt = torch.randint(0, 127, (1, 32), device='cuda')
    with torch.inference_mode():
        baseline = ref.generate(prompt, 16, temperature=0)
        native = fast.generate(prompt, 16, temperature=0)
        assert torch.equal(baseline, native), 'native eager differs from PyTorch'
        # Separate graph model (capturing and comparing on a single model risks
        # sharing mutated cache/scratch memory from eager baseline).
        graph_model = copy.deepcopy(ref)
        graph_model.enable_bolt_v05_native(gate=True, avq=True)
        graph_model.prepare_bolt_v05_native()
        session = graph_model.prepare_cuda_graph_generation(prompt, 16)
        graph_tokens = session.generate()
        assert torch.equal(baseline, graph_tokens), 'native CUDA Graph differs from PyTorch'
        assert all(torch.isfinite(cache.state).all() for cache in session.caches)
        assert all(b.attn._modern._v05_gate_output is not None for b in graph_model.blocks)
        assert all(b.attn._modern.avq._v05_features is not None for b in graph_model.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
def test_v05_native_kernel_replay_64_steps_state_finite():
    if os.environ.get('MLBRICKS_BOLT_V05_ALLOW_JIT') != '1':
        pytest.skip('set MLBRICKS_BOLT_V05_ALLOW_JIT=1 for GPU test')
    torch.manual_seed(5506)
    m = _model(layers=6, cache='tight', context=600).to('cuda', dtype=torch.float16)
    _stimulate(m)
    m.enable_bolt_optimized_inference(True)
    m.enable_bolt_v05_native(gate=True, avq=True)
    m.prepare_bolt_v05_native()
    ids = torch.randint(0, 127, (1, 512), device='cuda')
    with torch.inference_mode():
        session = m.prepare_cuda_graph_generation(ids, 64)
        out = session.generate()
        assert out.shape == (1, 576)
        assert all(torch.isfinite(cache.state).all() for cache in session.caches)
        assert all(torch.isfinite(cache.c).all() for cache in session.caches)
