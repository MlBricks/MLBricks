"""CPU contract/ownership tests. CUDA Graph execution qualified in GPU notebook."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from mlbricks import Bolt, BoltConfig, BoltModel
from mlbricks.bolt.layer_graph import _outer_graph_active, _is_nvidia_cuda
from mlbricks.bolt.presets import BoltCache, BoltPreset


def test_defaults_and_override():
    m = Bolt(64, 4, device='cpu').eval()
    assert (m.backend, m.graph) == ('auto', 'auto')
    assert m.execution_info()['backend'] == 'not_observed'
    m.set_backend('pytorch')
    m.set_graph(False)
    assert m.graph == 'off'
    assert m._modern.graph == 'off'
    assert m._modern.backend == 'pytorch'


def test_preallocated_cache_numerical_reference_fallback_cpu():
    torch.manual_seed(314)
    m = Bolt(64, 4, mode='quality', cache='tight', backend='auto', device='cpu').eval()
    ref = Bolt(64, 4, mode='quality', cache='tight', backend='pytorch', graph=False, device='cpu').eval()
    ref.load_state_dict(m.state_dict())
    a = m.allocate_cache(batch_size=1, capacity=128)
    b = ref.allocate_cache(batch_size=1, capacity=128)
    x = torch.randn(1, 1, 64)
    for pos in range(8):
        y = m.decode_with_cache(x, a, position=pos)
        y_ref = ref.decode_with_cache(x, b, position=pos)
        torch.testing.assert_close(y, y_ref, rtol=0, atol=0)
        torch.testing.assert_close(a.c[:, :, :pos+1], b.c[:, :, :pos+1], rtol=0, atol=0)
        torch.testing.assert_close(a.state, b.state, rtol=0, atol=0)
        assert a.length == b.length == pos + 1
        assert m.execution_info()['backend'] == 'eager'
        x = torch.randn(1, 1, 64)


def test_multiple_layers_and_sessions_independent_cpu():
    layers = nn.ModuleList([Bolt(32, 4, mode='quality', cache='tight', device='cpu').eval()
                            for _ in range(3)])
    caches = [m.allocate_cache(batch_size=1, capacity=128) for m in layers]
    data = torch.randn(1, 1, 32)
    for pos in range(4):
        for module, cache in zip(layers, caches):
            data = module.decode_with_cache(data, cache, position=pos)
    assert [c.length for c in caches] == [4, 4, 4]
    assert len({c.c.data_ptr() for c in caches}) == 3
    assert all(m.graph == 'auto' for m in layers)


def test_real_hybrid_forward_default_works_on_cpu():
    class Hybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.bolt1 = Bolt(32, 4, mode='quality', cache='tight', device='cpu')
            self.middle = nn.Linear(32, 32)
            self.bolt2 = Bolt(32, 4, mode='quality', cache='normal', device='cpu')
        def forward(self, x):
            h = self.attention(x, x, x, need_weights=False)[0]
            return self.bolt2(self.middle(self.bolt1(h)))
    model = Hybrid().eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32)).shape == (2, 3, 32)
    assert model.bolt1.backend == model.bolt1.graph == 'auto'
    assert model.bolt2.backend == model.bolt2.graph == 'auto'


def test_public_boltmodel_config_integration():
    m = BoltModel(BoltConfig(vocab_size=64, context=32, layers=2,
                             dim=32, heads=4, device='cpu')).eval()
    assert m.graph == 'auto'
    assert all(b.attn.graph == 'auto' for b in m.blocks)
    with torch.no_grad():
        out = m.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=3)
    assert out.shape == (1, 6)
    assert all(b.attn.execution_info()['backend'] == 'eager' for b in m.blocks)


def test_force_graph_on_cpu_fails_informatively():
    m = Bolt(32, 4, mode='quality', cache='tight', device='cpu', graph=True).eval()
    cache = m.allocate_cache(batch_size=1, capacity=128)
    with pytest.raises(RuntimeError, match='graph="on" requested but unavailable'):
        m.decode_with_cache(torch.randn(1, 1, 32), cache, position=0)
    assert cache.length == 0


def test_cache_reset_and_strict_position():
    cache = BoltCache.allocate(1, 4, 128, 16, 64, device='cpu', dtype=torch.float32)
    cache.length = 12
    cache.state.fill_(2)
    cache.reset()
    assert cache.length == 0
    assert not cache.state.any()
    m = BoltPreset(32, 4, cache='tight', backend='pytorch', graph=False).eval()
    with pytest.raises(ValueError, match='position must match'):
        m.decode_with_cache(torch.randn(1,1,32), cache, position=1)


def test_no_cuda_misidentified_as_nvidia():
    assert not _is_nvidia_cuda(torch.empty(1))
    assert not _outer_graph_active(torch.device('cpu'))


def test_custom_operator_registration_and_fake_shape():
    from mlbricks.bolt.custom_ops import bolt_dynamic_decode
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        q = torch.empty((1, 4, 1, 16), device='cpu', dtype=torch.float32)
        current = torch.empty_like(q, dtype=torch.float16)
        cache = torch.empty((1, 4, 128, 16), device='cpu', dtype=torch.float16)
        scratch = torch.empty((1, 4, 128), device='cpu', dtype=torch.float32)
        position = torch.empty((1,), device='cpu', dtype=torch.int64)
        out = bolt_dynamic_decode(q, current, cache, scratch, position,
                                  0.25, 512, 1, False)
    assert out.shape == q.shape and out.dtype == torch.float32
