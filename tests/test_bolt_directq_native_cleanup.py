"""Direct-Q CUDA ABI and width-matched AVQ regression tests.

CPU tests do not claim that native CUDA compilation or GPU parity passed.
GPU cases run only when CUDA and the compiled modern extension are available.
"""
from pathlib import Path
import inspect
import pytest
import torch
from mlbricks import Bolt
from mlbricks.bolt import native_backend


def test_native_python_api_has_only_direct_q_features():
    assert list(inspect.signature(native_backend.features).parameters) == ['q', 'u', 'g']
    assert list(inspect.signature(native_backend.supports_features).parameters) == ['q', 'u', 'g']
    assert not hasattr(native_backend, 'features_folded')
    assert not hasattr(native_backend, 'supports_folded_features')


def test_direct_q_wrapper_rejects_gradient_silencing(monkeypatch):
    class FakeNative:
        def features_forward(self, q, u, g):
            return q.float(), u * (1.0 + torch.tanh(g))
    monkeypatch.setattr(native_backend, 'get_extension', lambda: FakeNative())
    q = torch.randn(1, 2, 3, 16, requires_grad=True)
    u, g = torch.randn_like(q), torch.randn_like(q)
    with pytest.raises(RuntimeError, match='inference-only'):
        native_backend.features(q, u, g)
    with torch.no_grad():
        actual_q, actual_c = native_backend.features(q, u, g)
        assert torch.equal(actual_q, q.float())
        torch.testing.assert_close(actual_c, u * (1.0 + torch.tanh(g)))


def test_public_models_have_no_query_p_and_matched_identity_memory():
    for name, dim in [('compact', 16), ('standard', 32)]:
        m = Bolt(d_model=384, num_heads=12, profile=name, device='cpu')
        assert m._modern.latent_dim == dim
        assert m._modern.avq_dim == 12 * dim
        assert isinstance(m._modern.avq.back, torch.nn.Identity)
        assert not any(k.endswith('.p') for k in m.state_dict())


def test_native_modern_source_exports_no_legacy_p_symbols():
    root = Path(native_backend.__file__).parent / 'native_modern'
    cuda = (root / 'kernels.cu').read_text()
    bindings = (root / 'bindings.cpp').read_text()
    assert 'features_backward_kernel' not in cuda
    assert 'dp_parts' not in cuda
    assert 'APPLY_P' not in cuda
    assert 'features_forward_folded' not in cuda + bindings
    assert 'features_backward' not in bindings
    assert 'm.def("features_forward"' in bindings
    assert 'dp_parts.sum({0,2})' not in cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
@pytest.mark.parametrize('latent', [16, 32])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_cuda_direct_q_features_match_reference(latent, dtype):
    ext = native_backend.get_extension()
    if ext is None:
        pytest.skip('Modern BOLT CUDA extension unavailable')
    torch.manual_seed(703 + latent)
    shape = (2, 12, 7, latent)
    q = torch.randn(shape, device='cuda', dtype=dtype)
    u = torch.randn(shape, device='cuda', dtype=dtype)
    g = torch.randn(shape, device='cuda', dtype=dtype)
    with torch.no_grad():
        assert native_backend.supports_features(q, u, g)
        q_native, c_native = native_backend.features(q, u, g)
        q_ref, c_ref = q.float(), u * (1.0 + torch.tanh(g))
    torch.testing.assert_close(q_native, q_ref, rtol=0, atol=0)
    torch.testing.assert_close(c_native.float(), c_ref.float(),
                               rtol=2e-3 if dtype==torch.float16 else 2e-6,
                               atol=2e-3 if dtype==torch.float16 else 2e-6)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA GPUs required')
def test_cuda_native_features_reject_cross_gpu_inputs():
    ext = native_backend.get_extension()
    if ext is None:
        pytest.skip('Modern BOLT CUDA extension unavailable')
    q = torch.randn(1, 2, 3, 16, device='cuda:0')
    u = torch.randn(1, 2, 3, 16, device='cuda:1')
    g = torch.randn_like(q)
    assert not native_backend.supports_features(q, u, g)
    with pytest.raises(RuntimeError, match='same CUDA device'):
        ext.features_forward(q, u, g)
