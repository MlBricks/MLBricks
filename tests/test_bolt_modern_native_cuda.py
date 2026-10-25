"""Opt-in tests for compiled BOLT/AVQ native kernels (CUDA Toolkit required)."""
import copy
import os
import pytest
import torch
from mlbricks import Bolt
from mlbricks.bolt import native_backend as backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')

def _require_ext():
    if backend.get_extension() is None:
        pytest.skip('mlbricks._bolt_modern_cuda unavailable: MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 pip install -v .')

@pytest.mark.parametrize('cache', ['normal','tight'])
@pytest.mark.parametrize('mode', ['quality','fast'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float16])
def test_bolt_cuda_forward_and_backward(cache,mode,dtype):
    _require_ext(); torch.manual_seed(147)
    reference=Bolt(384,12,mode=mode,cache=cache,backend='pytorch').cuda()
    native=copy.deepcopy(reference).set_backend('native')
    if mode=='quality':
        with torch.no_grad():
            reference._modern.avq.gamma.fill_(.2)
            native._modern.avq.gamma.fill_(.2)
    x=torch.randn(1,25,384,device='cuda',requires_grad=True)
    y=x.detach().clone().requires_grad_()
    # Enable AMP for half GEMMs (parameters retain FP32 master weights).
    with torch.autocast('cuda',dtype=dtype,enabled=dtype==torch.float16):
        a=reference(x)
        b=native(y)
    torch.testing.assert_close(b.float(),a.float(),atol=.045 if dtype==torch.float16 else .002,
                               rtol=.02 if dtype==torch.float16 else .003)
    target=torch.randn_like(a.float())
    (a.float()*target).mean().backward()
    (b.float()*target).mean().backward()
    torch.testing.assert_close(y.grad,x.grad,atol=.03 if dtype==torch.float16 else .003,
                               rtol=.15 if dtype==torch.float16 else .03)
    for pname in ('q_proj.weight','c_proj.weight','g_proj.weight'):
        ga=dict(reference._modern.named_parameters())[pname].grad
        gb=dict(native._modern.named_parameters())[pname].grad
        torch.testing.assert_close(gb,ga,atol=.03 if dtype==torch.float16 else .003,
                                   rtol=.15 if dtype==torch.float16 else .03)
    if mode=='quality':
        g1=reference._modern.avq.project.weight.grad
        g2=native._modern.avq.project.weight.grad
        torch.testing.assert_close(g2,g1,atol=.03 if dtype==torch.float16 else .003,
                                   rtol=.15 if dtype==torch.float16 else .03)

@pytest.mark.parametrize('mode', ['quality','fast'])
@pytest.mark.parametrize('cache_mode', ['normal','tight'])
def test_bolt_cuda_prefill_decode_and_recurrent_state(mode,cache_mode):
    _require_ext();torch.manual_seed(901)
    ref=Bolt(384,12,mode=mode,cache=cache_mode,backend='pytorch').cuda().eval()
    nat=copy.deepcopy(ref).set_backend('native')
    if mode=='quality':
        with torch.no_grad():ref._modern.avq.gamma.fill_(.17);nat._modern.avq.gamma.fill_(.17)
    x=torch.randn(1,41,384,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        prefix, (cr,sr) = ref.prefill(x[:,:32])
        native_prefix,(cn,sn) = nat.prefill(x[:,:32])
        torch.testing.assert_close(native_prefix.float(),prefix.float(),atol=.035,rtol=.025)
        if mode=='quality':torch.testing.assert_close(sn,sr,atol=.02,rtol=.025)
        for t in range(32,41):
            yr,(cr,sr)=ref.decode_step(x[:,t:t+1,:],(cr,sr))
            yn,(cn,sn)=nat.decode_step(x[:,t:t+1,:],(cn,sn))
            torch.testing.assert_close(yn.float(),yr.float(),atol=.035,rtol=.035)
            if mode=='quality':torch.testing.assert_close(sn,sr,atol=.025,rtol=.03)

@pytest.mark.parametrize('mode', ['quality','fast'])
def test_bolt_native_preallocated_cache(mode):
    _require_ext()
    torch.manual_seed(402)
    m=Bolt(384,12,mode=mode,backend='native').cuda().eval()
    if mode=='quality':
        with torch.no_grad():m._modern.avq.gamma.fill_(.18)
    from mlbricks import BoltPresetCache
    x=torch.randn(1,64,384,device='cuda')
    cache=BoltPresetCache.allocate(1,12,72,32,198 if mode=='quality' else 0,device='cuda',dtype=torch.float16)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        pre,(c,s)=m.prefill(x[:,:48])
        cache.load_prefix(c,s)
        for t in range(48,64):
            next_o=m._modern.decode_with_cache(x[:,t:t+1,:],cache,position=t)
            assert next_o.isfinite().all()
        assert cache.length==64
        assert cache.c[:,:,:64,:].isfinite().all()


def test_no_legacy_substitution_on_missing_extension(monkeypatch):
    monkeypatch.setattr(backend,'get_extension',lambda:None)
    m=Bolt(384,12,mode='quality',backend='native').cuda()
    with pytest.raises(RuntimeError,match='mlbricks._bolt_modern_cuda'):
        m(torch.randn(1,4,384,device='cuda'))
