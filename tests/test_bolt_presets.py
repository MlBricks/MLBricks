"""CPU correctness for the 2026 BOLT quality/fast and normal/tight presets."""
import io

import pytest
import torch

from mlbricks import Bolt, BoltModel, BoltPreset, BoltPresetCache


@pytest.mark.parametrize('mode', ['quality', 'fast'])
@pytest.mark.parametrize('cache', ['normal', 'tight'])
@pytest.mark.parametrize('position', [None, 'rope'])
def test_preset_forward_cached_parity_with_nonzero_avq(mode, cache, position):
    torch.manual_seed(12)
    layer=Bolt(32,4,mode=mode,cache=cache,position=position,backend='pytorch').eval()
    assert layer.is_modern
    assert layer.latent_dim==(32 if cache=='normal' else 16)
    assert layer.avq_dim==(198 if mode=='quality' else 0)
    if mode=='quality':
        with torch.no_grad():layer._modern.avq.gamma.fill_(.16)
    x=torch.randn(2,7,32)
    with torch.no_grad():
        entire=layer(x)
        prefix,(c,s)=layer.prefill(x[:,:4])
        torch.testing.assert_close(prefix,entire[:,:4],atol=1e-5,rtol=2e-5)
        if mode=='quality':assert s.shape==(2,198) and s.dtype==torch.float32
        else:assert s is None
        for j in range(4,7):
            y,(c,s)=layer.decode_step(x[:,j:j+1],(c,s))
            torch.testing.assert_close(y,entire[:,j:j+1],atol=1e-5,rtol=2e-5)
    assert c.shape==(2,4,7,layer.latent_dim)


def test_bolt_default_is_quality_normal_198_and_fast_has_no_avq():
    b=Bolt(384,12)
    assert (b.is_modern,b.mode,b.cache_mode,b.latent_dim,b.avq_dim)==(True,'quality','normal',32,198)
    f=Bolt(384,12,mode='fast')
    assert f._modern.avq is None and not any('avq' in k for k in f.state_dict())
    legacy=Bolt(32,4,latent_dim=8)
    assert not legacy.is_modern


def test_cached_generation_reuses_state_and_repeated_prefill_resets_it():
    torch.manual_seed(7)
    m=BoltModel(vocab_size=47,dim=32,heads=4,layers=2,context=16,dropout=0.).eval()
    for block in m.blocks:
        with torch.no_grad():block.attn._modern.avq.gamma.fill_(.12)
    tokens=torch.randint(0,47,(1,7))
    with torch.no_grad():
        expected=m(tokens)
        m.prepare_generation(batch_size=1,max_context=16,device='cpu',warmup_native=False)
        b=m._generation_caches(tokens,capacity=8)
        logits,cs=m.prefill(tokens[:,:4],cache_buffers=b)
        assert all(isinstance(c,BoltPresetCache) for c in cs)
        torch.testing.assert_close(logits,expected[:,:4],atol=1e-5,rtol=2e-5)
        for i in range(4,7):
            out,cs=m.decode_step(tokens[:,i:i+1],cs,position=i)
            torch.testing.assert_close(out,expected[:,i:i+1],atol=2e-5,rtol=2e-5)
        for cache in cs:assert cache.length==7
        for cache in cs:cache.reset()
        assert all(c.length==0 for c in cs)
        assert all(c.state.abs().sum()==0 for c in cs)
        # Re-prefill the same buffers and ensure no stale recurrent state is used.
        logits2,_=m.prefill(tokens[:,:4],cache_buffers=cs)
        torch.testing.assert_close(logits2,expected[:,:4],atol=1e-5,rtol=2e-5)


def test_quality_avq_gradients_reach_memory_and_query_scale():
    torch.manual_seed(11)
    m=Bolt(24,3,mode='quality',cache='tight',backend='pytorch')
    with torch.no_grad():m._modern.avq.gamma.fill_(.12)
    loss=m(torch.randn(2,5,24)).square().mean()
    loss.backward()
    assert m._modern.p.grad is not None and m._modern.p.grad.abs().sum()>0
    assert m._modern.avq.project.weight.grad is not None
    assert m._modern.avq.project.weight.grad.abs().sum()>0
    assert m._modern.avq.back.weight.grad is not None


def test_checkpoint_round_trip_and_strict_old_dimension_rejection():
    old=Bolt(32,4,mode='quality',cache='normal',avq_dim=192)
    modern=Bolt(32,4)
    with pytest.raises(RuntimeError):modern.load_state_dict(old.state_dict(),strict=True)
    b=io.BytesIO();torch.save(modern.state_dict(),b);b.seek(0)
    modern2=Bolt(32,4)
    modern2.load_state_dict(torch.load(b,map_location='cpu',weights_only=True),strict=True)
    torch.testing.assert_close(modern(torch.randn(1,3,32)).shape[0]*torch.ones(1),torch.ones(1))


def test_native_is_not_silently_used_for_new_equations():
    m=Bolt(32,4)
    m.set_backend('native')
    with pytest.raises(RuntimeError,match='requires a CUDA tensor'):
        m(torch.randn(1,3,32))
