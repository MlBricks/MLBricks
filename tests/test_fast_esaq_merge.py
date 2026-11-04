"""Fast-EsaQ native mixer regression; no alternate ESA wrapper is tested."""
import pytest
import torch
from mlbricks import ESA, ESAModel, ESAModelConfig
from mlbricks.esa.fast_engine.public_api import CurrentESA,BaseESA


@pytest.mark.parametrize('heads',[1,2,4,6])
def test_mixer_heads_backward(heads):
    m=ESA(embd=24,head=heads,backend='pytorch',device='cpu')
    x=torch.randn(2,7,24,requires_grad=True)
    y,state=m(x)
    assert isinstance(m,CurrentESA) and y.shape==x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and m.qgv.weight.grad is not None
    assert m.log_half_life.grad is not None and m.raw_decay_bias.grad is not None


@pytest.mark.parametrize('variant',["current","base"])
def test_cached_continuation_matches_full(variant):
    torch.manual_seed(77)
    m=ESA(embd=24,head=6,backend='pytorch',variant=variant,device='cpu').eval()
    assert isinstance(m, BaseESA if variant=='base' else CurrentESA)
    seq=torch.randn(2,10,24)
    with torch.no_grad():
        full,_=m(seq)
        prefix,state=m.prefill(seq[:,:6])
        decoded=[]
        for t in range(6,10):
            out,state=m.decode_step(seq[:,t],state)
            decoded.append(out)
        torch.testing.assert_close(torch.cat((prefix,*decoded),dim=1),full,rtol=1e-5,atol=1e-5)
        assert state.dtype==torch.float32


def test_state_layout_and_native_checkpoint_keys():
    m=ESA(embd=24,head=6,backend='pytorch').eval()
    y,state=m(torch.randn(1,5,24))
    assert state.shape==(1,6,4)
    assert 'log_half_life' in m.state_dict() and 'layer.log_half_life' not in m.state_dict()
    torch.testing.assert_close(m.log_half_life,torch.tensor([16,64,256,1024,4096,8192],dtype=torch.float32).log())


def test_half_precision_fp32_recurrence():
    m=ESA(embd=24,head=6,backend='pytorch',device='cpu').half()
    with torch.no_grad():
        out,state=m.prefill(torch.randn(2,5,24).half())
    assert out.dtype==torch.float16 and state.dtype==torch.float32
    assert m.qgv.weight.dtype==torch.float16
    assert m.log_half_life.dtype==torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(state).all()


def test_old_thunder_is_not_implicitly_reinterpreted():
    cfg=ESAModelConfig(vocab_size=24,embd=24,head=6,block=16,n_layer=1,precision='fp32',training_compile=False)
    m=ESAModel(cfg,device='cpu')
    with pytest.raises(ValueError,match='Old Thunder'):
        m.load_state_dict({'blocks.0.esa.layer.out_proj.weight':torch.zeros(24,24)},strict=False)


def test_native_engine_matches_concrete_current_esa():
    torch.manual_seed(42)
    m=ESA(embd=24,head=6,backend='pytorch',device='cpu')
    original=CurrentESA(embd=24,head=6,backend='pytorch')
    original.load_state_dict(m.state_dict())
    x=torch.randn(2,9,24)
    with torch.no_grad():
        y0,s0=original(x);y1,s1=m(x)
    torch.testing.assert_close(y0,y1,rtol=0,atol=0)
    torch.testing.assert_close(s0,s1,rtol=0,atol=0)
