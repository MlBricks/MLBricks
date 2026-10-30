"""CPU and optional GPU regression for QKV model-level auto graph inference."""
import copy
import io
import pytest
import torch
from mlbricks import AttentionModel, BoltModel
from mlbricks.attention_model import QKVCudaGraphGenerator


def _model():
    return AttentionModel(113,dim=48,heads=3,layers=2,context=64)


def test_model_forward_training_and_checkpoint_keys(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','0')
    torch.manual_seed(2026)
    m=_model()
    ids=torch.randint(0,113,(2,10))
    original_keys=tuple(m.state_dict())
    m.train()
    logits,loss=m(ids,targets=ids)
    assert logits.shape==(2,10,113)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    for p in m.parameters():
        assert p.dtype==torch.float32
    m.eval()
    out=m.generate(ids,7,temperature=0)
    assert out.shape==(2,17)
    assert m.inference_info()['last_execution']=='pytorch_eager'
    assert m.inference_info()['precision_policy']=='preserve_model_dtype'
    assert tuple(m.state_dict())==original_keys
    buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
    copy_m=_model()
    copy_m.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    assert tuple(copy_m.state_dict())==original_keys


@pytest.mark.parametrize('seq',[4,12,21])
def test_qkv_prefill_equals_causal_forward(monkeypatch,seq):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','0')
    m=_model().eval()
    x=torch.randint(0,113,(1,seq))
    with torch.inference_mode():
        pref=m._prefill_cache(x,m._allocate_cache(1,seq+5))
        ref=m(x)[:,-1:]
    torch.testing.assert_close(pref,ref,rtol=0,atol=0)


@pytest.mark.parametrize('batch',[1,2])
def test_cpu_qkv_graph_step_simulated_exact(monkeypatch,batch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','0')
    torch.manual_seed(5505)
    m=_model().eval()
    ids=torch.randint(0,113,(batch,11))
    new_tokens=7
    with torch.inference_mode():
        reference=m.generate(ids,new_tokens,temperature=0)
        session=QKVCudaGraphGenerator.__new__(QKVCudaGraphGenerator)
        session.model=m;session.prompt=ids;session.length=ids.size(1)
        session.max_new_tokens=new_tokens
        session.capacity=ids.size(1)+new_tokens
        session.caches=m._allocate_cache(batch,session.capacity)
        session.position=torch.full((),session.length,dtype=torch.long)
        session.positions=torch.arange(session.capacity)
        session.tokens=torch.empty((batch,session.capacity),dtype=torch.long)
        session.static_token=torch.empty((batch,1),dtype=torch.long)
        session._seed()
        for _ in range(new_tokens-1):session._step()
        torch.testing.assert_close(reference,session.tokens,rtol=0,atol=0)


def test_dtype_policy_does_not_silently_half(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','0')
    m=_model().eval()
    ids=torch.randint(0,113,(1,8))
    before={k:v.clone() for k,v in m.state_dict().items()}
    _=m.generate(ids,3)
    assert all(v.dtype==before[k].dtype for k,v in m.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA GPU required')
def test_cuda_first_model_constructor(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','1')
    m=_model().eval()
    assert next(m.parameters()).is_cuda
    ids=torch.randint(0,113,(1,7))
    assert m(ids).is_cuda


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,5),reason='T4 required')
def test_t4_auto_qkv_graph_matches_eager_logits_and_cache(monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','1')
    torch.manual_seed(2026)
    m=_model().to(device='cuda',dtype=torch.float16).eval()
    ref=copy.deepcopy(m)
    ids=torch.randint(0,113,(1,10),device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        expected=ref.generate(ids,7,temperature=0)
        output=m.generate(ids,7,temperature=0)
    assert m.inference_info()['last_execution']=='strict_cuda_graph'
    assert torch.equal(expected,output)
