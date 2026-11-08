"""CPU contracts for experimental BOLT public whole-model graph integration.

CUDA graph capture/replay requires a separate two-T4 GPU qualification.
"""
import copy
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from mlbricks import BoltModel, BoltConfig
from mlbricks.bolt.model_graph import BoltModelGraphEngine


def _model(width=16):
    return BoltModel(BoltConfig(
        vocab_size=71,context=32,layers=2,dim=64,heads=4,
        profile='compact' if width == 16 else 'standard',
        precision='fp32',device='cpu',position='none',
        ffn='standard',residual='standard',dropout=0.0,
        backend='pytorch'
    )).eval()


def test_cpu_model_rejects_graph_and_preserves_greedy_outputs():
    torch.manual_seed(45)
    m=_model()
    prompt=torch.tensor([[2,4,6,9]])
    assert BoltModelGraphEngine.eligibility(m,prompt,8) == 'CUDA is unavailable'
    a=m.generate(prompt,4,temperature=0)
    b=m.generate(prompt,4,temperature=0)
    assert torch.equal(a,b) and a.shape == (1,8)


def test_graph_off_never_tries_manager():
    m=_model()
    m.graph='off'
    prompt=torch.tensor([[1,2,3]])
    with patch.object(BoltModelGraphEngine,'supports',side_effect=AssertionError('manager must not run')):
        output=m.generate(prompt,3,temperature=0)
    assert output.shape==(1,6)


def test_sampling_never_tries_manager():
    m=_model()
    m.graph='auto'
    prompt=torch.tensor([[1,2,3]])
    with patch.object(BoltModelGraphEngine,'supports',side_effect=AssertionError('manager must not run')):
        output=m.generate(prompt,3,temperature=1.0)
    assert output.shape==(1,6)


def test_auto_policy_requires_64_tokens_and_greedy():
    m=_model()
    m.graph='auto'
    prompt=torch.tensor([[1,2,3]])
    with patch.object(BoltModelGraphEngine,'supports',side_effect=AssertionError('no graph for short request')):
        out=m.generate(prompt,3,temperature=0)
    assert out.shape==(1,6)


def test_policy_auto_dispatch_uses_full_gpu_token_tensor():
    m=_model()
    m.graph='auto'
    prompt=torch.tensor([[1,2,3]])
    class FakeEngine:
        using_graph=True
        def generate_tensor(self,n):
            return torch.full((1,n), 12, dtype=torch.long)
        def status(self):
            return {'using_graph':True,'graph_variants':1}
    with patch.object(BoltModelGraphEngine,'supports',return_value=True), \
         patch.object(BoltModelGraphEngine,'from_prompt',return_value=FakeEngine()):
        result=m.generate(prompt,64,temperature=0)
    assert result.shape==(1,67)
    assert torch.equal(result[:,:3],prompt)
    assert torch.equal(result[:,3:],torch.full((1,64),12))
    assert m._mlbricks_model_graph_status['graph_variants']==1


class ReferenceDynamicDecoder:
    """CPU implementation of the native device-position attention kernel."""
    def decode(self,q,current,history,scratch,pos_gpu,scale,chunk_tokens,chunks,split):
        pos=int(pos_gpu.item())
        history[:,:,pos:pos+1,:].copy_(current)
        keys=history[:,:,:pos+1,:]
        return F.scaled_dot_product_attention(q,keys,keys,
                  is_causal=False,dropout_p=0.0,scale=scale)


@pytest.mark.parametrize('width',[16,32])
def test_dynamic_whole_model_kernel_matches_eager_step_on_cpu(width):
    torch.manual_seed(99)
    model=_model(width)
    prompt=torch.tensor([[1,7,12,6]])
    model.prepare_generation(batch_size=1,max_context=20)
    _,cache=model.prefill(prompt,cache_buffers=model._prepared_generation['caches'])
    cloned=copy.deepcopy(cache)
    for c in cloned:c.scratch=torch.empty(c.c.shape[:3],dtype=torch.float32)
    engine=object.__new__(BoltModelGraphEngine)
    engine.model=model
    engine.ext=ReferenceDynamicDecoder()
    engine.chunk_tokens=512
    x=torch.tensor([[3]])
    position=prompt.size(1)
    expected,cache=model.decode_step(x,cache,position=position)
    actual=engine._decode_dynamic(cloned,x,torch.tensor([position]),'fast',1)
    assert torch.equal(actual,expected[:,-1,:].argmax(-1,keepdim=True))
    for a,b in zip(cloned,cache):
        assert torch.allclose(a.c[:,:,:position+1],b.c[:,:,:position+1],atol=1e-5)
        assert torch.allclose(a.state,b.state,atol=1e-5)


def test_no_nested_layer_decode_method_in_capture():
    # The specialized path bypasses layer-owned sessions by calling a direct
    # dynamic extension. This makes an enclosing model CUDA graph viable.
    import inspect
    source=inspect.getsource(BoltModelGraphEngine._decode_dynamic)
    assert '.decode_with_cache(' not in source
    assert 'self.ext.decode(' in source
