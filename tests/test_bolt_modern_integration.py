"""Public integration coverage for modern-only BOLT modes."""
import pytest
import torch
from mlbricks import Bolt, BoltConfig, BoltModel, Brick, Bricks, Attention, RoPE
from mlbricks.bolt.enhanced_mqa import EnhancedMQACache
from mlbricks.bolt.shared_c import SharedCCache

@pytest.mark.parametrize('mode,kv,cache_type', [('quantised',1,EnhancedMQACache),('quantised',2,EnhancedMQACache),('base',None,EnhancedMQACache),('keyless',None,SharedCCache)])
@pytest.mark.parametrize('retention',['learned','decay'])
@pytest.mark.parametrize('avq',[False,True])
def test_public_cache_and_dynamic_static_decode(mode,kv,cache_type,retention,avq):
    torch.manual_seed(101)
    layer=Bolt(32,4,mode=mode,kv_heads=kv,retention=retention,avq=avq,
               backend='pytorch',device='cpu',position=RoPE()).eval()
    x=torch.randn(2,7,32)
    with torch.no_grad():
        expected=layer(x)
        prefix,dynamic=layer.prefill_with_cache(x[:,:4])
        c=layer.allocate_cache(batch_size=2,capacity=10)
        assert isinstance(c,cache_type)
        c.load_prefix(*dynamic)
        assert c.state.dtype==torch.float32
        assert c.state.shape==(2,32 if avq else 0)
        assert c.length==4
        for t in ('k','v','c'):
            if hasattr(c,t):
                n=1 if mode=='keyless' else (4 if mode=='base' else kv)
                assert getattr(c,t).shape==(2,n,10,8)
        for i in range(4,7):
            y,c_dynamic=layer.decode_step(x[:,i:i+1],dynamic)
            z=layer.decode_with_cache(x[:,i:i+1],c,position=i)
            torch.testing.assert_close(y,expected[:,i:i+1],rtol=5e-5,atol=5e-5)
            torch.testing.assert_close(z,y,rtol=5e-5,atol=5e-5)
            dynamic=c_dynamic
        assert c.length==7

@pytest.mark.parametrize('mode,kv',[('quantised',1),('quantised',2),('keyless',None),('base',None)])
def test_bricks_mixed_generation_full_forward_and_fixed_cache(mode,kv):
    torch.manual_seed(47)
    m=Bricks(vocab_size=31,dim=32,context=16,layers=[
        Brick(mixer=Bolt(32,4,mode=mode,kv_heads=kv,device='cpu',backend='pytorch'),dim=32),
        Brick(mixer=Attention(32,4,backend='pytorch'),dim=32),
    ],backend='pytorch').eval()
    ids=torch.randint(0,31,(2,7))
    with torch.no_grad():
        expected=m(ids)
        p,s=m.prefill(ids[:,:4])
        torch.testing.assert_close(p,expected[:,:4],rtol=5e-5,atol=5e-5)
        m.prepare_generation(batch_size=2,max_context=12,warmup_native=False)
        buffers=m._prepared_generation['buffers']
        p,s=m.prefill(ids[:,:4],cache_buffers=buffers)
        for t in range(4,7):
            y,s=m.decode_step(ids[:,t:t+1],s,position=t)
            torch.testing.assert_close(y,expected[:,t:t+1],atol=5e-5,rtol=5e-5)
        assert buffers[0].length==7
        assert m.generate(ids[:,:4],3,temperature=0).shape==(2,7)

def test_retired_mode_and_graph_fail_closed():
    with pytest.raises(ValueError): Bolt(32,4,mode='legacy',device='cpu')
    with pytest.raises(ValueError): BoltConfig(vocab_size=31,bolt_mode='legacy')
    m=BoltModel(vocab_size=31,context=12,layers=1,dim=32,heads=4,device='cpu',graph='auto').eval()
    assert not m.blocks[0].attn.graph_allowed()
    ids=torch.randint(0,31,(1,4))
    assert m.generate(ids,2,temperature=0).shape==(1,6)
    assert not m.model_graph_status()['using_graph']
