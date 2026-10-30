"""Regression tests for opt-in, Triton-free modern BOLT inference patch."""
import copy
import io

import pytest
import torch

from mlbricks import Bolt, BoltModel, BoltPresetCache


@pytest.mark.parametrize('cache', ('tight', 'normal'))
@pytest.mark.parametrize('mode', ('quality', 'fast'))
@pytest.mark.parametrize('position', (None, 'rope'))
@pytest.mark.parametrize('bias', (False, True))
def test_opt_in_preset_full_and_cache_parity(cache, mode, position, bias):
    torch.manual_seed(78)
    ref = Bolt(48, 3, mode=mode, cache=cache, position=position,
               bias=bias, backend='pytorch').eval()
    opt = copy.deepcopy(ref).enable_optimized_inference(True)
    with torch.no_grad():
        ref._modern.p.uniform_(.6, 1.5)
        if ref._modern.avq is not None:
            ref._modern.avq.gamma.fill_(.23)
            ref._modern.avq.norm.weight.uniform_(.7, 1.4)
        opt.load_state_dict(ref.state_dict())
        x = torch.randn(2, 6, 48)
        torch.testing.assert_close(ref(x), opt(x), rtol=2e-5, atol=2e-5)
        ref_prefix, rc = ref.prefill(x[:,:3])
        opt_prefix, oc = opt.prefill(x[:,:3])
        torch.testing.assert_close(ref_prefix,opt_prefix,rtol=2e-5,atol=2e-5)
        # Explicit fixed-buffer cache avoids concatenating the growing history.
        fixed=BoltPresetCache.allocate(2,3,7,opt.latent_dim,opt.avq_dim,
                                       device=x.device,dtype=x.dtype).load_prefix(*oc)
        for i in range(3,6):
            out,rc=ref.decode_step(x[:,i:i+1],rc)
            o=opt._modern.decode_with_cache(x[:,i:i+1],fixed,position=i)
            torch.testing.assert_close(out,o,rtol=3e-5,atol=3e-5)
        if position=='rope':
            assert fixed.k is not None  # pre-rotated key cache


def test_pack_rebuilds_on_p_and_bias_changes_and_keeps_state_dict():
    torch.manual_seed(8)
    layer=Bolt(48, 3, mode='quality',cache='tight',bias=True,backend='pytorch').eval()
    base_keys=set(layer.state_dict())
    layer.enable_optimized_inference(True)
    x=torch.randn(1,3,48)
    with torch.no_grad():
        layer._modern.p.fill_(1.1)
        layer(x)
        old=layer._modern._packed_inference[0]
        layer._modern.p[0,0].add_(.7)
        layer._modern.q_proj.bias[0].add_(.15)
        layer(x)
        new=layer._modern._packed_inference[0]
        assert old.data_ptr()!=new.data_ptr()
        assert torch.equal(new[0],layer._modern.q_proj.weight[0]*layer._modern.p[0,0])
        # AVQ folded back cache invalidates on gamma / norm changes.
        layer._modern.avq.gamma.fill_(.5)
        layer(x)
        folded=layer._modern.avq._folded_back
        layer._modern.avq.norm.weight[0].add_(.3)
        layer(x)
        assert folded.data_ptr()!=layer._modern.avq._folded_back.data_ptr()
        assert set(layer.state_dict())==base_keys
        buf=io.BytesIO();torch.save(layer.state_dict(),buf);buf.seek(0)
        loaded=Bolt(48,3,mode='quality',cache='tight',bias=True,backend='pytorch').eval()
        loaded.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(layer(x),loaded(x),rtol=2e-5,atol=2e-5)


def test_training_uses_original_forward_gradients_when_option_enabled():
    torch.manual_seed(24)
    a=Bolt(48,3,mode='quality',cache='tight',backend='pytorch').train()
    a._modern.avq.gamma.data.fill_(.23)
    b=copy.deepcopy(a).enable_optimized_inference(True).train()
    x=torch.randn(2,5,48)
    aa=a(x).square().mean();bb=b(x).square().mean()
    torch.testing.assert_close(aa,bb,atol=1e-7,rtol=1e-7)
    aa.backward();bb.backward()
    for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        torch.testing.assert_close(pa.grad,pb.grad,atol=1e-7,rtol=1e-6)


def test_graph_safe_cpu_step_matches_regular_fixed_cache_decode():
    torch.manual_seed(41)
    m=Bolt(48,3,mode='quality',cache='tight',backend='pytorch').eval()
    m.enable_optimized_inference(True)
    with torch.no_grad():
        m._modern.avq.gamma.fill_(.2)
        m._modern.p.uniform_(.5,1.5)
        x=torch.randn(1,7,48)
        _,pref=m.prefill(x[:,:4])
        normal=BoltPresetCache.allocate(1,3,8,16,198,device=x.device,dtype=x.dtype).load_prefix(*pref)
        graph=BoltPresetCache.allocate(1,3,8,16,198,device=x.device,dtype=x.dtype).load_prefix(*pref)
        positions=torch.arange(8)
        for i in range(4,7):
            eager=m._modern.decode_with_cache(x[:,i:i+1],normal,position=i)
            graph_out=m._modern.graph_decode_with_cache(
                x[:,i:i+1],graph,position=torch.tensor(i),positions=positions)
            torch.testing.assert_close(eager,graph_out,atol=2e-5,rtol=2e-5)


def test_bolt_model_opt_in_matches_full_model_generation_and_unchanged_keys():
    torch.manual_seed(45)
    m=BoltModel(vocab_size=49,dim=48,heads=3,layers=2,context=20,
                bolt_cache='tight',backend='pytorch',dropout=0).eval()
    keys=set(m.state_dict())
    ids=torch.randint(0,49,(1,6))
    with torch.no_grad():
        for block in m.blocks:
            block.attn._modern.p.uniform_(.5,1.3)
            block.attn._modern.avq.gamma.fill_(.13)
        baseline=m.generate(ids,3,temperature=0)
        m.enable_bolt_optimized_inference(True)
        optimized=m.generate(ids,3,temperature=0)
        assert torch.equal(baseline,optimized)
        assert set(m.state_dict())==keys
        m.enable_bolt_optimized_inference(False)
        assert torch.equal(baseline,m.generate(ids,3,temperature=0))


def test_cuda_graph_backend_rejects_cpu_cleanly():
    m=BoltModel(vocab_size=49,dim=48,heads=3,layers=1,context=20,
                bolt_cache='tight',backend='pytorch').eval()
    m.enable_bolt_optimized_inference(True)
    with pytest.raises(RuntimeError,match='CUDA GPU'):
        m.prepare_cuda_graph_generation(torch.ones(1,3,dtype=torch.long),2)


def test_complete_graph_body_cpu_simulation_matches_eager_greedy():
    """Exercise the graph's complete model body without requiring CUDA capture."""
    from mlbricks.bolt.graph_generation import BoltCudaGraphGenerator
    torch.manual_seed(2026)
    m=BoltModel(vocab_size=55,dim=48,heads=3,layers=2,context=16,
                bolt_cache='tight',backend='pytorch',dropout=0).eval()
    m.enable_bolt_optimized_inference(True)
    prompt=torch.randint(0,55,(1,5))
    with torch.inference_mode():
        eager=m.generate(prompt,4,temperature=0)
        g=BoltCudaGraphGenerator.__new__(BoltCudaGraphGenerator)
        g.model=m
        g.prompt=prompt
        g.prompt_length=prompt.size(1)
        g.max_new_tokens=4
        g.capacity=9
        g.caches=[BoltPresetCache.allocate(1,3,9,16,198,device='cpu',dtype=torch.float32)
                  for _ in m.blocks]
        g.position=torch.tensor(5,dtype=torch.long)
        g.positions=torch.arange(9)
        g.tokens=torch.empty((1,9),dtype=torch.long)
        g.static_token=torch.empty((1,1),dtype=torch.long)
        g._seed()
        for _ in range(3):
            g._graph_step()
        assert torch.equal(eager,g.tokens)
