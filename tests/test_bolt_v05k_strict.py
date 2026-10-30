"""Regression tests for supported opt-in stock-arithmetic inference (v0.5k)."""
import copy
import io
import pytest
import torch
from mlbricks import Bolt, BoltModel
from mlbricks.bolt.presets import BoltPresetCache
from mlbricks.bolt.graph_generation import BoltCudaGraphGenerator


def model(cache='tight', mode='quality'):
    m = BoltModel(vocab_size=109, layers=2, dim=48, heads=3,
                  context=48, bolt_mode=mode, bolt_cache=cache,
                  avq_dim=198, position='none', dropout=0,
                  backend='pytorch').eval()
    with torch.no_grad():
        for b in m.blocks:
            b.attn._modern.p.uniform_(0.55,1.45)
            if b.attn._modern.avq is not None:
                b.attn._modern.avq.gamma.fill_(.23)
                b.attn._modern.avq.norm.weight.uniform_(.7,1.4)
    return m


@pytest.mark.parametrize('cache',['tight','normal'])
@pytest.mark.parametrize('mode',['quality','fast'])
def test_strict_matches_stock_and_checkpoint_keys(cache,mode):
    torch.manual_seed(2026)
    stock=model(cache,mode)
    strict=copy.deepcopy(stock)
    original=set(stock.state_dict().keys())
    assert all(not b.attn._modern.optimized_inference for b in strict.blocks)
    strict.enable_bolt_strict_inference(True)
    assert set(strict.state_dict().keys()) == original
    ids=torch.randint(0,109,(2,7))
    with torch.inference_mode():
        torch.testing.assert_close(stock(ids),strict(ids),rtol=0,atol=0)
        torch.testing.assert_close(stock.generate(ids,5,temperature=0),
                                   strict.generate(ids,5,temperature=0),rtol=0,atol=0)
    for block in strict.blocks:
        att=block.attn._modern
        assert att.strict_inference and att.optimized_inference
        assert att._packed_inference is None
        assert not att.v05_native_gate
        assert att.avq is None or not att.avq.fold_output_inference
    buf=io.BytesIO();torch.save(strict.state_dict(),buf);buf.seek(0)
    restored=model(cache,mode)
    restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(stock(ids),restored(ids),atol=0,rtol=0)


@pytest.mark.parametrize('cache',['tight','normal'])
def test_strict_graph_body_cpu_simulation_matches_stock(cache):
    torch.manual_seed(5505)
    stock=model(cache)
    m=copy.deepcopy(stock).enable_bolt_strict_inference(True)
    prompt=torch.randint(0,109,(1,9))
    n=5; capacity=prompt.shape[1]+n
    with torch.inference_mode():
        target=stock.generate(prompt,n,temperature=0)
        g=BoltCudaGraphGenerator.__new__(BoltCudaGraphGenerator)
        g.model=m;g.prompt=prompt;g.prompt_length=prompt.shape[1];g.max_new_tokens=n
        g.capacity=capacity
        g.caches=[BoltPresetCache.allocate(1,a.num_heads,capacity,a.latent_dim,a.avq_dim,
                                           device='cpu',dtype=torch.float32)
                  for a in (b.attn._modern for b in m.blocks)]
        g.position=torch.tensor(prompt.shape[1]);g.positions=torch.arange(capacity)
        g.tokens=torch.empty((1,capacity),dtype=torch.long)
        g.static_token=torch.empty((1,1),dtype=torch.long)
        g._seed()
        for i in range(n-1):
            g._graph_step()
        assert torch.equal(g.tokens,target)
        assert all(b.attn._modern._packed_inference is None for b in m.blocks)


def test_strict_disables_existing_native_and_switches_cleanly():
    torch.manual_seed(1)
    m=model()
    m.enable_bolt_optimized_inference(True)
    m.enable_bolt_v05_native(gate=True,avq=True)
    m.enable_bolt_strict_inference(True)
    for b in m.blocks:
        a=b.attn._modern
        assert not a.v05_native_gate and not a.avq.v05_native_avq
        assert not a.avq.fold_output_inference
        with pytest.raises(ValueError,match='strict inference'):
            a.enable_v05_native_fusions(gate=True,avq=True)
    m.enable_bolt_optimized_inference(True)
    assert not m.blocks[0].attn._modern.strict_inference
    assert m.blocks[0].attn._modern.avq.fold_output_inference
    m.enable_bolt_strict_inference(True)
    m.enable_bolt_strict_inference(False)
    assert not m.blocks[0].attn._modern.optimized_inference
    assert not m.blocks[0].attn._modern.strict_inference


def test_strict_rejects_legacy_or_wrong_backend():
    a=Bolt(48,3,mode='legacy',latent_dim=16,backend='pytorch')
    with pytest.raises(ValueError,match='modern'):
        a.enable_strict_inference(True)
    a=Bolt(48,3,mode='quality',cache='tight',backend='auto')
    with pytest.raises(ValueError,match='backend'):
        a.enable_strict_inference(True)


def test_strict_preserves_training_grads():
    torch.manual_seed(41)
    stock=model().train()
    strict=copy.deepcopy(stock).enable_bolt_strict_inference(True).train()
    x=torch.randint(0,109,(2,6))
    loss1=stock(x).square().mean();loss2=strict(x).square().mean()
    torch.testing.assert_close(loss1,loss2,atol=0,rtol=0)
    loss1.backward();loss2.backward()
    for (n,p),(n2,p2) in zip(stock.named_parameters(),strict.named_parameters()):
        assert n==n2
        torch.testing.assert_close(p.grad,p2.grad,atol=0,rtol=0)
