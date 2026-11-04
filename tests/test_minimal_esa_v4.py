"""Acceptance tests for the native ESA-only public interface."""
import importlib
import torch
import pytest
import mlbricks
from mlbricks import ESA, ESAModel, ESAModelConfig, SOUP, Brick, Bricks
from mlbricks.esa.fast_engine.public_api import ESA as NativeESA


def test_one_public_class_no_wrappers():
    assert ESA is NativeESA
    assert importlib.import_module('mlbricks.esa').ESA is ESA
    assert not hasattr(mlbricks, 'ESAAdapter')
    assert not hasattr(mlbricks, 'ESAConfig')
    assert not hasattr(mlbricks, 'FastCurrentESA')
    assert not hasattr(mlbricks, 'esa') or not isinstance(mlbricks.esa, type)
    assert not hasattr(ESA(embd=24, head=6, backend='pytorch'), 'layer')


@pytest.mark.parametrize('variant', ['current','base'])
def test_output_state_gradients_and_checkpoint(variant):
    torch.manual_seed(11)
    esa = ESA(embd=24, head=6, variant=variant, backend='pytorch', device='cpu')
    x = torch.randn(2, 7, 24, requires_grad=True)
    y, state = esa(x)
    assert y.shape == x.shape
    assert state.shape == (2, 6, 4) and state.dtype == torch.float32
    (y.square().mean()).backward()
    assert esa.qgv.weight.grad is not None
    with torch.inference_mode():
        out, new_state = esa.decode_step(x.detach()[:,0], state.detach())
        assert out.shape == (2, 1, 24) and new_state.shape == state.shape
        ref, ref_state = esa(x.detach()[:,:1], state.detach())
        torch.testing.assert_close(out, ref, atol=0, rtol=0)
        torch.testing.assert_close(new_state, ref_state, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_mixed_precision_controls_are_fp32(dtype):
    esa = ESA(embd=24, head=6, backend='pytorch')
    esa.to(dtype=dtype)
    assert esa.qgv.weight.dtype == dtype
    assert esa.log_half_life.dtype == torch.float32
    assert esa.raw_decay_bias.dtype == torch.float32
    assert esa.state_norm.weight.dtype == torch.float32


def test_full_lm_native_train_generate_and_state_keys():
    cfg=ESAModelConfig(vocab_size=64,block=16,n_layer=2,head=6,embd=24,
                       dropout=0.0,precision='fp32',training_compile=False)
    model=ESAModel(cfg,device='cpu')
    assert isinstance(model.blocks[0].esa, ESA)
    toks=torch.randint(0,64,(2,6))
    logits,loss=model(toks,targets=toks)
    assert logits.shape==(2,6,64)
    loss.backward()
    assert model.blocks[0].esa.qgv.weight.grad is not None
    assert 'blocks.0.esa.qgv.weight' in model.state_dict()
    assert not any('.esa.layer.' in k for k in model.state_dict())
    model.eval()
    with torch.inference_mode():
        y=model.generate(input_ids=toks[:1],seek=2,compile=False)
    assert y.shape==(1,8)


def test_fast_esaq_adapter_checkpoint_migration_is_load_time_only():
    cfg=ESAModelConfig(vocab_size=64,block=16,n_layer=2,head=6,embd=24,
                       dropout=0.0,precision='fp32',training_compile=False)
    m=ESAModel(cfg,device='cpu')
    old={k.replace('.esa.','.esa.layer.'):v.clone() for k,v in m.state_dict().items()}
    reloaded=ESAModel(cfg,device='cpu')
    reloaded.load_state_dict(old,strict=True)
    for k,v in m.state_dict().items():
        torch.testing.assert_close(v,reloaded.state_dict()[k],atol=0,rtol=0)


def test_soup_builtin_esa_has_no_wrapper():
    s=SOUP(dim=24,width=16,depth=2,mixer='esa',ffn='saffn').eval()
    assert isinstance(s.layers[0].mixer, ESA)
    x=torch.randn(2,5,24)
    with torch.inference_mode():
        y=s(x)
        _,cache=s.prefill(x)
        step,cache2=s.decode_step(x[:,:1],cache)
    assert y.shape==x.shape and step.shape==(2,1,24)
    assert cache2.mixer_states[0].shape==(2,8,3)


def test_bricks_direct_esa():
    m=Bricks(vocab_size=64,dim=24,context=16,
             layers=[Brick(mixer=ESA(embd=24,head=6,backend='pytorch'))],
             backend='pytorch').eval()
    with torch.inference_mode():
        assert m(torch.randint(0,64,(2,5))).shape==(2,5,64)


def test_vesa_direct_esa():
    from mlbricks.vesa.layers.mixer import ESAMixer
    from mlbricks.vesa.config import ESAConfig
    m=ESAMixer(ESAConfig(dim=24,heads=6,backend='pytorch'))
    assert isinstance(m.engine, ESA)
    x=torch.randn(2,5,24)
    y,s=m(x)
    assert y.shape==x.shape and s.shape==(2,24)
    step,s2=m.step(x[:,0],s)
    assert step.shape==(2,24) and s2.shape==(2,24)
    yy,ss=m(x,reverse=True)
    assert yy.shape==x.shape and ss.shape==(2,24)
