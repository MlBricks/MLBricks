"""Stable default API, FP32 training and nonregistered FP16 mirror invariants."""
import io
import torch
import pytest

from mlbricks import BoltModel, AttentionModel
from mlbricks.bolt.inference_mirror import _tensor_signature


def small_bolt():
    return BoltModel(vocab_size=67, context=48, layers=2, dim=24, heads=3,
                     bolt_mode='quality', bolt_cache='tight', avq_dim=198,
                     backend='pytorch', position='none')


def small_qkv():
    return AttentionModel(vocab_size=67, context=48, layers=2, dim=24, heads=3)


@pytest.mark.parametrize('factory', [small_bolt, small_qkv])
def test_default_greedy_and_original_master_weights(factory, monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '0')
    torch.manual_seed(2006)
    model = factory()
    ids = torch.randint(0, 67, (1, 7))
    before = {key: p.clone() for key,p in model.state_dict().items()}
    before_keys = tuple(model.state_dict())
    model.train()
    default_out = model.generate(ids, 4)
    explicit_out = model.generate(ids, 4, temperature=0)
    assert torch.equal(default_out, explicit_out)
    assert model.training, 'generate must restore the original training flag'
    assert model.inference_info()['last_execution'] == 'pytorch_eager'
    assert model.inference_info()['precision_policy'] == 'preserve_model_dtype'
    assert before_keys == tuple(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    state=torch.load(buffer,map_location='cpu',weights_only=True)
    assert tuple(state) == before_keys
    assert '_mlbricks_fp16_inference_mirror' not in model._modules


@pytest.mark.parametrize('factory', [small_bolt, small_qkv])
def test_gradient_update_and_cache_invalidation(factory, monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA', '0')
    model = factory()
    original = _tensor_signature(model)
    ids = torch.randint(0, 67, (1, 6))
    optimizer=torch.optim.SGD(model.parameters(),lr=.002)
    logits, loss = model(ids,targets=ids)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    assert _tensor_signature(model) != original
    dummy=object()
    model.__dict__['_mlbricks_fp16_inference_mirror']=('test',dummy)
    assert '_mlbricks_fp16_inference_mirror' not in model._modules
    assert '_mlbricks_fp16_inference_mirror' not in model.state_dict()
    model.train()
    assert '_mlbricks_fp16_inference_mirror' not in model.__dict__
    model.__dict__['_mlbricks_fp16_inference_mirror']=('test',dummy)
    model.to('cpu')
    assert '_mlbricks_fp16_inference_mirror' not in model.__dict__


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(7,5),
                    reason='Tesla T4 required for validated FP16 mirror graph')
@pytest.mark.parametrize('factory',[small_bolt,small_qkv])
def test_t4_default_fp32_auto_graph_and_mirror_rebuild(factory, monkeypatch):
    monkeypatch.setenv('MLBRICKS_AUTO_CUDA','1')
    torch.manual_seed(2026)
    model=factory()
    assert next(model.parameters()).dtype==torch.float32
    assert next(model.parameters()).is_cuda
    prompt=torch.randint(0,67,(1,8),device='cuda')
    before={k:v.clone() for k,v in model.state_dict().items()}
    # Deliberately omit .eval(), .half() and temperature=0.
    tokens=model.generate(prompt,6)
    info=model.inference_info()
    assert info['last_execution']=='strict_cuda_graph',info
    assert info['precision_policy']=='fp32_master_fp16_inference_mirror'
    assert info['inference_dtype']=='torch.float16'
    assert next(model.parameters()).dtype==torch.float32
    assert model.training
    assert '_mlbricks_fp16_inference_mirror' not in model._modules
    shadow=model.__dict__['_mlbricks_fp16_inference_mirror'][1]
    shadow_id=id(shadow)
    assert next(shadow.parameters()).dtype==torch.float16
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert torch.equal(tokens,model.generate(prompt,6))
    assert id(model.__dict__['_mlbricks_fp16_inference_mirror'][1])==shadow_id
    # Training clears shadow and changes FP32 master only.
    model.train()
    assert '_mlbricks_fp16_inference_mirror' not in model.__dict__
    optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    loss=model(prompt,targets=prompt)[1]
    loss.backward();optimizer.step();optimizer.zero_grad()
    tokens2=model.generate(prompt,6)
    assert tokens2.shape==tokens.shape
    assert model.inference_info()['last_execution']=='strict_cuda_graph'
    assert id(model.__dict__['_mlbricks_fp16_inference_mirror'][1])!=shadow_id
