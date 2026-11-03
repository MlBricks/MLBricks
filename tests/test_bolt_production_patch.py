"""Production public BOLT API, checkpoint layout and math-preserving tests."""
import copy
from dataclasses import replace
import pytest
import torch
from mlbricks import (Bolt, BoltConfig, BoltModel, BoltStateMemory, BoltCache,
                      Gaussian, GaussianConfig, PostSoftmaxAVQ, BoltPresetCache)


def test_api_names_devices_and_profiles():
    assert BoltModel is not Gaussian
    assert BoltConfig is not GaussianConfig
    assert BoltStateMemory is not PostSoftmaxAVQ
    assert issubclass(PostSoftmaxAVQ, BoltStateMemory)
    assert BoltCache is BoltPresetCache
    for profile, latent in [('standard', 32), ('compact', 16)]:
        cfg = BoltConfig(vocab_size=71, dim=32, heads=4, layers=1,
                         context=12, profile=profile, precision='fp32', backend='pytorch')
        model = BoltModel(cfg)
        assert next(model.parameters()).device.type == 'cpu' if not torch.cuda.is_available() else 'cuda'
        assert model.blocks[0].attn.latent_dim == latent
        assert model.blocks[0].attn._modern.attention_scale == 'latent'
        assert model.blocks[0].attn._modern.projection == 'packed'
        assert model.blocks[0].attn._modern.avq.gamma.item() == pytest.approx(.25)
        x = torch.randint(0, 71, (1, 6))
        logits, loss = model(x, targets=x)
        assert logits.shape == (1, 6, 71) and loss.isfinite()
        loss.backward()
        assert model.blocks[0].attn._modern.packed_qug_weight.grad is not None


def test_old_model_math_preserved():
    model = Gaussian(GaussianConfig(vocab_size=71, dim=32, heads=4, layers=1,
                                    context=12, backend='pytorch'))
    assert model.blocks[0].attn._modern.projection == 'separate'
    assert model.blocks[0].attn._modern.attention_scale == 'head'
    assert model.blocks[0].attn._modern.avq.gamma.item() == 0.
    assert model(torch.randint(0, 71, (1, 4))).shape == (1,4,71)


@pytest.mark.parametrize('latent', [16, 32])
def test_packed_parameters_exact_checkpoint_compatibility(latent):
    seed = 59
    args = dict(d_model=32, num_heads=4, mode='quality', cache=('tight' if latent == 16 else 'normal'),
                backend='pytorch', bias=True, dropout=0.,
                attention_scale='latent', memory_init='balanced')
    from mlbricks.bolt.presets import BoltPreset
    torch.manual_seed(seed); legacy = BoltPreset(**args, projection='separate')
    torch.manual_seed(seed); packed = BoltPreset(**args, projection='packed')
    assert set(packed.state_dict()) == set(legacy.state_dict())
    for k in legacy.state_dict():
        torch.testing.assert_close(packed.state_dict()[k], legacy.state_dict()[k], rtol=0, atol=0)
    x = torch.randn(2, 5, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.testing.assert_close(legacy(x), packed(y), atol=2e-6, rtol=2e-6)
    legacy(x).square().mean().backward()
    packed(y).square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=5e-6, rtol=2e-5)
    state = copy.deepcopy(legacy.state_dict())
    packed.load_state_dict(state, strict=True)
    fresh = BoltPreset(**args, projection='separate')
    fresh.load_state_dict(packed.state_dict(), strict=True)
    torch.testing.assert_close(legacy(x.detach()), fresh(x.detach()), atol=0, rtol=0)


def test_new_model_exports_loadable_into_separate_projection():
    cfg = BoltConfig(vocab_size=51, context=10, layers=1, dim=32, heads=4,
                     profile='compact', precision='fp32', backend='pytorch')
    packed = BoltModel(cfg)
    separate = BoltModel(replace(cfg, projection='separate'))
    separate.load_state_dict(packed.state_dict(), strict=True)
    ids = torch.randint(0, 51, (1, 5))
    torch.testing.assert_close(separate(ids), packed(ids), atol=1e-6, rtol=1e-6)


def test_standalone_profile_and_explicit_cuda_error():
    layer = Bolt(embed_dim=32,num_heads=4,profile='compact', device='cpu', backend='pytorch')
    assert layer.is_modern and layer._modern.projection=='packed'
    assert layer(torch.randn(1,4,32)).shape==(1,4,32)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError,match='unavailable'):
            Bolt(32,4,device='cuda')
        with pytest.raises(RuntimeError,match='unavailable'):
            BoltModel(BoltConfig(vocab_size=20,dim=32,heads=4,layers=1,device='cuda'))
