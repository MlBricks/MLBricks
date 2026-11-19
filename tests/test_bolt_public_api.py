import inspect

import torch

import mlbricks
from mlbricks import Bolt, BoltAttention, BoltModel, BoltConfig
from mlbricks.bolt import Bolt as BoltFromPackage


def test_bolt_is_canonical_and_legacy_alias_is_identical():
    assert BoltFromPackage is Bolt
    assert BoltAttention is Bolt
    assert Bolt is Bolt
    assert mlbricks.Bolt is Bolt
    assert Bolt.__name__ == "Bolt"
    assert inspect.signature(Bolt).parameters["backend"].default == "auto"


def test_bolt_model_aliases_preserve_ready_made_model_api():
    assert BoltModel.__module__ == "mlbricks.bolt.model"
    assert BoltConfig.__module__ == "mlbricks.bolt.config"
    model = BoltModel(vocab_size=31, context=16, layers=1, dim=32, heads=4, latent_dim=8)
    ids = torch.randint(0, 31, (1, 4))
    assert model(ids).shape == (1, 4, 31)


def test_bolt_dynamic_head_constructor():
    layer = Bolt(32, 4, latent_dim=8)
    assert isinstance(layer, Bolt)
    assert type(layer).__name__ == "Bolt"


def test_bolt_prefill_decode_step_matches_full_causal_forward():
    torch.manual_seed(7)
    layer = Bolt(16, 4, latent_dim=4, backend="pytorch", dropout=0.0, device="cpu", precision="fp32").eval()
    prefix = torch.randn(2, 5, 16)
    continuation = torch.randn(2, 3, 16)

    with torch.no_grad():
        prefill_y, cache = layer.prefill(prefix)
        torch.testing.assert_close(prefill_y, layer(prefix), rtol=1e-5, atol=1e-6)

        sequence = prefix
        for i in range(continuation.size(1)):
            token = continuation[:, i:i + 1]
            sequence = torch.cat((sequence, token), dim=1)
            expected = layer(sequence)[:, -1:]
            actual, cache = layer.decode_step(token, cache)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    k,v,state = cache
    assert k.shape == v.shape == (2, 1, 8, 4)
    assert state.shape == (2, 16) and state.dtype == torch.float32






def test_mqa_without_avq_matches_explicit_sdpa_and_gradients():
    import torch.nn.functional as F
    torch.manual_seed(19)
    layer=Bolt(32,4,avq=False,head_output_gate=False,backend='pytorch',device='cpu')
    m=layer._modern
    x1=torch.randn(2,7,32,requires_grad=True); x2=x1.detach().clone().requires_grad_(True)
    y1=layer(x1)
    q=m.q_proj(x2).reshape(2,7,4,8).transpose(1,2)
    k=m.k_proj(x2).reshape(2,7,1,8).transpose(1,2).expand(-1,4,-1,-1)
    v=m.v_proj(x2).reshape(2,7,1,8).transpose(1,2).expand(-1,4,-1,-1)
    y2=m.out_proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(2,7,32))
    torch.testing.assert_close(y1,y2,rtol=1e-5,atol=1e-6)
    grad=torch.randn_like(y1);y1.backward(grad);y2.backward(grad)
    torch.testing.assert_close(x1.grad,x2.grad,rtol=2e-5,atol=2e-6)
    assert '_modern.k_proj.weight' in layer.state_dict()
