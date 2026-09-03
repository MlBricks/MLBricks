import inspect

import torch

import mlbricks
from mlbricks import Bolt, BoltAttention, Bolt, BoltModel, BoltConfig, Gaussian, GaussianConfig
from mlbricks.bolt import Bolt as BoltFromPackage


def test_bolt_is_canonical_and_legacy_alias_is_identical():
    assert BoltFromPackage is Bolt
    assert BoltAttention is Bolt
    assert Bolt is Bolt
    assert mlbricks.Bolt is Bolt
    assert Bolt.__name__ == "Bolt"
    assert inspect.signature(Bolt).parameters["backend"].default == "auto"


def test_bolt_model_aliases_preserve_ready_made_model_api():
    assert BoltModel is Gaussian
    assert BoltConfig is GaussianConfig
    model = BoltModel(vocab_size=31, context=16, layers=1, dim=32, heads=4, latent_dim=8)
    ids = torch.randint(0, 31, (1, 4))
    assert model(ids).shape == (1, 4, 31)


def test_legacy_gauss_import_constructs_bolt():
    layer = Bolt(32, 4, latent_dim=8)
    assert isinstance(layer, Bolt)
    assert type(layer).__name__ == "Bolt"


def test_bolt_prefill_decode_step_matches_full_causal_forward():
    torch.manual_seed(7)
    layer = Bolt(16, 4, latent_dim=4, backend="pytorch", dropout=0.0).eval()
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

    c_cache, rho_cache = cache
    assert c_cache.shape == (2, 4, 8, 4)
    assert rho_cache.shape == (2, 4, 8)



def test_bolt_packed_training_matches_explicit_sdpa_and_preserves_parameters():
    import math
    import torch.nn.functional as F

    torch.manual_seed(19)
    layer = Bolt(32, 4, latent_dim=8, backend="pytorch", dropout=0.0, use_sdpa=True).train()
    x1 = torch.randn(2, 7, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = layer(x1)

    q = layer.q_proj(x2).view(2, 7, 4, 8).transpose(1, 2)
    u = layer.c_proj(x2)
    g = layer.g_proj(x2)
    c = (u * (1.0 + torch.tanh(g))).view(2, 7, 4, 8).transpose(1, 2)
    rho = torch.rsqrt(c.float().square().mean(dim=-1) + layer.eps)
    k = (c.float() * rho.unsqueeze(-1)).to(c.dtype)
    o = F.scaled_dot_product_attention(
        q, k, c, dropout_p=0.0, is_causal=True,
        scale=1.0 / math.sqrt(float(layer.head_dim)),
    )
    y2 = layer.out_proj(o.transpose(1, 2).contiguous().view(2, 7, 32))
    torch.testing.assert_close(y1, y2, rtol=1e-6, atol=1e-7)

    grad = torch.randn_like(y1)
    y1.backward(grad)
    g1 = x1.grad.detach().clone()
    for param in layer.parameters():
        param.grad = None
    y2.backward(grad)
    torch.testing.assert_close(g1, x2.grad, rtol=2e-5, atol=2e-6)

    keys = set(layer.state_dict())
    assert "q_proj.weight" in keys
    assert "c_proj.weight" in keys
    assert "g_proj.weight" in keys
    assert not any("qcg" in key for key in keys)


def test_bolt_native_stage1_cuda_matches_pytorch_when_available():
    import pytest
    from mlbricks._backend import load_cuda_extension

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    ext = load_cuda_extension()
    if ext is None or not hasattr(ext, "gauss_stage1_forward"):
        pytest.skip("native Bolt Stage-1 extension unavailable")

    torch.manual_seed(23)
    ref = Bolt(128, 4, latent_dim=16, backend="pytorch", dropout=0.0, use_sdpa=True).cuda().half().train()
    opt = Bolt(128, 4, latent_dim=16, backend="auto", dropout=0.0, use_sdpa=True).cuda().half().train()
    opt.load_state_dict(ref.state_dict())

    x1 = torch.randn(2, 128, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1 = ref(x1)
    y2 = opt(x2)
    torch.testing.assert_close(y2, y1, rtol=4e-3, atol=4e-3)

    grad = torch.randn_like(y1)
    y1.backward(grad)
    y2.backward(grad)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=8e-3, atol=8e-3)
    assert opt.resolved_backend() in {"planner(auto forward) + native-decode", "planner(auto full-sequence) + native-decode"}
