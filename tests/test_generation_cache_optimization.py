import torch

from mlbricks import Attention, Brick, Bricks, ESA, Bolt, Gaussian
from mlbricks.cache import GaussCache, KVCache


def _legacy_gaussian_generate(model, input_ids, steps):
    was = model.training
    model.eval()
    logits, caches = model.prefill(input_ids)  # tuple caches: legacy path
    total = input_ids.size(1) + steps
    tokens = torch.empty(input_ids.size(0), total, dtype=input_ids.dtype)
    tokens[:, : input_ids.size(1)].copy_(input_ids)
    nxt = logits[:, -1, :]
    for i in range(steps):
        tok = nxt.argmax(dim=-1, keepdim=True)
        pos = input_ids.size(1) + i
        tokens[:, pos:pos + 1].copy_(tok)
        if i + 1 < steps:
            logits, caches = model.decode_step(tok, caches, position=pos)
            nxt = logits[:, -1, :]
    if was:
        model.train()
    return tokens


def test_gaussian_fixed_cache_matches_legacy_cat_path():
    torch.manual_seed(1)
    model = Gaussian(vocab_size=43, context=32, layers=2, dim=32, heads=4, latent_dim=8)
    ids = torch.randint(0, 43, (2, 5))
    expected = _legacy_gaussian_generate(model, ids, 4)
    actual = model.generate(ids, 4, temperature=0)
    torch.testing.assert_close(actual, expected)
    prepared = model._prepared_generation
    assert prepared is not None
    assert all(isinstance(c, GaussCache) for c in prepared["caches"])
    assert all(c.length == 8 for c in prepared["caches"])


def test_attention_and_gauss_used_length_ignore_unwritten_capacity():
    torch.manual_seed(2)
    xhist = torch.randn(1, 5, 32)
    x = torch.randn(1, 32)

    attn = Attention(32, 4, backend="pytorch").eval()
    k, v = attn.project_cache_state(xhist)
    kc = torch.randn(1, 4, 16, 8)
    vc = torch.randn(1, 4, 16, 8)
    kc[:, :, :5].copy_(k); vc[:, :, :5].copy_(v)
    ref = attn.decode(x, k, v, used_length=5)
    got = attn.decode(x, kc, vc, used_length=5)
    torch.testing.assert_close(got, ref)

    gauss = Bolt(32, 4, latent_dim=8, backend="pytorch").eval()
    c, rho = gauss.project_cache_state(xhist)
    cc = torch.randn(1, 4, 16, 8)
    rr = torch.randn(1, 4, 16)
    cc[:, :, :5].copy_(c); rr[:, :, :5].copy_(rho)
    ref = gauss.decode(x, c, rho, used_length=5)
    got = gauss.decode(x, cc, rr, used_length=5)
    torch.testing.assert_close(got, ref)


def test_bricks_prepares_heterogeneous_fixed_caches():
    model = Bricks(
        vocab_size=41,
        dim=32,
        context=24,
        layers=[
            Brick(mixer=Bolt(32, 4, latent_dim=8), dim=32),
            Brick(mixer=Attention(32, 4), dim=32),
            Brick(mixer=ESA(embd=32, head=4, device="cpu"), dim=32),
        ],
    )
    model.prepare_generation(batch_size=2, max_context=16, warmup_native=False)
    buffers = model._prepared_generation["buffers"]
    assert isinstance(buffers[0], GaussCache)
    assert isinstance(buffers[1], KVCache)
    assert buffers[2] is None
    ids = torch.randint(0, 41, (2, 4))
    out = model.generate(ids, 3, temperature=0)
    assert out.shape == (2, 7)


def test_execution_plan_reports_zero_device_transfers():
    model = Gaussian(vocab_size=31, context=16, layers=1, dim=32, heads=4, latent_dim=8)
    plan = model.execution_plan()
    assert plan.device_transfers == 0
    assert plan.requested_backend == "auto"


def test_gaussian_prefill_logits_match_eval_forward():
    torch.manual_seed(3)
    model = Gaussian(vocab_size=47, context=24, layers=2, dim=32, heads=4, latent_dim=8).eval()
    ids = torch.randint(0, 47, (2, 7))
    with torch.no_grad():
        direct = model(ids)
        prefill, _ = model.prefill(ids)
    torch.testing.assert_close(prefill, direct, rtol=1e-5, atol=1e-6)


def test_packed_gauss_projection_refreshes_after_weight_change():
    torch.manual_seed(4)
    g = Bolt(32, 4, latent_dim=8, backend="pytorch").eval()
    x = torch.randn(2, 32)
    q1, c1, r1 = g.project_decode_state(x, start_pos=0)
    with torch.no_grad():
        g.q_proj.weight.add_(0.01)
    q2, c2, r2 = g.project_decode_state(x, start_pos=0)
    assert not torch.allclose(q1, q2)
    # C/G parameters did not change, so cache state is unchanged.
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(r1, r2)
