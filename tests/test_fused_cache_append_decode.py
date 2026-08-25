import torch

from mlbricks import Attention, Brick, Bricks, Bolt, Gaussian
from mlbricks.cache import GaussCache, KVCache


def test_attention_decode_append_projected_matches_append_then_decode_pytorch():
    torch.manual_seed(101)
    attn = Attention(32, 4, backend="pytorch", position="rope").eval()
    hist = torch.randn(2, 5, 32)
    x = torch.randn(2, 32)
    k, v = attn.project_cache_state(hist, start_pos=0)
    q, kn, vn = attn.project_decode_state(x, start_pos=5)

    ref_k = torch.empty(2, 4, 12, 8)
    ref_v = torch.empty_like(ref_k)
    ref_k[:, :, :5].copy_(k)
    ref_v[:, :, :5].copy_(v)
    ref_k[:, :, 5:6].copy_(kn)
    ref_v[:, :, 5:6].copy_(vn)
    ref = attn.decode_projected(q, ref_k, ref_v, used_length=6)

    got_k = torch.empty_like(ref_k)
    got_v = torch.empty_like(ref_v)
    got_k[:, :, :5].copy_(k)
    got_v[:, :, :5].copy_(v)
    got = attn.decode_append_projected(
        q, kn, vn, got_k, got_v, position=5
    )

    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(got_k[:, :, :6], ref_k[:, :, :6])
    torch.testing.assert_close(got_v[:, :, :6], ref_v[:, :, :6])


def test_gauss_decode_append_projected_matches_append_then_decode_pytorch_rope():
    torch.manual_seed(102)
    gauss = Bolt(32, 4, latent_dim=8, backend="pytorch", position="rope").eval()
    hist = torch.randn(2, 5, 32)
    x = torch.randn(2, 32)
    c, rho = gauss.project_cache_state(hist)
    q, cn, rn = gauss.project_decode_state(x, start_pos=5)

    ref_c = torch.empty(2, 4, 12, 8)
    ref_r = torch.empty(2, 4, 12)
    ref_c[:, :, :5].copy_(c)
    ref_r[:, :, :5].copy_(rho)
    ref_c[:, :, 5:6].copy_(cn)
    ref_r[:, :, 5:6].copy_(rn)
    ref = gauss.decode_projected(q, ref_c, ref_r, used_length=6)

    got_c = torch.empty_like(ref_c)
    got_r = torch.empty_like(ref_r)
    got_c[:, :, :5].copy_(c)
    got_r[:, :, :5].copy_(rho)
    got = gauss.decode_append_projected(
        q, cn, rn, got_c, got_r, position=5
    )

    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(got_c[:, :, :6], ref_c[:, :, :6])
    torch.testing.assert_close(got_r[:, :, :6], ref_r[:, :, :6])


def test_gaussian_prepared_generation_updates_cache_length_without_cache_append(monkeypatch):
    torch.manual_seed(103)
    model = Gaussian(vocab_size=47, context=24, layers=2, dim=32, heads=4, latent_dim=8)
    model.prepare_generation(batch_size=1, max_context=12, warmup_native=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared generation must not call GaussCache.append")

    monkeypatch.setattr(GaussCache, "append", forbidden)
    ids = torch.randint(0, 47, (1, 5))
    out = model.generate(ids, 4, temperature=0)
    assert out.shape == (1, 9)
    assert all(cache.length == 8 for cache in model._prepared_generation["caches"])


def test_bricks_prepared_generation_avoids_python_cache_append(monkeypatch):
    torch.manual_seed(104)
    model = Bricks(
        vocab_size=53,
        dim=32,
        context=24,
        layers=[
            Brick(mixer=Bolt(32, 4, latent_dim=8), dim=32),
            Brick(mixer=Attention(32, 4), dim=32),
        ],
    )
    model.prepare_generation(batch_size=1, max_context=12, warmup_native=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared generation must not call cache.append")

    monkeypatch.setattr(GaussCache, "append", forbidden)
    monkeypatch.setattr(KVCache, "append", forbidden)
    ids = torch.randint(0, 53, (1, 5))
    out = model.generate(ids, 4, temperature=0)
    assert out.shape == (1, 9)
    buffers = model._prepared_generation["buffers"]
    assert buffers[0].length == 8
    assert buffers[1].length == 8


def test_native_source_exports_fused_cache_apis():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bindings = (root / "mlbricks" / "bolt" / "bolt_attention_bindings.cpp").read_text()
    cuda = (root / "mlbricks" / "bolt" / "bolt_attention_cuda.cu").read_text()
    for name in (
        "baseline_append_cache",
        "gauss_append_cache",
        "baseline_decode_append_out",
        "gauss_decode_append_out",
        "gauss_rope_decode_out_used",
        "gauss_rope_decode_append_out",
    ):
        assert name in bindings
        assert name in cuda
