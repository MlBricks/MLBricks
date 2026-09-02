import torch

import mlbricks
from mlbricks import SOUP, soup


def test_soup_top_level_exports_and_forward():
    model = SOUP(dim=16, width=24, depth=1, mixer="esa", backend="pytorch", precision="fp32")
    model = model.cpu()
    x = torch.randn(2, 5, 16)
    y = model(x)
    assert y.shape == x.shape
    assert mlbricks.soup is soup


def test_soup_layerwise_mixer_selection():
    model = soup(
        dim=16, width=[24, 20], depth=2,
        mixer=["esa", "bolt"], ffn=["saffn", "ffn"],
        backend="pytorch", precision="fp32",
    ).cpu()
    x = torch.randn(1, 4, 16)
    y = model(x)
    assert y.shape == x.shape
    assert model.mixer_names == ["esa", "bolt"]


def test_soup_mixed_esa_bolt_recurrent_generation_matches_full_forward():
    torch.manual_seed(11)
    model = SOUP(
        dim=16,
        width=[24, 20],
        depth=2,
        mixer=["esa", "bolt"],
        ffn=["saffn", "ffn"],
        mixer_config=[{"head": 4}, {"num_heads": 4, "latent_dim": 4}],
        backend="pytorch",
        precision="fp32",
        memory_dim=8,
        fusion_hidden=24,
    ).cpu().eval()

    prefix = torch.randn(2, 5, 16)
    continuation = torch.randn(2, 3, 16)

    with torch.no_grad():
        prefill_y, cache = model.prefill(prefix)
        torch.testing.assert_close(prefill_y, model(prefix), rtol=1e-5, atol=1e-6)

        sequence = prefix
        for i in range(continuation.size(1)):
            token = continuation[:, i:i + 1]
            sequence = torch.cat((sequence, token), dim=1)
            expected = model(sequence)[:, -1:]
            actual, cache = model.decode_step(token, cache)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_soup_mixed_generation_fast_request_falls_back_safely():
    model = SOUP(
        dim=16,
        width=[24, 20],
        depth=2,
        mixer=["esa", "bolt"],
        ffn=["saffn", "ffn"],
        mixer_config=[{"head": 4}, {"num_heads": 4, "latent_dim": 4}],
        backend="pytorch",
        precision="fp32",
        memory_dim=8,
        fusion_hidden=24,
    ).cpu().eval()

    model.prepare_generation(fast=True)
    assert model._generation_fast is False
    assert model._generation_plans is None

    x = torch.randn(1, 4, 16)
    y, cache = model.prefill(x)
    next_y, _ = model.decode_step(torch.randn(1, 1, 16), cache)
    assert y.shape == x.shape
    assert next_y.shape == (1, 1, 16)
