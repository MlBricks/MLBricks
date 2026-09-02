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
