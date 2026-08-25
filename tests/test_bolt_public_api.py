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
