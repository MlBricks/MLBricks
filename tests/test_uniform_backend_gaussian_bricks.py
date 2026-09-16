from __future__ import annotations

import inspect
import torch
import pytest

from mlbricks import (
    Attention,
    Brick,
    Bricks,
    ESA,
    ElasticBit,
    FFN,
    Bolt,
    Gaussian,
    ResController,
    StateAwareFFN,
    VesaConfig,
)


def test_default_backend_is_auto_everywhere_relevant():
    assert inspect.signature(ESA).parameters["backend"].default == "auto"
    assert inspect.signature(Attention).parameters["backend"].default == "auto"
    assert inspect.signature(Bolt).parameters["backend"].default == "auto"
    assert inspect.signature(StateAwareFFN).parameters["backend"].default == "auto"
    assert inspect.signature(ResController).parameters["backend"].default == "auto"
    assert VesaConfig().backend == "auto"
    assert ElasticBit().backend == "auto"


def test_set_backend_override():
    a = Bolt(32, 4)
    assert a.backend == "auto"
    assert a.set_backend("pytorch") is a
    assert a.backend == "pytorch"
    assert a.set_backend("native").backend == "native"


def test_gaussian_forward_all_positions():
    ids = torch.randint(0, 97, (2, 8))
    for position in ("none", "learned", "sinusoidal", "rope"):
        model = Gaussian(
            vocab_size=97,
            context=32,
            layers=2,
            dim=32,
            heads=4,
            latent_dim=8,
            position=position,
            dropout=0.0,
        )
        assert model(ids).shape == (2, 8, 97)


def test_gaussian_saffn_and_rescontroller():
    model = Gaussian(
        vocab_size=97,
        context=16,
        layers=2,
        dim=32,
        heads=4,
        latent_dim=8,
        ffn="saffn",
        residual="rescontroller",
        dropout=0.0,
    )
    ids = torch.randint(0, 97, (1, 6))
    assert model(ids).shape == (1, 6, 97)


def test_gaussian_cache_generation():
    model = Gaussian(
        vocab_size=97,
        context=16,
        layers=2,
        dim=32,
        heads=4,
        latent_dim=8,
        dropout=0.0,
    )
    ids = torch.randint(0, 97, (1, 5))
    out = model.generate(ids, 3, temperature=0.0)
    assert out.shape == (1, 8)


def test_bricks_mixed_pipeline_forward_and_generation():
    layers = [
        Brick(
            mixer=ESA(embd=32, head=4, precision="fp32", device=None),
            ffn=FFN(32, 64, activation="silu"),
            dim=32,
        ),
        Brick(
            mixer=Bolt(32, 4, latent_dim=8),
            ffn=FFN(32, 64, activation=lambda x: x * torch.sigmoid(x)),
            position="rope",
            dim=32,
        ),
        Brick(
            mixer=Attention(32, 4),
            ffn=FFN(32, 64),
            dim=32,
        ),
    ]
    model = Bricks(
        vocab_size=97,
        dim=32,
        context=20,
        layers=layers,
        position="sinusoidal",
    )
    ids = torch.randint(0, 97, (1, 5))
    assert model(ids).shape == (1, 5, 97)
    assert model.generate(ids, 2, temperature=0.0).shape == (1, 7)
    names = {row["component"] for row in model.backend_report()}
    assert {"ESA", "Bolt", "Attention"}.issubset(names)


def test_native_override_fails_cleanly_when_not_supported():
    model = Bolt(32, 4, latent_dim=8, backend="native")
    x = torch.randn(1, 32)
    c = torch.randn(1, 4, 4, 8)
    rho = torch.ones(1, 4, 4)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="backend='native'"):
            model.decode(x, c, rho)
