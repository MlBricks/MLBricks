"""BOLT constructor interoperability with the familiar PyTorch embed_dim keyword."""

import pytest
import torch

from mlbricks import Bolt


def test_embed_dim_keyword_builds_identical_default_preset_and_checkpoint():
    torch.manual_seed(123)
    original = Bolt(48, 6)
    alias = Bolt(embed_dim=48, num_heads=6)
    alias.load_state_dict(original.state_dict(), strict=True)
    assert (alias.embed_dim, alias.d_model, alias.num_heads, alias.latent_dim, alias.avq_dim) == (48, 48, 6, 32, 198)
    x = torch.randn(2, 5, 48)
    torch.testing.assert_close(original(x), alias(x))
    assert alias.resolved_backend().startswith("pytorch")


def test_embed_dim_alias_supports_fast_tight_and_legacy():
    fast = Bolt(embed_dim=48, num_heads=6, mode="fast", cache="tight")
    assert fast.latent_dim == 16 and fast.avq_dim == 0
    legacy = Bolt(embed_dim=48, num_heads=6, mode="legacy", latent_dim=8)
    assert not legacy.is_modern
    x = torch.randn(1, 3, 48)
    assert fast(x).shape == legacy(x).shape == x.shape


@pytest.mark.parametrize("kwargs", [
    dict(d_model=48, embed_dim=48, num_heads=6),
    dict(d_model=48, embed_dim=64, num_heads=6),
    dict(embed_dim=48),
    dict(num_heads=6),
    dict(embed_dim=0, num_heads=6),
    dict(embed_dim=50, num_heads=6),
    dict(embed_dim=48, num_heads=0),
])
def test_invalid_or_ambiguous_arguments_raise(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Bolt(**kwargs)


def test_modern_preset_never_silently_uses_legacy_native_kernel():
    modern = Bolt(embed_dim=48, num_heads=6, backend="auto")
    assert modern.resolved_backend().startswith("pytorch")
    modern.set_backend("native")
    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        modern(torch.randn(1, 3, 48))
