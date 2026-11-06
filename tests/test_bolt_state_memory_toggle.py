"""The professional BOLT state_memory toggle is architectural, not a no-op."""
from dataclasses import asdict

import pytest
import torch

from mlbricks import Bolt, BoltConfig, BoltModel


def _config(*, profile='standard', state_memory=True):
    return BoltConfig(
        vocab_size=43, context=12, layers=2, dim=32, heads=4,
        profile=profile, state_memory=state_memory,
        device='cpu', backend='pytorch', precision='fp32', dropout=0.,
    )


@pytest.mark.parametrize('profile,latent', [('standard', 32), ('compact', 16)])
def test_default_and_opt_out_are_genuine_architecture_choices(profile, latent):
    default = _config(profile=profile)
    off = _config(profile=profile, state_memory=False)
    assert default.state_memory is True and default.bolt_mode == 'quality'
    assert off.state_memory is False and off.bolt_mode == 'fast'
    expected_width = 4 * latent
    assert off.state_dim == default.state_dim == expected_width

    with_memory = BoltModel(default)
    without_memory = BoltModel(off)
    assert sum(p.numel() for p in without_memory.parameters()) < sum(p.numel() for p in with_memory.parameters())
    assert all(b.attn.state_memory and b.attn.avq_dim == expected_width for b in with_memory.blocks)
    assert all(isinstance(b.attn._modern.avq.back, torch.nn.Identity) for b in with_memory.blocks)
    assert all(not b.attn.state_memory and b.attn.avq_dim == 0 for b in without_memory.blocks)
    assert all(b.attn.latent_dim == latent for b in without_memory.blocks)
    assert all(b.attn._modern.avq is None for b in without_memory.blocks)
    assert not any('.avq.' in name for name in without_memory.state_dict())

    # The attention-only path equals an enabled model with its memory-read
    # contribution mathematically zeroed, once shared parameters are aligned.
    shared = {k: v for k, v in with_memory.state_dict().items() if '.avq.' not in k}
    without_memory.load_state_dict(shared, strict=True)
    for block in with_memory.blocks:
        with torch.no_grad():
            block.attn._modern.avq.gamma.zero_()
    ids = torch.randint(0, 43, (2, 6))
    with_memory.eval(); without_memory.eval()
    torch.testing.assert_close(with_memory(ids), without_memory(ids), atol=2e-6, rtol=2e-6)

    logits, caches = without_memory.prefill(ids)
    assert logits.shape == (2, 6, 43)
    assert len(caches) == 2
    assert all((cache[1] if isinstance(cache, tuple) else cache.state) is None for cache in caches)

    with pytest.raises(RuntimeError):
        without_memory.load_state_dict(with_memory.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        with_memory.load_state_dict(without_memory.state_dict(), strict=True)


def test_config_serialization_and_legacy_fast_compatibility():
    off = _config(state_memory=False)
    restored = BoltConfig(**asdict(off))
    assert restored.state_memory is False
    assert restored.bolt_mode == 'fast'
    assert restored.profile == 'standard'
    assert restored.device == 'cpu'

    historical_fast = BoltConfig(vocab_size=43, dim=32, heads=4, bolt_mode='fast')
    assert historical_fast.state_memory is False
    assert historical_fast.bolt_mode == 'fast'
    historical_legacy = BoltConfig(vocab_size=43, dim=32, heads=4, bolt_mode='legacy')
    assert historical_legacy.state_memory is False
    assert historical_legacy.bolt_mode == 'legacy'
    with pytest.raises(TypeError, match='state_memory'):
        BoltConfig(vocab_size=43, state_memory='false')


def test_standalone_bolt_opt_out_and_historical_calls():
    enabled = Bolt(embed_dim=32, num_heads=4, profile='compact', device='cpu', backend='pytorch')
    disabled = Bolt(embed_dim=32, num_heads=4, profile='compact', state_memory=False, device='cpu', backend='pytorch')
    assert enabled.state_memory is True
    assert disabled.state_memory is False
    assert disabled.avq_dim == 0 and disabled._modern.avq is None
    assert not any('.avq.' in k for k in disabled.state_dict())
    assert disabled(torch.randn(2, 5, 32)).shape == (2, 5, 32)

    historical_fast = Bolt(32, 4, mode='fast', device='cpu')
    assert not historical_fast.state_memory
    historical_legacy = Bolt(32, 4, latent_dim=8, device='cpu')
    assert not historical_legacy.is_modern
    with pytest.raises(TypeError, match='state_memory'):
        Bolt(32, 4, profile='standard', state_memory=None, device='cpu')


def test_attention_only_generation_has_no_recurrent_cache():
    model = BoltModel(_config(state_memory=False)).eval()
    ids = torch.randint(0, 43, (1, 5))
    with torch.no_grad():
        tokens = model.generate(ids, max_new_tokens=3, temperature=0)
    assert tokens.shape == (1, 8)
    assert torch.equal(tokens[:, :5], ids)
    assert all(cache.state is None for cache in model._prepared_generation['caches'])
