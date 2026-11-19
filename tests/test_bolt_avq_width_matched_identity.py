"""AVQ width-matching contract for the BOLT public API."""

import pytest
import torch
from torch import nn

from mlbricks import Bolt, BoltConfig, BoltModel, BoltStateMemory


@pytest.mark.parametrize("mode", ["quantised", "keyless", "base"])
@pytest.mark.parametrize("heads", [4, 12])
def test_public_bolt_defaults_match_full_attention_width(mode, heads):
    torch.manual_seed(1)
    module = Bolt(d_model=48, num_heads=heads, mode=mode,
                  device="cpu", backend="pytorch", precision="fp32")
    full_attention_width = 48
    latent = 48 // heads
    assert module.latent_dim == latent
    assert module.avq_dim == full_attention_width
    assert module._modern.avq.width == full_attention_width
    assert isinstance(module._modern.avq.back, nn.Identity)
    assert not any("avq.back.weight" in name for name in module.state_dict())
    assert module._modern.avq.norm.eps == torch.finfo(torch.float32).eps
    assert not hasattr(module._modern, "p")  # direct-Q configuration

    with torch.no_grad():
        module._modern.avq.gamma.fill_(0.1)
    inp = torch.randn(1, 5, 48, requires_grad=True)
    module.train()
    out = module(inp)
    out.square().mean().backward()
    assert inp.grad is not None and torch.isfinite(inp.grad).all()
    assert module._modern.avq.project.weight.grad is not None

    module.eval()
    with torch.inference_mode():
        expected = module(inp.detach())
        _, cache = module.prefill_with_cache(inp.detach()[:, :4])
        last, _ = module.decode_step(inp.detach()[:, 4:], cache)
    torch.testing.assert_close(last, expected[:, 4:], atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("mode", ["quantised", "keyless", "base"])
def test_bolt_model_config_default_matches_width(mode):
    config = BoltConfig(vocab_size=47, context=8, layers=1, dim=48,
                        heads=12, bolt_mode=mode, device="cpu",
                        precision="fp32", backend="pytorch")
    assert config.avq_dim == 48
    model = BoltModel(config)
    attention = model.blocks[0].attn
    assert attention.avq_dim == 48
    assert isinstance(attention._modern.avq.back, nn.Identity)
    assert not any("avq.back.weight" in name for name in model.state_dict())


def test_standalone_memory_default_to_identity():
    state = BoltStateMemory(input_dim=384, device="cpu", backend="pytorch")
    assert state.width == 384
    assert isinstance(state.back, nn.Identity)


def test_explicit_nonmatching_width_remains_a_deliberate_ablation():
    module = Bolt(d_model=48, num_heads=12, mode="quantised",
                  avq_dim=194, device="cpu", backend="pytorch")
    assert module.avq_dim == 194
    assert isinstance(module._modern.avq.back, nn.Linear)
    assert module._modern.avq.back.weight.shape == (48, 194)
    with pytest.raises(ValueError, match="identity AVQ return"):
        BoltStateMemory(48, 194, return_map="identity", device="cpu")
