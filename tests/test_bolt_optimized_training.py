"""Numerical and checkpoint contract for the notebook-matched BOLT train route.

The route is CUDA-qualified for production. These tests force it on CPU to
verify the identical v8 computation without requiring an NVIDIA GPU in CI.
"""
import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from mlbricks import BoltConfig, BoltModel
from mlbricks.bolt.presets import BoltPreset


@pytest.mark.parametrize("profile", ["compact", "standard"])
@pytest.mark.parametrize("state_memory", [True, False])
def test_v8_direct_math_matches_reference_forward_gradient_optimizer_and_checkpoint(
    profile, state_memory, monkeypatch
):
    torch.manual_seed(72)
    config = BoltConfig(
        vocab_size=83, context=24, layers=2, dim=48, heads=6,
        profile=profile, state_memory=state_memory, state_dim=198,
        attention_scale="latent", memory_init="balanced",
        device="cpu", precision="fp32", backend="auto", projection="packed",
    )
    base = BoltModel(config).train()
    optimized = copy.deepcopy(base)
    for block in optimized.blocks:
        assert block.attn.is_modern
        monkeypatch.setattr(block.attn._modern, "_optimized_training_available", lambda x: True)
    ids = torch.randint(0, config.vocab_size, (2, 11))
    targets = torch.randint(0, config.vocab_size, (2, 11))

    y0 = base(ids)
    y1 = optimized(ids)
    torch.testing.assert_close(y0, y1, atol=2e-4, rtol=2e-4)
    loss0 = F.cross_entropy(y0.reshape(-1, config.vocab_size), targets.flatten())
    loss1 = F.cross_entropy(y1.reshape(-1, config.vocab_size), targets.flatten())
    loss0.backward()
    loss1.backward()
    for (k0, p0), (k1, p1) in zip(base.named_parameters(), optimized.named_parameters()):
        assert k0 == k1
        assert p0.grad is not None and p1.grad is not None
        torch.testing.assert_close(p0.grad, p1.grad, atol=2e-4, rtol=2e-4)
    o0 = torch.optim.AdamW(base.parameters(), lr=1e-3)
    o1 = torch.optim.AdamW(optimized.parameters(), lr=1e-3)
    o0.step(); o1.step()
    for (k0,p0),(k1,p1) in zip(base.named_parameters(), optimized.named_parameters()):
        assert k0 == k1
        torch.testing.assert_close(p0, p1, atol=2e-4, rtol=2e-4)
    # The packed QUG weights must save using historical separate-projection keys.
    checkpoint=optimized.state_dict()
    assert any(k.endswith('q_proj.weight') for k in checkpoint)
    assert not any(k.endswith('packed_qug_weight') for k in checkpoint)
    restored = BoltModel(config)
    restored.load_state_dict(checkpoint, strict=True)
    restored.eval(); optimized.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(ids), optimized(ids), atol=0, rtol=0)


def test_v8_cuda_qualification_does_not_change_reference_or_eval():
    model=BoltPreset(48,6,cache='tight',projection='packed',backend='auto')
    x=torch.randn(1,8,48)
    assert model._optimized_training_available(x) is False
    model.eval()
    assert model._optimized_training_available(x) is False


def test_v8_shipped_kernel_sources_match_validated_notebook():
    # SHA256 values are taken from the actual source files in the executed
    # 10K-step Kaggle notebook. Their byte identity is the provenance check.
    root=Path(__file__).resolve().parents[1] / 'mlbricks' / 'bolt' / 'native_accel'
    expected={
        'state_short.cu':'1f7941f038f3f4699fdff7f9badb6ab694ed08eef8e7c3657ce4389ff7f5ef4a',
        'state_segmented.cu':'34b4b6719a730c599e45054a0e9b584613b1e6570180b168a9e2384f3c4a198f',
    }
    for name,sha in expected.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha

@pytest.mark.parametrize("profile", ["compact", "standard"])
def test_v8_direct_math_handles_mixed_precision_query_key_dtype(profile, monkeypatch):
    # Under autocast, Q/U/G come from half/BF16 GEMMs while the learned
    # query-multiplier P is a master FP32 parameter. Both paths must cast
    # Q to the shared C key/value dtype before entering SDPA.
    torch.manual_seed(95)
    cfg = BoltConfig(vocab_size=67, context=24, layers=1, dim=48, heads=6,
                     profile=profile, device='cpu', precision='fp32',
                     backend='auto', state_memory=True, projection='packed')
    baseline = BoltModel(cfg).train()
    port = copy.deepcopy(baseline).train()
    monkeypatch.setattr(port.blocks[0].attn._modern,
                        '_optimized_training_available', lambda x: True)
    ids = torch.randint(0,67,(2,9))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        y0=baseline(ids)
        y1=port(ids)
    torch.testing.assert_close(y0.float(), y1.float(), atol=0.025, rtol=0.02)
    assert y0.shape==y1.shape
