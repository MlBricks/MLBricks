from __future__ import annotations

import pytest
import torch

from mlbricks import Bolt, Brick, Bricks, ESA, FFN, StateAwareFFN


def _make_composed_bricks(device: torch.device) -> Bricks:
    return Bricks(
        vocab_size=97,
        dim=32,
        context=16,
        layers=[
            Brick(
                mixer=ESA(embd=32, head=4, precision="fp32", device=None),
                ffn=StateAwareFFN(32),
                dim=32,
            ),
            Brick(
                mixer=Bolt(32, 4, latent_dim=8),
                ffn=FFN(32, 64),
                position="rope",
                dim=32,
            ),
        ],
    ).to(device)


def test_composed_bricks_parent_owned_device_cpu():
    device = torch.device("cpu")
    model = _make_composed_bricks(device).eval()
    ids = torch.randint(0, 97, (1, 6), device=device)

    with torch.no_grad():
        logits = model(ids)

    assert logits.device == device
    assert logits.shape == (1, 6, 97)
    assert all(p.device == device for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for composed-device regression")
def test_composed_bricks_parent_owned_device_cuda():
    device = torch.device("cuda")
    model = _make_composed_bricks(device).eval()
    ids = torch.randint(0, 97, (1, 6), device=device)

    with torch.no_grad():
        logits = model(ids)

    assert logits.is_cuda
    assert logits.shape == (1, 6, 97)
    assert all(p.device.type == "cuda" for p in model.parameters())
