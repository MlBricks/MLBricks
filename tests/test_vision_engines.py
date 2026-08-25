import torch
import pytest

from mlbricks import Vesa, VisionBolt, VisionBoltConfig
from mlbricks.vision_common import scan_indices


COMMON = dict(
    image_size=8,
    patch_size=4,
    in_channels=3,
    num_classes=5,
    dim=16,
    depth=2,
    perspective_groups=4,
    heads=4,
    latent_dim=4,
    backend="pytorch",
)


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
@pytest.mark.parametrize("engine", ["Serpentine", "ViT  ", "CNN"])
def test_classifier_engines_forward_backward(cls, engine):
    torch.manual_seed(3)
    model = cls(engine=engine, **COMMON)
    images = torch.randn(2, 3, 8, 8)
    logits = model(images)
    assert logits.shape == (2, 5)
    loss = logits.square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
def test_engine_auto_position_policy(cls):
    serp = cls(engine="Serpentine", **COMMON)
    vit = cls(engine="vision transformer", **COMMON)
    cnn = cls(engine="CNN", **COMMON)
    assert serp.config.position is None
    assert vit.config.position == "2d_sincos"
    assert cnn.config.position is None


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
def test_explicit_none_overrides_vit_auto(cls):
    model = cls(engine="ViT", position=None, **COMMON)
    assert model.config.position is None
    assert model(torch.randn(1, 3, 8, 8)).shape == (1, 5)


def test_bolt_directionality_matches_engine():
    serp = VisionBolt(engine="Serpentine", **COMMON)
    vit = VisionBolt(engine="ViT", **COMMON)
    cnn = VisionBolt(engine="CNN", **COMMON)
    assert all(block.mixer.causal for block in serp.engine_model.blocks)
    assert all(not block.mixer.causal for block in vit.engine_model.blocks)
    assert all(block.mixer.causal for block in cnn.engine_model.blocks)


def test_cross_scan_cycles_four_directions():
    device = torch.device("cpu")
    orders = [
        scan_indices(3, 4, scan="cross", layer_index=i, device=device)
        for i in range(4)
    ]
    assert all(order.numel() == 12 for order in orders)
    assert len({tuple(order.tolist()) for order in orders}) == 4
    base = set(range(12))
    assert all(set(order.tolist()) == base for order in orders)


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
def test_diffusion_engine_forward_backward_and_sample(cls):
    model = cls(engine="Diffusion", **COMMON)
    images = torch.randn(2, 3, 8, 8, requires_grad=True)
    timesteps = torch.tensor([1, 3])
    out = model(images, timesteps)
    assert out.shape == images.shape
    out.square().mean().backward()
    assert images.grad is not None
    with torch.no_grad():
        sample = model.benchmark_sample_loop(torch.randn(1, 3, 8, 8), 2)
    assert sample.shape == (1, 3, 8, 8)


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
def test_ar_engine_forward_backward_and_generate(cls):
    model = cls(engine="AR", vocab_size=37, **COMMON)
    ids = torch.randint(0, 37, (2, 3))
    logits = model(ids)
    assert logits.shape == (2, 3, 37)
    logits.float().square().mean().backward()
    generated = model.generate(ids[:1], 2)
    assert generated.shape == (1, 2)


def test_visionbolt_config_validates_head_shape():
    with pytest.raises(ValueError, match="divisible by heads"):
        VisionBoltConfig(dim=18, heads=4, perspective_groups=3)


@pytest.mark.parametrize("cls", [Vesa, VisionBolt])
def test_invalid_engine_rejected(cls):
    with pytest.raises(ValueError, match="engine must be"):
        cls(engine="unknown", **COMMON)
