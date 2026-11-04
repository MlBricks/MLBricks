"""Device and precision ownership when a CUDA-first BOLT is nested in a model."""
import pytest
import torch

from mlbricks import Bolt, BoltConfig, BoltModel, Brick, Bricks, VisionBolt


@pytest.mark.parametrize("engine", ["ViT", "CNN", "Serpentine", "Diffusion", "AR"])
def test_visionbolt_nested_bolt_matches_parent_device(engine):
    cfg = dict(image_size=8, patch_size=4, in_channels=3, dim=16,
               depth=1, heads=4, latent_dim=4, perspective_groups=4,
               engine=engine, backend="pytorch")
    if engine == "AR":
        cfg["vocab_size"] = 23
    elif engine != "Diffusion":
        cfg["num_classes"] = 3
    parent = VisionBolt(**cfg)
    owned = dict(parent.named_modules())
    nested = [(k, v) for k, v in owned.items() if isinstance(v, Bolt)]
    assert nested
    # Children must not auto-migrate independently of their containing model.
    assert all(next(v.parameters()).device == next(parent.parameters()).device
               for _, v in nested)
    assert all(v.precision == "fp32" for _, v in nested)
    parent.to("cpu")
    if engine == "AR":
        y = parent(torch.randint(0, 23, (1, 4)))
    elif engine == "Diffusion":
        y = parent(torch.randn(1, 3, 8, 8), torch.ones(1, dtype=torch.long))
    else:
        y = parent(torch.randn(1, 3, 8, 8))
    assert torch.isfinite(y).all() and y.device.type == "cpu"


def test_bricks_parent_takes_ownership_of_cuda_first_mixer():
    # The constructor can accept a standalone BOLT with its own placement;
    # the parent chooses one device for its embeddings and all layers.
    child = Bolt(32, 4, latent_dim=8, backend="pytorch", device="cpu", precision="fp32")
    parent = Bricks(vocab_size=29, dim=32, context=16,
                    layers=[Brick(mixer=child, dim=32)])
    assert next(parent.embedding.parameters()).device == next(child.parameters()).device
    ids = torch.randint(0, 29, (1, 4))
    out = parent(ids)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("state_memory", [False, True])
def test_boltmodel_preserves_device_and_legacy_precision(state_memory):
    cfg = BoltConfig(vocab_size=41, context=16, layers=1, dim=32, heads=4,
                     profile="compact", state_memory=state_memory,
                     device="cpu", precision="fp32", backend="pytorch")
    model = BoltModel(cfg).eval()
    assert all(v.precision == "fp32" for v in model.modules() if isinstance(v, Bolt))
    ids = torch.randint(0, 41, (1, 5))
    with torch.no_grad():
        entire = model(ids)
        prefix, caches = model.prefill(ids)
    torch.testing.assert_close(entire, prefix)
    assert all(next(v.parameters()).device.type == "cpu"
               for v in model.modules() if isinstance(v, Bolt))


def test_nested_visionbolt_does_not_auto_allocate_cuda_when_gpu_is_visible(monkeypatch):
    # Simulates the Kaggle condition without requiring a CUDA device locally.
    # Standalone Bolt is CUDA-first, whereas VisionBolt owns its children.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = VisionBolt(image_size=8, patch_size=4, in_channels=3,
                       num_classes=3, dim=16, depth=1, heads=4,
                       latent_dim=4, perspective_groups=4,
                       engine="ViT", backend="pytorch")
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(m.precision == "fp32" for m in model.modules()
               if isinstance(m, Bolt))
