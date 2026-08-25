from pathlib import Path

import pytest
import torch

from mlbricks import Bolt, Vesa, VisionBolt
from mlbricks.vesa.layers.normalization import PerspectiveNorm
from mlbricks.vision_common import (
    add_sinusoidal_2d_native_or_pytorch,
    apply_scan_native_or_pytorch,
    restore_scan_native_or_pytorch,
    scan_indices,
    sinusoidal_2d_positions,
)


def test_scan_fallback_matches_reference_and_backward():
    x = torch.randn(2, 12, 7, requires_grad=True)
    for phase in range(4):
        y = apply_scan_native_or_pytorch(
            x, 3, 4, scan="cross", layer_index=phase, backend="pytorch"
        )
        order = scan_indices(3, 4, scan="cross", layer_index=phase, device=x.device)
        torch.testing.assert_close(y, x.index_select(1, order))
        z = restore_scan_native_or_pytorch(
            y, 3, 4, scan="cross", layer_index=phase, backend="pytorch"
        )
        torch.testing.assert_close(z, x)
    apply_scan_native_or_pytorch(
        x, 3, 4, scan="cross", layer_index=2, backend="pytorch"
    ).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_native_2d_position_fallback_matches_reference_and_grad():
    x = torch.randn(2, 12, 10, requires_grad=True)
    y = add_sinusoidal_2d_native_or_pytorch(x, 3, 4, backend="pytorch")
    ref = x + sinusoidal_2d_positions(3, 4, 10, device=x.device, dtype=x.dtype)[None]
    torch.testing.assert_close(y, ref)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_perspective_norm_vectorized_matches_historical_groups():
    torch.manual_seed(4)
    norm = PerspectiveNorm(16, 4, backend="pytorch")
    x = torch.randn(3, 5, 16, requires_grad=True)
    parts = x.split(4, dim=-1)
    ref = torch.cat([m(p) for m, p in zip(norm.norms, parts, strict=True)], dim=-1)
    got = norm(x)
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-6)


def test_vision_native_sources_are_packaged_and_setup_has_switch():
    root = Path(__file__).resolve().parents[1]
    assert (root / "mlbricks/vision_csrc/ops.cpp").exists()
    assert (root / "mlbricks/vision_csrc/ops_cpu.cpp").exists()
    assert (root / "mlbricks/vision_csrc/ops_cuda.cu").exists()
    setup = (root / "setup.py").read_text()
    assert "MLBRICKS_BUILD_VISION_NATIVE" in setup
    manifest = (root / "MANIFEST.in").read_text()
    assert "mlbricks/vision_csrc" in manifest


@pytest.mark.parametrize("family", [Vesa, VisionBolt])
@pytest.mark.parametrize("engine", ["Serpentine", "ViT", "CNN"])
def test_classifier_engine_pytorch_backward_survives_native_refactor(family, engine):
    model = family(
        image_size=16, patch_size=4, in_channels=3, num_classes=5,
        dim=32, depth=2, perspective_groups=4, heads=4, latent_dim=8,
        engine=engine, backend="pytorch",
    )
    x = torch.randn(2, 3, 16, 16)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_bolt_pytorch_forward_still_uses_exact_public_math_path():
    torch.manual_seed(0)
    bolt = Bolt(32, 4, latent_dim=8, causal=True, backend="pytorch")
    x = torch.randn(2, 9, 32, requires_grad=True)
    y = bolt(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None


def test_native_backend_requires_extension_when_unavailable():
    from mlbricks import vision_native_available
    if vision_native_available():
        pytest.skip("native extension is installed in this test environment")
    x = torch.randn(1, 6, 4)
    with pytest.raises(RuntimeError, match="vision native extension"):
        apply_scan_native_or_pytorch(
            x, 2, 3, scan="cross", layer_index=0, backend="native"
        )
