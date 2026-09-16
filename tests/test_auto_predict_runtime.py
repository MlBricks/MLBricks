import torch

from mlbricks import Vesa, VisionBolt, predict_module


def _vision_kwargs():
    return dict(
        image_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=3,
        dim=16,
        depth=1,
        heads=2,
        latent_dim=4,
        perspective_groups=2,
        backend="auto",
    )


def test_visionbolt_predict_is_one_call_inference_cpu():
    model = VisionBolt(engine="ViT", **_vision_kwargs())
    x = torch.randn(2, 3, 8, 8)
    y = model.predict(x)
    assert y.shape == (2, 3)
    assert not model.training
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    assert next(model.parameters()).device.type == expected_device
    assert next(model.parameters()).dtype == expected_dtype
    assert model.execution_plan().requested_backend == "auto"


def test_vesa_predict_can_skip_calibration_and_still_auto_place():
    model = Vesa(engine="Serpentine", **_vision_kwargs())
    x = torch.randn(1, 3, 8, 8)
    y = model.predict(x, calibrate=False)
    assert y.shape == (1, 3)
    assert not model.training
    expected_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else x.device
    assert next(model.parameters()).device == expected_device


def test_predict_module_public_helper_matches_model_predict_without_calibration():
    m1 = VisionBolt(engine="ViT", **_vision_kwargs())
    m2 = VisionBolt(engine="ViT", **_vision_kwargs())
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(1, 3, 8, 8)
    y1 = m1.predict(x, calibrate=False)
    y2 = predict_module(m2, x, calibrate=False)
    torch.testing.assert_close(y1, y2)


def test_visual_ar_predict_preserves_integer_token_dtype():
    model = VisionBolt(
        engine="AR",
        vocab_size=32,
        **_vision_kwargs(),
    )
    ids = torch.randint(0, 32, (2, 5), dtype=torch.long)
    y = model.predict(ids, calibrate=False)
    assert y.shape == (2, 5, 32)
    assert ids.dtype == torch.long


def test_predict_can_force_cpu_even_when_cuda_is_available():
    model = VisionBolt(engine="ViT", **_vision_kwargs())
    x = torch.randn(1, 3, 8, 8)
    y = model.predict(x, device="cpu", calibrate=False)
    assert y.shape == (1, 3)
    assert y.device.type == "cpu"
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32
