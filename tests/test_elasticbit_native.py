from __future__ import annotations

import numpy as np
import pytest
import torch

from mlbricks import ElasticBit


pytestmark = pytest.mark.skipif(
    not ElasticBit.backend().available,
    reason="ElasticBit native CUDA extension is unavailable",
)


def _rel(actual, reference):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-12))


def test_threshold_contract_and_policy_switch(tmp_path):
    rng = np.random.default_rng(20260918)
    weights = rng.normal(0, 0.04, size=(96, 128)).astype(np.float32)
    calibrationData = rng.normal(0, 1, size=(12, 128)).astype(np.float32)
    threshold = 0.02

    analysis = ElasticBit.analyze(weights, calibrationData, threshold)
    assert 3 <= analysis.selectedBits <= 16
    assert analysis.selectedError <= threshold + 1e-12

    matrix = ElasticBit.compressMatrix(weights, calibrationData, threshold)
    assert matrix.storageBits == analysis.selectedBits
    assert matrix.activateDType == "float16"
    backend = ElasticBit.backend()
    if backend.validated:
        assert matrix.executionPlanner == "t4Validated"
        expected = 4 if matrix.storageBits <= 4 else 8 if matrix.storageBits <= 8 else 16
        assert matrix.executionWidth == expected
    else:
        assert matrix.executionPlanner == "cudaAutoTune"
        legal = ({4, 8, 16} if matrix.storageBits <= 4 else {8, 16} if matrix.storageBits <= 8 else {16})
        assert matrix.executionWidth in legal

    x = calibrationData[0]
    hardware = np.asarray(matrix.forward(x))
    matrix.setDecodePolicy("fullPrecision")
    full = np.asarray(matrix.forward(x))
    assert _rel(hardware, full) <= 2.0 * threshold + 1e-3

    path = tmp_path / "matrix.mlb"
    matrix.setDecodePolicy("hardwareNative")
    matrix.save(path)
    loaded = ElasticBit.loadMatrix(path, decodePolicy="hardwareNative")
    loadedOut = np.asarray(loaded.forward(x))
    if ElasticBit.backend().validated:
        # T4 uses the frozen validated mapping, so reload is bitwise-stable.
        np.testing.assert_array_equal(loadedOut, hardware)
    else:
        # Auto-tune may legally choose a different width after reload; both
        # paths remain bounded by the compression threshold contract.
        assert _rel(loadedOut, hardware) <= 2.0 * threshold + 1e-3


def test_model_compress_switch_and_artifact(tmp_path):
    torch.manual_seed(20260918)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 96),
        torch.nn.GELU(),
        torch.nn.Linear(96, 32),
    ).cuda().half().eval()

    calibrationData = [
        torch.randn(4, 64, device="cuda", dtype=torch.float16)
        for _ in range(2)
    ]
    threshold = 0.03
    ElasticBit.compress(model, calibrationData, threshold)

    # Multi-row input exercises the native/vendor prefill path.
    prefillX = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    prefillY = model(prefillX)
    assert prefillY.shape == (4, 32)
    assert torch.isfinite(prefillY).all()

    x = torch.randn(1, 64, device="cuda", dtype=torch.float16)
    model.elasticbit.setDecodePolicy("hardwareNative")
    hardware = model(x).detach()
    model.elasticbit.setDecodePolicy("fullPrecision")
    full = model(x).detach()
    assert torch.nn.functional.cosine_similarity(
        hardware.float(), full.float(), dim=-1
    ).item() > 0.999

    artifact = tmp_path / "model.elasticbit"
    ElasticBit.save(model, artifact)

    shell = torch.nn.Sequential(
        torch.nn.Linear(64, 96),
        torch.nn.GELU(),
        torch.nn.Linear(96, 32),
    ).cuda().half().eval()
    ElasticBit.load(shell, artifact, decodePolicy="fullPrecision")
    restored = shell(x).detach()
    torch.testing.assert_close(restored, full, atol=0, rtol=0)
