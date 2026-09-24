from __future__ import annotations

import numpy as np
import pytest
import torch

from mlbricks import ElasticBit, RuntimeMatrix


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



def test_model_analyzer_result_reuse_and_fastwarp_policy(tmp_path):
    """Compressed/full-precision policies share selected math; M=1 stays stable."""
    torch.manual_seed(20260918)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 128, bias=False),
    ).cuda().half().eval()
    calibrationData = [
        torch.randn(8, 128, device="cuda", dtype=torch.float16)
    ]
    ElasticBit.compress(model, calibrationData, threshold=0.03)

    layer = model[0]
    matrix = layer.matrix
    assert matrix.activateDType == "float16"
    assert 3 <= matrix.storageBits <= 16

    x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    model.elasticbit.setDecodePolicy("fullPrecision")
    fp16_fastwarp = model(x).detach()
    model.elasticbit.setDecodePolicy("hardwareNative")
    hardware_fastwarp = model(x).detach()

    # Policies execute the same selected mathematical model; hardwareNative can
    # differ slightly because integer accumulation and FP16 FastWarp round in a
    # different order, but should stay tightly aligned.
    assert torch.nn.functional.cosine_similarity(
        hardware_fastwarp.float(), fp16_fastwarp.float(), dim=-1
    ).item() > 0.999

    artifact = tmp_path / "fastwarp.elasticbit"
    ElasticBit.save(model, artifact)
    assert artifact.exists()

def test_t4_fastwarp_production_backend_contract():
    """Exact Tesla T4 / SM75 reports the frozen production FastWarp contract."""
    backend = ElasticBit.backend()
    if not backend.validated:
        pytest.skip("production FastWarp contract is specific to the validated Tesla T4")

    assert "T4" in backend.device.upper()
    assert backend.architecture == "sm75"
    assert backend.decodePlanner == "t4Validated"
    assert backend.decodeEngine == "FastWarp"
    assert backend.decodeStatus == "production"
    assert backend.decodeValidation == "fastwarp_t4_sm75_m1_w4w8w16_v1"
    assert backend.prefill == "native"


@pytest.mark.parametrize(
    ("storage_bits", "expected_width"),
    [
        (3, 4),
        (4, 4),
        (5, 8),
        (6, 8),
        (7, 8),
        (8, 8),
        (9, 16),
        (10, 16),
        (15, 16),
    ],
)
def test_t4_fastwarp_production_family_mapping(storage_bits, expected_width):
    """The validated T4 path must never widen outside the frozen storage family map."""
    backend = ElasticBit.backend()
    if not backend.validated:
        pytest.skip("production FastWarp family mapping is specific to the validated Tesla T4")

    rows, cols = 8, 32
    codes = np.zeros((rows, cols), dtype=np.int16)
    scales = np.ones((rows,), dtype=np.float32)
    matrix = RuntimeMatrix._compressSelectedCodes(
        codes,
        scales,
        storage_bits,
        threshold=0.01,
        selectedError=0.0,
        decodePolicy="hardwareNative",
    )

    assert matrix.executionPlanner == "t4Validated"
    assert matrix.executionWidth == expected_width
    assert matrix.decodePolicy == "hardwareNative"

    x = torch.randn(1, cols, device="cuda", dtype=torch.float16)
    y = matrix.forward(x)
    assert y.shape == (1, rows)
    assert torch.isfinite(y).all()



@pytest.mark.parametrize("storage_bits", [3, 4, 5, 7, 8, 9, 15])
@pytest.mark.parametrize("rows,cols", [(1, 1), (7, 13), (8, 31), (17, 32), (33, 65)])
def test_t4_fastwarp_model_agnostic_shape_contract(storage_bits, rows, cols):
    """FastWarp correctness is a matrix contract, not a model-name contract."""
    backend = ElasticBit.backend()
    if not backend.validated:
        pytest.skip("requires production Tesla T4 / SM75 FastWarp backend")

    # Use valid signed integer codes for each storage width.  The runtime expands
    # them into the fixed W4/W8/W16 execution family selected by storage bits.
    qmax = (1 << (storage_bits - 1)) - 1
    low = -max(qmax, 1)
    high = max(qmax, 1)
    rng = np.random.default_rng(rows * 1000 + cols * 10 + storage_bits)
    codes = rng.integers(low, high + 1, size=(rows, cols), dtype=np.int16)
    scales = rng.uniform(0.002, 0.05, size=(rows,)).astype(np.float32)

    matrix = RuntimeMatrix._compressSelectedCodes(
        codes, scales, storage_bits, threshold=0.01, selectedError=0.0,
        decodePolicy="hardwareNative",
    )
    x = torch.randn(1, cols, device="cuda", dtype=torch.float16)
    y = matrix.forward(x)

    weight = matrix._materializeTorch()
    reference = torch.nn.functional.linear(x, weight)
    assert y.shape == (1, rows)
    assert torch.isfinite(y).all()
    assert torch.nn.functional.cosine_similarity(
        y.float(), reference.float(), dim=-1
    ).item() > 0.999
