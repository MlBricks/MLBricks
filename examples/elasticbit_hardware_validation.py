"""Validate ElasticBit hardware planning on the current CUDA GPU.

Build the native extension first, then run:
    python examples/elasticbit_hardware_validation.py
"""
from __future__ import annotations

import numpy as np

from mlbricks import ElasticBit


def relativeError(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    denom = max(float(np.linalg.norm(reference)), 1e-12)
    return float(np.linalg.norm(actual - reference) / denom)


def main() -> None:
    backend = ElasticBit.backend()
    print(backend)
    if not backend.available:
        raise RuntimeError("ElasticBit CUDA runtime is not available")

    rng = np.random.default_rng(20260918)
    weights = rng.normal(0.0, 0.04, size=(512, 512)).astype(np.float32)
    calibrationData = rng.normal(0.0, 1.0, size=(16, 512)).astype(np.float32)
    threshold = 0.01

    matrix = ElasticBit.compressMatrix(
        weights,
        calibrationData,
        threshold=threshold,
        decodePolicy="hardwareNative",
    )

    x = calibrationData[0]
    hardware = np.asarray(matrix.forward(x))
    hardwareMs = matrix.benchmark(x, iterations=200)

    print("storageBits      :", matrix.storageBits)
    print("executionWidth   :", matrix.executionWidth)
    print("executionPlanner :", matrix.executionPlanner)
    print("hardwareNative ms:", hardwareMs)

    if backend.validated:
        expected = 4 if matrix.storageBits <= 4 else 8 if matrix.storageBits <= 8 else 16
        assert matrix.executionPlanner == "t4Validated"
        assert matrix.executionWidth == expected

    matrix.setDecodePolicy("fullPrecision")
    full = np.asarray(matrix.forward(x))
    fullMs = matrix.benchmark(x, iterations=200)

    print("fullPrecision ms :", fullMs)
    print("policy relL2     :", relativeError(hardware, full))
    print("threshold        :", matrix.threshold)
    print("selected error   :", matrix.error)

    assert np.isfinite(hardware).all()
    assert np.isfinite(full).all()
    assert matrix.error <= matrix.threshold + 1e-12


if __name__ == "__main__":
    main()
