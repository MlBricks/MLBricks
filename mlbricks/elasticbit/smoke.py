"""Native ElasticBit smoke test for CUDA source builds."""
from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np

from . import ElasticBit


def run() -> None:
    rng = np.random.default_rng(7)
    weights = rng.normal(0, 0.05, size=(64, 128)).astype(np.float32)
    calibrationData = rng.normal(0, 1, size=(8, 128)).astype(np.float32)
    threshold = 0.02

    analysis = ElasticBit.analyze(weights, calibrationData, threshold)
    matrix = ElasticBit.compressMatrix(weights, calibrationData, threshold)
    x = calibrationData[0]
    hardware = np.asarray(matrix.forward(x))

    matrix.setDecodePolicy("fullPrecision")
    full = np.asarray(matrix.forward(x))
    rel = np.linalg.norm(hardware - full) / max(np.linalg.norm(full), 1e-12)

    with tempfile.TemporaryDirectory() as tempDir:
        path = Path(tempDir) / "smoke.mlb"
        matrix.save(path)
        loaded = ElasticBit.loadMatrix(path, decodePolicy="hardwareNative")
        loadedOut = np.asarray(loaded.forward(x))
        if ElasticBit.backend().validated:
            np.testing.assert_allclose(loadedOut, hardware, rtol=0, atol=0)
        else:
            reloadRel = np.linalg.norm(loadedOut - hardware) / max(np.linalg.norm(hardware), 1e-12)
            assert reloadRel <= 2.0 * threshold + 1e-3

    print("backend:", ElasticBit.backend())
    print("selectedBits:", analysis.selectedBits)
    print("executionWidth:", matrix.executionWidth)
    print("executionPlanner:", matrix.executionPlanner)
    print("hardware/full relL2:", float(rel))
    print("PASS")


if __name__ == "__main__":
    run()
