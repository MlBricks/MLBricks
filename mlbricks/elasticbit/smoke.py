"""Correctness, storage, loader, and kernel smoke test."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import ElasticBit


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - reference)
        / max(float(np.linalg.norm(reference)), 1.0e-12)
    )


def run_smoke_test(
    *,
    rows: int = 4096,
    cols: int = 1024,
    calibration_samples: int = 4,
    threshold: float = 0.01,
    iterations: int = 500,
    output_dir: str | Path | None = None,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(42)
    weights = np.ascontiguousarray(
        rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(0.08)
    )
    calibration = np.ascontiguousarray(
        rng.standard_normal((calibration_samples, cols), dtype=np.float32)
    )
    input_vector = np.ascontiguousarray(
        rng.standard_normal(cols, dtype=np.float32)
    )

    analysis = ElasticBit.bitsAnaliser(weights, calibration, threshold, 4, 32)
    bits = int(analysis["selected_bits"])
    compact = ElasticBit.RuntimeMatrix(weights, bits, "compact")
    fast = ElasticBit.RuntimeMatrix(weights, bits, "fast")
    fp16 = ElasticBit.NativeFP16Matrix(weights)

    compact_output = np.asarray(compact.forward(input_vector))
    fast_output = np.asarray(fast.forward(input_vector))
    reference_output = np.asarray(compact.forward_reference(input_vector))

    compact_reference_error = _relative_error(compact_output, reference_output)
    fast_compact_error = _relative_error(fast_output, compact_output)
    if bits <= 8 and fast_compact_error > 1.0e-6:
        raise AssertionError(
            f"Fast direct-widening mismatch: {fast_compact_error:.8e}"
        )

    if output_dir is None:
        output = Path(tempfile.mkdtemp(prefix="elasticbit-smoke-"))
    else:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

    model_path = output / f"smoke_{bits}bit.mlb"
    compact.save(str(model_path))
    loaded_compact = ElasticBit.RuntimeMatrix.load(str(model_path), "compact")
    loaded_fast = ElasticBit.RuntimeMatrix.load(str(model_path), "fast")
    load_compact_error = _relative_error(
        np.asarray(loaded_compact.forward(input_vector)), compact_output
    )
    load_fast_error = _relative_error(
        np.asarray(loaded_fast.forward(input_vector)), fast_output
    )

    corrupt_path = output / "corrupt.mlb"
    data = bytearray(model_path.read_bytes())
    data[-1] ^= 1
    corrupt_path.write_bytes(data)
    checksum_pass = False
    try:
        ElasticBit.RuntimeMatrix.load(str(corrupt_path), "compact")
    except RuntimeError:
        checksum_pass = True
    finally:
        corrupt_path.unlink(missing_ok=True)
    if not checksum_pass:
        raise AssertionError("MLB3 checksum validation accepted a corrupt file")

    compact_ms = float(compact.benchmark(input_vector, iterations))
    fast_ms = float(fast.benchmark(input_vector, iterations))
    fp16_ms = float(fp16.benchmark(input_vector, iterations))

    report = {
        "selected_bits": bits,
        "compute_type": str(compact.compute_type),
        "compact_reference_error": compact_reference_error,
        "fast_compact_error": fast_compact_error,
        "load_compact_error": load_compact_error,
        "load_fast_error": load_fast_error,
        "checksum_validation": "PASS",
        "compact_ms": compact_ms,
        "fast_ms": fast_ms,
        "fp16_ms": fp16_ms,
        "fast_vs_compact": compact_ms / fast_ms,
        "fast_vs_fp16": fp16_ms / fast_ms,
        "file_weight_bytes": int(compact.file_weight_bytes),
        "compact_gpu_bytes": int(compact.gpu_weight_bytes),
        "fast_gpu_bytes": int(fast.gpu_weight_bytes),
        "fp16_gpu_bytes": int(fp16.gpu_weight_bytes),
        "model_path": str(model_path),
    }
    return report
