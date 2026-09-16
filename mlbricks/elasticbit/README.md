# ElasticBit

ElasticBit is a CUDA runtime for **adaptive 4–32-bit matrix weight storage**. It analyzes each matrix against calibration inputs, selects the smallest storage width that satisfies an error threshold, and executes the matrix through a compute path appropriate to the selected precision.

## API

ElasticBit exposes one public namespace:

```python
from elasticbit import ElasticBit
```

The primary API is:

```python
ElasticBit.bitsAnaliser(...)
ElasticBit.RuntimeMatrix(...)
ElasticBit.NativeFP16Matrix(...)
```

When integrated into MLBricks, the same API is intended to be available as:

```python
from mlbricks import ElasticBit
```

## Precision range

ElasticBit supports storage widths from **4 through 32 bits**.

| Storage width | Runtime compute |
|---|---|
| 4-bit | INT4 |
| 5–8-bit | INT8 |
| 9–16-bit | FP16 |
| 17–32-bit | FP32 |

- 16-bit uses native FP16 storage.
- 32-bit uses native FP32 storage.
- Intermediate widths use exact packed integer storage with row-wise symmetric scales.
- Widths above 16 bits are higher-precision fallback options and can use more storage than an FP16 baseline.

## Quick start

```python
import numpy as np
from elasticbit import ElasticBit

rng = np.random.default_rng(42)
weights = np.ascontiguousarray(
    rng.standard_normal((4096, 1024), dtype=np.float32) * 0.08
)
calibration = np.ascontiguousarray(
    rng.standard_normal((32, 1024), dtype=np.float32)
)
x = np.ascontiguousarray(
    rng.standard_normal(1024, dtype=np.float32)
)

analysis = ElasticBit.bitsAnaliser(
    weights,
    calibration,
    threshold=0.01,
    min_bits=4,
    max_bits=32,
)

bits = int(analysis["selected_bits"])
matrix = ElasticBit.RuntimeMatrix(weights, bits, "compact")
y = matrix.forward(x)
```

## `ElasticBit.bitsAnaliser`

Analyzes a matrix against calibration inputs and selects the smallest bit width whose measured error satisfies the requested threshold.

```python
analysis = ElasticBit.bitsAnaliser(
    weights,
    calibration,
    threshold=0.01,
    min_bits=4,
    max_bits=32,
)
```

Inputs:

- `weights`: contiguous `float32` NumPy array with shape `[rows, cols]`
- `calibration`: contiguous `float32` NumPy array with shape `[samples, cols]`
- `threshold`: maximum accepted calibration error
- `min_bits`: lowest width to test, default `4`
- `max_bits`: highest width to test, default `32`

The returned dictionary contains:

- `selected_bits`
- `selected_error`
- `selected_compute_type`
- `analyses`

Each item in `analyses` reports the tested width, compute type, measured error, payload bytes, scale bytes, runtime bytes, and storage reduction relative to FP16.

## `ElasticBit.RuntimeMatrix`

Creates and executes an ElasticBit matrix.

```python
matrix = ElasticBit.RuntimeMatrix(
    weights,
    storage_bits=bits,
    runtime_mode="compact",
)
```

Runtime modes:

- `compact` — keeps the exact packed representation on the GPU.
- `fast` — widens the representation to the selected compute bucket for execution.

### Automatic construction

```python
matrix = ElasticBit.RuntimeMatrix.from_auto(
    weights,
    calibration,
    threshold=0.01,
    runtime_mode="compact",
    min_bits=4,
    max_bits=32,
)
```

### Inference

```python
y = matrix.forward(x)
```

### Reference output

```python
reference = matrix.forward_reference(x)
```

### Benchmark

```python
milliseconds = matrix.benchmark(x, iterations=500)
```

### Reconstruct weights

```python
weights_fp32 = matrix.dequantize()
```

### Save and load

```python
matrix.save("projection.mlb")
loaded = ElasticBit.RuntimeMatrix.load("projection.mlb", "fast")
```

### Matrix properties

```python
matrix.rows
matrix.cols
matrix.storage_bits
matrix.compute_type
matrix.runtime_mode
matrix.file_payload_bytes
matrix.file_scale_bytes
matrix.file_weight_bytes
matrix.gpu_weight_bytes
matrix.fp16_weight_bytes
matrix.file_reduction_vs_fp16
matrix.gpu_reduction_vs_fp16
```

## `ElasticBit.NativeFP16Matrix`

Native FP16 CUDA baseline for speed and memory comparison.

```python
baseline = ElasticBit.NativeFP16Matrix(weights)
y = baseline.forward(x)
ms = baseline.benchmark(x, 500)
print(baseline.gpu_weight_bytes)
```

## Real-model export

```python
from elasticbit.real_model import export_and_benchmark_model

report = export_and_benchmark_model(
    model=model,
    calibration_batches=calibration_batches,
    validation_batches=validation_batches,
    output_dir="./elasticbit_export",
    device=next(model.parameters()).device,
    threshold=0.01,
    min_bits=4,
    max_bits=32,
    calibration_rows_per_layer=32,
    benchmark_iterations=500,
)
```

The exporter analyzes each `torch.nn.Linear` matrix independently, saves ElasticBit matrices, reconstructs weights for quality evaluation, and compares compact/fast ElasticBit execution against native FP16.

## Smoke test

```python
from elasticbit.smoke import run_smoke_test

report = run_smoke_test(iterations=2000)
print(report)
```

## Installation

ElasticBit currently targets **Linux + NVIDIA CUDA** and requires `nvcc` when building from source.

```bash
git clone https://github.com/MlBricks/ElasticBit.git
cd ElasticBit
pip install --no-cache-dir .
```

For a specific GPU architecture:

```bash
ELASTICBIT_CUDA_ARCHS="75" pip install .
```

Multiple architectures can be supplied:

```bash
ELASTICBIT_CUDA_ARCHS="75;80;86;89;90" pip install .
```

## License

ElasticBit is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use requires a separate written commercial license. See `LICENSE` for the complete required notices and ownership terms.
