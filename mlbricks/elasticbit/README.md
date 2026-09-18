# ElasticBit

ElasticBit is threshold-driven adaptive weight compression for CUDA inference.

The public API has no fixed 4-bit or 8-bit mode. The user supplies representative calibration data and an allowed relative error threshold. ElasticBit tests storage widths from 3 through 15 bits and chooses the smallest width that satisfies the threshold under both supported decode policies. If none pass, ElasticBit stores FP16 instead of violating the threshold.

Activations remain FP16 throughout execution.

## Execution policies

- `hardwareNative`: hardware-aware decode. On the validated Tesla T4 path, stored 3–4 bit -> W4A16, 5–8 bit -> W8A16, and 9–16 bit -> W16A16. M=1 uses the validated FastWarp topology: one warp owns one output row and an 8-warp block computes up to eight rows without inter-warp reduction. On other CUDA GPUs, ElasticBit benchmarks the legal built-in W4A16/W8A16/W16A16 candidates for each matrix shape.
- `fullPrecision`: all stored widths execute as W16A16. M=1 uses the same FP16 FastWarp topology, which keeps the policy comparison on the same decode work mapping.

The compressed storage does not change when switching policies. Execution width is a runtime decision, not a property of the stored file.

## Whole-model API

```python
from mlbricks import ElasticBit

model = model.cuda().half().eval()
model = ElasticBit.compress(
    model,
    calibrationData=calibrationData,
    threshold=0.01,
)

model.elasticbit.summary()
model.elasticbit.setDecodePolicy("fullPrecision")
model.elasticbit.setDecodePolicy("hardwareNative")
```

Prefill expands the compressed weight directly on the current GPU to a transient FP16 tensor, then uses the framework/vendor `Linear` GEMM path; this behavior is intentionally unchanged. M=1 decode uses the ElasticBit FastWarp runtime. No activation quantization is used. During whole-model compression, analyzer-selected integer codes and row scales are packed directly into canonical storage rather than recomputing quantization a second time.

## Matrix API

```python
analysis = ElasticBit.analyze(weights, calibrationData, threshold=0.01)
print(analysis.selectedBits, analysis.selectedError)

matrix = ElasticBit.compressMatrix(weights, calibrationData, threshold=0.01)
y = matrix.forward(x)

matrix.setDecodePolicy("fullPrecision")
matrix.setDecodePolicy("hardwareNative")

matrix.save("projection.mlb")
matrix = ElasticBit.loadMatrix("projection.mlb")
```

Useful matrix properties:

```python
matrix.storageBits
matrix.executionWidth
matrix.executionPlanner
matrix.decodePolicy
matrix.activateDType
matrix.storageBytes
matrix.executionBytes
matrix.originalBytes
matrix.memoryReduction
matrix.error
matrix.threshold
```

## Model artifacts

```python
ElasticBit.save(model, "model.elasticbit")
model = ElasticBit.load(modelArchitecture, "model.elasticbit")
```

The architecture instance is required when loading because ElasticBit stores model state and compressed matrices, not arbitrary executable Python class code.

## Backend

```python
print(ElasticBit.backend())
```

The backend reports the detected CUDA architecture, native prefill policy, and decode planner. T4 reports `t4Validated`; other CUDA devices report `cudaAutoTune`. Auto-tuning selects the fastest legal built-in ElasticBit decode width for the actual matrix shape. This optimizes among ElasticBit's available kernels; architecture-specific kernels can still be added later without changing the compressed format.

## File format

The clean runtime writes MLB4 matrix files. MLB4 records exact adaptive storage, threshold/error metadata, checksums, and dimensions. Decode policy is deliberately not persisted as storage semantics; it can be selected again at load time.

## License

ElasticBit is licensed under the PolyForm Noncommercial License 1.0.0. Commercial use requires a separate written commercial license.
