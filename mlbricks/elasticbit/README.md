# ElasticBit

ElasticBit is threshold-driven adaptive weight compression for CUDA inference.

Users provide representative calibration data and an allowed error threshold. ElasticBit searches compressed storage widths from **3 through 15 bits** and chooses the smallest width that satisfies the threshold. If no compressed width is safe, ElasticBit falls back to **FP16 storage** rather than violating the requested threshold.

ElasticBit is **weight-only**: activations remain FP16 in every supported execution policy.

## Core model

ElasticBit deliberately separates **storage precision** from **execution width**.

A matrix may be stored at an exact adaptive width such as 6, 7, 8, 9, or 10 bits while the hardware backend selects the fastest supported execution kernel for that device.

On the validated Tesla T4 path:

| Stored width | Decode execution |
| --- | --- |
| 3–4 bit | W4A16 FastWarp |
| 5–8 bit | W8A16 FastWarp |
| 9–16 bit | W16A16 FastWarp |

For example:

```text
stored 6-bit  -> W8A16 execution
stored 7-bit  -> W8A16 execution
stored 8-bit  -> W8A16 execution
stored 9-bit  -> W16A16 execution
stored 10-bit -> W16A16 execution
```

The canonical compressed representation does **not** change when the execution policy changes. Storage stays hardware-neutral; execution is selected at runtime.

## Execution policies

ElasticBit exposes two decode policies.

### `hardwareNative`

Uses the hardware planner to select an execution width compatible with the stored matrix.

On the validated Tesla T4 backend:

```text
3–4 bit  -> W4A16
5–8 bit  -> W8A16
9–16 bit -> W16A16
```

For `M=1` decode, W4A16, W8A16, and W16A16 use the FastWarp topology:

```text
1 warp       -> 1 output row
8-warps/block -> up to 8 output rows
```

There is no inter-warp reduction for an output row.

On other supported CUDA GPUs, ElasticBit can benchmark legal W4A16/W8A16/W16A16 execution candidates per matrix shape and select the runtime width independently from storage.

### `fullPrecision`

Executes every stored width through W16A16 while keeping the same selected mathematical model.

For `M=1`, `fullPrecision` uses the FP16 FastWarp kernel so policy comparisons use the same one-warp-per-output-row work mapping.

Switching policy does **not** require recalibration or recompression.

## Tesla T4 FastWarp production status

The exact Tesla T4 / CUDA `sm75` `hardwareNative` decode route is a
**production-validated FastWarp path**.

The production validation used Granite 4.2 8B and exercised all 40 decoder
layers, covering all 280 ElasticBit projection matrices (`q`, `k`, `v`, `o`,
`gate`, `up`, and `down`) with `M=1` decode.

Measured summed projection latency on the physical Tesla T4:

| Route | 280-projection latency |
| --- | ---: |
| `hardwareNative` FastWarp | 70.163 ms |
| `fullPrecision` W16 FastWarp | 96.004 ms |
| resident FP16 `F.linear` | 195.182 ms |
| generic materialize + `F.linear` | 381.871 ms |

This corresponds to:

- **5.443x** FastWarp speedup versus the generic materialize + `F.linear` fallback;
- **2.782x** versus resident FP16 `F.linear`;
- **1.368x** versus forced W16 FastWarp.

Numerical agreement in the same sweep remained tight:

- minimum `hardwareNative` vs `fullPrecision` cosine: **0.999999821**;
- maximum `hardwareNative` vs `fullPrecision` relative L2: **0.0003664**.

These are **projection-level decode measurements**, not end-to-end model
tokens/second. Prefill, attention-cache work, normalization, embeddings, and
the LM head are outside this specific FastWarp confirmation.

`ElasticBit.backend()` reports this route as:

```text
decodeEngine     = FastWarp
decodeStatus     = production
decodePlanner    = t4Validated
decodeValidation = granite42-8b_t4_40x7_20260923
```

Other CUDA GPUs keep the existing auto-tuned decode planner and are reported
as `experimental` until separately validated.

## Prefill and decode

ElasticBit intentionally uses different paths for prefill and single-token decode:

```text
M > 1
compressed weight
    -> transient FP16 materialization
    -> framework/vendor Linear GEMM

M = 1
compressed runtime matrix
    -> ElasticBit FastWarp decode kernel
```

Prefill remains on the framework/vendor FP16 GEMM path.

No W4A4/W8A8 activation quantization is used.

## Public API

```python
ElasticBit.compress(...)
ElasticBit.analyze(...)
ElasticBit.compressMatrix(...)
ElasticBit.loadMatrix(...)
ElasticBit.save(...)
ElasticBit.load(...)
ElasticBit.backend()
ElasticBit.optimize(...)
```

Compressed models also expose:

```python
model.elasticbit.summary()
model.elasticbit.setDecodePolicy(...)
```

A single `RuntimeMatrix` exposes:

```python
matrix.forward(...)
matrix.benchmark(...)
matrix.dequantize()
matrix.setDecodePolicy(...)
matrix.save(...)
```

## Whole-model compression

```python
from mlbricks import ElasticBit

model = model.cuda().half().eval()

model = ElasticBit.compress(
    model,
    calibrationData=calibrationData,
    threshold=0.01,
    decodePolicy="hardwareNative",
)

model.elasticbit.summary()
```

Change decode policy without changing the compressed storage:

```python
model.elasticbit.setDecodePolicy("fullPrecision")
model.elasticbit.setDecodePolicy("hardwareNative")
```

Run the validated Stage-2F T4 planner explicitly when desired:

```python
model = ElasticBit.optimize(model)

# Ignore a matching cached profile and benchmark again.
model = ElasticBit.optimize(model, force=True)
```

Stage-2F v1 intentionally matches the validated one-layer Granite 4.2 8B /
Tesla T4 research notebook. It does not silently generalize its candidate set,
sequence lengths, safety margins, model shape, or GPU support. A matching profile
can be saved inside the model artifact and reused without benchmarking again.

Whole-model compression captures representative activations from the supplied calibration data, selects a safe width for each eligible Linear weight, and directly packs the analyzer-selected integer codes and row scales into canonical ElasticBit storage.

The current whole-model CUDA path requires FP16 Linear weights.

## Analyze without compressing

Use `ElasticBit.analyze()` to inspect adaptive selection for one matrix:

```python
analysis = ElasticBit.analyze(
    weights,
    calibrationData,
    threshold=0.01,
)

print(analysis.threshold)
print(analysis.selectedBits)
print(analysis.selectedError)
print(analysis.executionWidth)
print(analysis.candidates)
```

Candidate information includes the candidate storage width, policy-specific errors, combined error, storage bytes, memory reduction, and whether the candidate satisfies the requested threshold.

## Single-matrix API

```python
matrix = ElasticBit.compressMatrix(
    weights,
    calibrationData,
    threshold=0.01,
    decodePolicy="hardwareNative",
)

y = matrix.forward(x)
```

Switch execution policy:

```python
matrix.setDecodePolicy("fullPrecision")
matrix.setDecodePolicy("hardwareNative")
```

Benchmark the matrix runtime:

```python
milliseconds = matrix.benchmark(x, iterations=500)
```

Recover the selected matrix as a dequantized NumPy array when needed:

```python
weights_fp32 = matrix.dequantize()
```

### RuntimeMatrix properties

```python
matrix.rows
matrix.cols
matrix.shape

matrix.storageBits
matrix.executionWidth
matrix.decodePolicy
matrix.activateDType

matrix.storageBytes
matrix.executionBytes
matrix.originalBytes
matrix.memoryReduction

matrix.error
matrix.threshold
```

`storageBits` describes the canonical stored representation.

`executionWidth` describes the hardware execution width currently selected for decode.

These values are intentionally allowed to differ.

## Model artifacts

Save a compressed model:

```python
ElasticBit.save(model, "model.elasticbit")
```

Load it into an instance of the original model architecture:

```python
modelArchitecture = modelArchitecture.cuda().half().eval()

model = ElasticBit.load(
    modelArchitecture,
    "model.elasticbit",
    decodePolicy="hardwareNative",
)
```

The architecture instance is required because an ElasticBit model artifact stores compressed matrices and model state, not arbitrary executable Python class definitions.

## Matrix artifacts

```python
matrix.save("projection.mlb")

matrix = ElasticBit.loadMatrix(
    "projection.mlb",
    decodePolicy="hardwareNative",
)
```

ElasticBit matrix artifacts use the **MLB4** format.

MLB4 records:

- exact adaptive storage width;
- packed weight payload;
- per-row scales;
- matrix dimensions;
- threshold and selected error metadata;
- integrity/checksum information.

Decode policy is runtime metadata, not part of the stored mathematical representation, so a matrix can be loaded with a different decode policy without recompression.

## Backend inspection

```python
backend = ElasticBit.backend()
print(backend)
```

Depending on the installed native runtime, backend information can include:

```python
backend.available
backend.device
backend.architecture
backend.weightWidths
backend.activateDType
backend.prefill
backend.decodePlanner
backend.decodeEngine
backend.decodeStatus
backend.decodeValidation
backend.validated
```

For the validated Tesla T4 path the architecture is CUDA `sm75`; the decode engine reports `FastWarp`, the status reports `production`, and the planner remains `t4Validated`.

## Native CUDA build

The native ElasticBit runtime is optional at package-build time and is required for CUDA compression/execution.

Build/install with:

```bash
MLBRICKS_BUILD_ELASTICBIT_NATIVE=1 pip install .
```

For editable development:

```bash
MLBRICKS_BUILD_ELASTICBIT_NATIVE=1 pip install -e . --no-build-isolation
```

## Migration from older ElasticBit APIs

The current public API is threshold-driven only.

Do not use older fixed-bit examples such as:

```python
ElasticBit(bits=4)
ElasticBitConfig(...)
ElasticBit.bitsAnaliser(...)
ElasticBit.RuntimeMatrix(weights, selected_bits, "compact")
```

Manual public bit-width selection, `bitsAnaliser`, `RuntimeMatrix.from_auto`, activation-quantized W4A4/W8A8 execution, and the older `runtime` / `runtime_mode` modes are not part of the current public API.

Use:

```python
analysis = ElasticBit.analyze(
    weights,
    calibrationData,
    threshold=0.01,
)

matrix = ElasticBit.compressMatrix(
    weights,
    calibrationData,
    threshold=0.01,
)
```

## Performance results

Benchmark numbers are intentionally not included in this README yet. Publish performance claims only from the final reproducible benchmark configuration for the target hardware.

## License

ElasticBit is licensed under the **PolyForm Noncommercial License 1.0.0**.

Commercial use requires a separate written commercial license.
