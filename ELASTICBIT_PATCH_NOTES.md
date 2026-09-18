# ElasticBit FastWarp production patch — 2026-09-18

- Keeps prefill unchanged on the existing FP16 framework/vendor GEMM path.
- Replaces M=1 W4A16/W8A16/W16A16 decode topology with one warp per output row (8 rows per 256-thread block).
- `fullPrecision` M=1 decode now uses FP16 FastWarp rather than ATen/cuBLAS, matching the clean Granite benchmark.
- `hardwareNative` <=8-bit matrices use W4/W8 FastWarp; >8-bit matrices use W16 FastWarp.
- Reuses analyzer-selected integer codes and row scales when building canonical 3..15-bit storage, removing duplicate quantization and the original FP16 GPU->CPU copy for compressed selections.
- MLB4 file format and public threshold-driven API are unchanged.

Validated experiment on Granite 4.2 3B / Tesla T4 before library patch:

```text
Native FP16 / F.linear         : 19.606 tok/s
Adaptive FP16 / FastWarp       : 21.106 tok/s
HardwareNative / FastWarp      : 23.187 tok/s
HardwareNative vs native       : 1.183x
Decode-only resident reduction : ~39.1%
```

---

# ElasticBit clean adaptive API patch

## Frozen algorithm

ElasticBit is threshold-driven only. There is no public fixed-bit quantization API.

For every weight matrix:

1. Capture representative calibration activations.
2. Build the FP16 execution reference using FP16 weights and FP16 activations.
3. Test storage widths 3 through 15.
4. For each width, measure both:
   - `hardwareNative` execution error;
   - `fullPrecision` execution error.
5. Use the larger of those two errors as the matrix's threshold error.
6. Select the smallest storage width whose error is `<= threshold`.
7. If no 3–15-bit width passes, store FP16 (16-bit fallback) rather than violate the threshold.

This dual-policy check is deliberate: a compressed matrix can switch between
`hardwareNative` and `fullPrecision` without recalibration or recompression.

## Execution

Activations remain FP16 in every policy.

`hardwareNative`:

- Tesla T4 keeps the validated mapping: 3–4-bit -> W4A16, 5–8-bit -> W8A16, 9–16-bit -> W16A16.
- Other CUDA GPUs detect their architecture and auto-tune the legal built-in execution widths per matrix shape:
  - storage 3–4 bit: candidates W4A16, W8A16, W16A16;
  - storage 5–8 bit: candidates W8A16, W16A16;
  - storage 9–16 bit: W16A16.
- The fastest candidate is selected for decode. The stored bit width is never changed.

`fullPrecision`:

- all stored widths -> W16A16

No low-bit activation quantization is used.

Prefill is phase-separated from decode. ElasticBit materializes the compressed weight directly on-device as transient FP16 and delegates the matrix multiplication to PyTorch/vendor GEMM. This avoids the previous CPU dequantization + host-to-device round trip.

T4 low-bit decode uses the validated vectorized W4A16/W8A16 kernels (packed 32-bit weight loads, half2 activation loads, warp-shuffle reduction). W16A16 model execution is delegated to ATen/cuBLAS over a zero-copy view of RuntimeMatrix-owned FP16 execution memory instead of the compatibility scalar GEMV kernel.

## Public API

```python
ElasticBit.compress(...)
ElasticBit.analyze(...)
ElasticBit.compressMatrix(...)
ElasticBit.loadMatrix(...)
ElasticBit.save(...)
ElasticBit.load(...)
ElasticBit.backend()

model.elasticbit.summary()
model.elasticbit.setDecodePolicy(...)

matrix.forward(...)
matrix.setDecodePolicy(...)
matrix.save(...)
```

Public names use camelCase where multiple words are required.

## Removed

Removed rather than aliased/deprecated:

- `ElasticBit(bits=...)`
- `ElasticBitConfig`
- `PackedElasticBit` public export
- `ElasticLinear` / `ElasticEmbedding` public exports
- `quantize_tensor`
- `dequantize_tensor`
- `quantize_module`
- `bitsAnaliser`
- `RuntimeMatrix.from_auto`
- `runtime`, `runtime_mode`, `compact`, `fast`, `cached`, `packed`
- manual bit-width selection
- W4A4 / W8A8 activation-quantized execution
- lifecycle `mlbricks.quantize(...)`

## Serialization

Matrix serialization is MLB4 and stores threshold/error metadata, exact adaptive
storage, dimensions, and checksums. Decode policy is selected at runtime and is
not part of the compressed storage semantics.

## Validation performed in this patch environment

- Python/public API import checks passed.
- Legacy public-name audit passed.
- Full repository tests: 235 passed, 6 skipped.
- CPU wheel build succeeded.
- CUDA-native tests were added and are skipped when the native CUDA extension is unavailable.

This patch environment has CPU-only PyTorch and no `nvcc`, so the rewritten CUDA
runtime must still be compiled and executed once on a CUDA machine (the Tesla T4
benchmark environment is appropriate) before the native kernel patch is called
production-verified.

## Hardware planner update

- Compression/storage remains hardware-independent.
- `hardwareNative` is now a runtime execution planner.
- Exact Tesla T4 detection preserves the experimentally validated T4 mapping.
- Other CUDA devices use `cudaAutoTune`, selecting the fastest legal built-in ElasticBit decode kernel for each matrix shape.
- `fullPrecision` remains a deterministic W16A16 path.
- Auto-tuning is optimization among the kernels currently shipped by ElasticBit; it does not claim that those kernels are globally optimal for every future GPU. New architecture-specific kernels can be registered later without changing MLB4 storage.
