# MLBricks BOLT — AVQ state width 384 native CUDA scan fix

## Scope

This change permits the two optional accelerated BOLT CUDA scan extensions
(`mlbricks._bolt_scan_short`, `mlbricks._bolt_scan_segmented`) to process more
than 256 AVQ memory channels without skipping channels or assuming they all fit
in one CUDA thread block.

- BOLT-16 / AVQ-192 continues to work with the same mathematical recurrence.
- BOLT-32 / AVQ-384 now has one block per 256-channel tile in each segmented
  scan stage. The short-sequence warp scan already tiled by groups of 32.
- The summary/carry stages were already channel-tiled and remain unchanged.
- The AVQ recurrence, FP32 accumulated state, and derivatives are unchanged.
- No Query P parameter or AVQ return projection is reintroduced.
- This changes the native extension source and requires rebuilding CUDA .so
  files; reinstalling the Python package alone is not sufficient.

## GPU qualification

Enable both native extensions, then run the direct and autograd tests:

```bash
MLBRICKS_BUILD_BOLT_ACCEL=1 MAX_JOBS=1 python setup.py build_ext --inplace
MLBRICKS_VALIDATE_AVQ384_NATIVE=1 python -m pytest -q tests/test_bolt_avq384_native_scan_cuda.py
```

The tests cover widths 192, 256, 257, 384 and 513; FP16/FP32 raw logits;
short-context and 256-token segmented paths; first-state gradients; non-contiguous
packed Q/A/V slices; and Python autograd dispatch. Cross-device safety requires
both GPUs to be visible in the same process.

**GPU results must be obtained on Kaggle or another compatible NVIDIA system.**
Passing CPU source-integrity tests is not proof of native CUDA correctness.
