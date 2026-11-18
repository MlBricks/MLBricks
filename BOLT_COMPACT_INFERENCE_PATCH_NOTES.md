# BOLT compact inference candidate

Base: `optimised-bolt`, commit `0b75339e8a066b3c7f49c3b33d6aaf20290e207e`.

## Changes

- `mlbricks/bolt/enhanced_mqa.py`: clone the final FP32 AVQ state in prefill so a singleton-batch state view does not retain the full scan-history allocation.
- For single-token, noncausal, evaluation/no-grad inference with exactly one KV head, reinterpret query heads as independent query rows for SDPA. Avoids physically repeating shared K/V. Both tuple and preallocated decode call this helper. Full-sequence, causal, gradient-enabled and multiple-KV-head attention retain the existing path.
- `tests/test_bolt_compact_inference.py`: compact storage, dynamic/static decode vs full forward, 48/64 query heads, learned/decay, batch 1/4, disabled AVQ, explicit-repeat attention and gradient parity tests. CUDA cases run when available.

No model parameters, checkpoint keys, API signatures or package version changed. Experimental grouped prefill, batch microchunking and new cache allocation logic are intentionally not integrated. Existing static-cache API retained.

## Local verification

PyTorch 2.10.0+cpu. Selected tests only:

```
python -m pytest -q tests/test_bolt_compact_inference.py tests/test_optimised_bolt_shared_c.py tests/test_bolt_fixed_retention.py tests/test_bolt_optimized_training.py tests/test_bolt_public_api.py
```

Patched: 63 passed, 9 failed, 9 skipped. New regression suite alone: 33 passed, 8 CUDA cases skipped. Original ZIP, same four existing test files: 30 passed, 9 failed, 1 skipped. The same nine existing tests fail before and after the patch: seven in `test_bolt_optimized_training.py` (obsolete BOLT16/profile/training hooks) and two in `test_bolt_public_api.py` (legacy cache tuple/projection API assumptions). No tests removed or weakened. Full suite, trained-checkpoint parity and CUDA execution of this exact patched package not performed. The preceding notebook tests provided GPU evidence for the optimization concept, not a new GPU validation of this package. Validate on GPU before merging or publishing.
