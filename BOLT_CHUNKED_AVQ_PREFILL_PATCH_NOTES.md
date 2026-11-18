# Chunked AVQ cached prefill

Base: `1f9b839b85871f0b97f5f8d3c4ec1af482990c9d` (`optimised-bolt`).

- Ports the `chunk_avq_2048` candidate from the completed Tesla T4 inference experiment.
- Eval-mode cached causal MQA prefill with PyTorch AVQ automatically chunks inputs longer than 2048 tokens. Each chunk carries the final FP32 recurrent state into the next affine scan. The full enhanced sequence and complete attention history are retained.
- Ordinary forward/training, decode math, Keyless, multi-KV/GQA, and non-PyTorch AVQ retain their existing paths. No shared-KV prefill or batch microchunking is introduced by this patch. Nonchunked cached prefill clones the final state before attention rather than afterward.
- No parameters, state-dict keys, or cache tuple schemas change. Numerical equivalence is tolerance-based because scan association and GEMM shapes may change FP rounding.
- For an A/B test, set `EnhancedMQA.avq_prefill_chunk_size = 0` (or on a particular component instance); default is 2048. This is a runtime attribute, not a serialized configuration field.

Research notebook evidence (not a new native-library GPU measurement): batch 4 / 32K / FP16 with external batch microchunking reduced peak prefill allocation from 4.351 to 2.298 GiB for Learned and 3.906 to 2.228 GiB for Decay. Actual peak depends on batch policy, allocator, backend, and model. Native-package CUDA benchmarking and trained-checkpoint validation remain necessary.

Run focused regressions: `python -m pytest -q tests/test_bolt_chunked_avq_prefill.py tests/test_bolt_compact_inference.py tests/test_optimised_bolt_shared_c.py tests/test_bolt_fixed_retention.py`.

## Explicit low-precision return projection fix

The non-identity AVQ return projection receives an FP32 readout but can have FP16/BF16 weights. With autocast disabled, cast only the linear input to its weight dtype in full forward, chunked prefill and step. FP32 recurrence, identity mapping, autocast behavior, parameters and checkpoint schema are unchanged. The original 2048-boundary CUDA cases now also check decode; CPU FP16/BF16 regressions reproduce the dtype issue without CUDA.
