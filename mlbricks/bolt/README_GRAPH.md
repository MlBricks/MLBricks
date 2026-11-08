# BOLT single-token CUDA Graph engine (experimental)

This opt-in feature ports the dynamic-position BOLT CUDA reader and guarded
one-token capture/replay path from the T4×2 qualification harness. It does **not**
replace the existing BOLT decoder, training path, V2 Strict AVQ math, or the
FP32 default public configuration. It is not enabled automatically.

## Qualified scope

The graph fast path currently supports the separately tested **WikiLM-like
adapter protocol**, with `model.kind` of `bolt16` or `bolt32`, batch 1, no RoPE,
FP16 weights, FP16/FP32 compact C cache, and FP32 recurrent AVQ state. The
model must expose `embedding`, `layers` (`mixer`, `norm1`, `norm2`, `ffn`),
`final_norm`, and `head`, matching the qualification architecture. Models
outside that scope use the available `decode_one` or `decode_step` eager path,
not a partly validated graph. This includes the public `BoltModel`, pending a
separate model-block adapter and trained-model qualification.

## Build

On a CUDA environment where **NVCC and the CUDA version used to build PyTorch
are compatible**, enable optional extensions (Linux example):

```bash
MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 \
MLBRICKS_BUILD_BOLT_V2_STRICT=1 \
MLBRICKS_BUILD_BOLT_DYNAMIC_GRAPH=1 \
python -m pip install -v --no-build-isolation --no-deps -e .
```

A successful build must also pass a fresh-process import test:

```bash
python -c 'import mlbricks._bolt_dynamic_cuda as m; assert callable(m.decode)'
```

The graph kernel is **not compiled by default**. If unavailable, the engine
falls back to eager decoding unless `strict_graph=True` is specified.

## API

```python
from mlbricks import BoltGraphEngine

# model must be in eval mode and frozen for the graph fast path.
# The engine owns a persistent one-token buffer and a GPU-resident position.
engine = BoltGraphEngine.from_prompt(
    model, prompt_tokens,
    max_new_tokens=256,
    enable_graph=True,
)
for token_id in engine.generate(256, eos_id=50256):
    print(token_id, flush=True)
print(engine.status())
```

`from_prompt` works with the tested WikiLM-style `prefill(ids, capacity)`
and with the public `BoltModel.prepare_generation` / `prefill` interface,
**using eager fallback** for the latter.

This API yields integer token IDs, not decoded text. Detokenize each ID in the
caller; GPU→CPU token synchronization happens once per yielded token. Measured
throughput is workload/hardware dependent and excludes prefill/capture unless
explicitly benchmarked.

## Safety and limitations

- Captured kernels read a persistent **CUDA int64 position tensor**, updated
  inside replay; the host chooses a bounded graph variant as the context grows.
- Cache, AVQ state, position, input token and weights retain their memory
  addresses for the lifetime of a captured graph.
- Capture is guarded by in-place snapshots and restoration; recoverable
  host-side capture exceptions can fall back to eager.
- **Do not continue after a fatal CUDA fault** (e.g. illegal address): the CUDA
  context can be poisoned. Replay errors are propagated, never silently retried.
- Cache capacity is fixed. EOS stopping is checked after each returned token.
- Prefill, token transfer, capture costs, arbitrary sampling/top-k, multi-batch,
  RoPE, changing models after capture and deployment on untested GPUs remain
  outside the validated accelerated path.

## Validation before merging

Run CPU regression tests and native CUDA tests, then T4×2 graph/eager parity and
EOS/capacity/1,201-token/8K–32K stress tests using **this branch's built
extension**. The original experimental notebook's pass does not automatically
prove the integrated package has passed. Full repository CI and review are
required before merging into `main`.
