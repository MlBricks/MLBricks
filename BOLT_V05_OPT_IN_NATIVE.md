# BOLT v0.5: Opt-in U/G + AVQ native CUDA inference (candidate patch)

**Status: experimental, not merged, GPU integration unverified.**

This patch applies to the **MLBricks BOLT v0.2 optimized-inference snapshot**
provided with the development artifacts. It does **not** apply automatically to
all current `main` commits. Apply on a branch with the v0.2 patch already
present. Review conflicts rather than forcing a three-way apply.

## Scope

- Q+P weight folding and packed Q/U/G projection: reuse the existing v0.2 path.
- Native CUDA fused `U * (1+tanh(G))` for **single-token decode** only.
- Native CUDA fused post-softmax AVQ-198 sigmoid/tanh recurrence, FP32 state
  update, normalization and read gate for **single-token decode** only.
- The existing prefill, training and checkpoint APIs remain unchanged.
- No Triton and no `torch.compile`. The native extensions use C++/CUDA.
- No new trainable `Parameter` objects or `state_dict` entries.
- Native operations operate with persistent per-module outputs to make them
  compatible with CUDA Graph capture.
- The cached key/value tensor is now zero-initialized and inactive padding is
  cleared on cache reset/prefix load to avoid stale/non-finite masked entries.

## Kaggle / CUDA usage

1. Use a **fresh GPU runtime with the CUDA toolkit and nvcc available**.
2. Unzip the patched MLBricks source in a local working directory. You can
   directly import it from its root without installing a conflicting PyTorch
   version. To build native on demand, set:

```python
import os
os.environ['MLBRICKS_BOLT_V05_ALLOW_JIT'] = '1'
```

3. Ensure your MLBricks checkout is on the Python import path (or install it
   with `pip install -e . --no-deps` from the patched source directory).
4. Enable native fusions **explicitly**, after v0.2 optimized inference:

```python
import torch
from mlbricks import BoltModel

model = BoltModel(
    vocab_size=2048, dim=384, heads=12, layers=6, context=600,
    bolt_mode='quality', bolt_cache='tight', avq_dim=198,
    position='none', backend='pytorch', dropout=0,
).eval().cuda().half()

model.enable_bolt_optimized_inference(True)
model.enable_bolt_v05_native(gate=True, avq=True)
model.prepare_bolt_v05_native()  # compile/load OUTSIDE graph capture
prompt = torch.randint(0, 2048, (1,512), device='cuda')
with torch.inference_mode():
    # One-shot session (graph capture / prefill are part of setup).
    session = model.prepare_cuda_graph_generation(prompt, 64)
    generated = session.generate()
print(generated.shape)  # [1,576]
```

**Note:** This is a *random-weights API example* demonstrating integration,
not a language-quality evaluation. Test actual trained checkpoints separately.

To disable without touching weights:

```python
model.enable_bolt_v05_native(gate=False, avq=False)
```

`bolt_mode='fast'` omits AVQ; use `avq=False`. Graph mode currently requires
`position='none'`, standard FFN, standard residual, dropout=0, greedy decoding,
preallocated cache, and a fresh one-shot graph for each prompt. Eager fallback
supports the existing model's other options, but fused kernels are decode-only.

## Prebuilt CUDA option

For a wheel built from source (instead of JIT), use
`MLBRICKS_BUILD_BOLT_V05_NATIVE=1` when invoking `pip wheel`/`pip install`.
This requires a matching CUDA-enabled PyTorch build and CUDA toolkit. Both
flags are *off by default*. The code imports `mlbricks._bolt_v05_cuda` when
prebuilt, otherwise JIT compilation is allowed only with the explicit flag.

## Validation

CPU (does not compile CUDA kernels):

```bash
python -m pytest -q tests/test_bolt_optimized_inference.py tests/test_bolt_v05_native.py
```

Kaggle GPU native smoke + graph parity, requires `nvcc`:

```bash
MLBRICKS_BOLT_V05_ALLOW_JIT=1 python -m pytest -q tests/test_bolt_v05_native.py
```

Full GPU A/B (two cache sizes, two seeds, eager + graph, four variants):

```bash
MLBRICKS_BOLT_V05_ALLOW_JIT=1 python examples/bolt_v05_native_benchmark.py
```

This outputs `bolt_v05_mlbricks_gpu_validation.json`. For speed, it measures
only `max_new_tokens-1` decode steps: the initial next token comes from
prefill. CUDA Graph construction/prefill are excluded from steady-state timing.
The benchmark also rejects incorrect tokens and non-finite AVQ cache states.
It does **not** automatically establish parity of *all logits* or state vectors;
for full numerical audits, add teacher-forced tensor comparisons on trained
checkpoints and stress batch sizes, RoPE, and other precisions.

## Known limitations / release gates

- CUDA kernel compilation and execution have **not** been run in the authoring
  environment (CPU-only). GPU test cases are skipped locally.
- Kernel code has been prepared from the previously T4-tested standalone v0.5
  equations, but MLBricks production integration may change performance.
- CUDA Graph Python capture and replay need independent validation on T4.
- Do not merge into the default public inference path until all native GPU
  parity tests and multi-seed, multi-config benchmarks pass.
- Only quality-mode AVQ-198, batch 1, FP16, no RoPE is the intended initial
  performance target; other dimensions are test cases, not performance claims.
- `BoltCudaGraphGenerator` is one-shot and not suitable for concurrent callers
  sharing the same model's persistent kernel scratch buffers.
- Existing `pyproject.toml` pins PyTorch `<2.11`; Kaggle may ship a newer
  runtime. Avoid dependency-driven reinstall by using `--no-deps` and verify
  compatibility independently. Do not assume a PyTorch 2.11 GPU run is covered
  by the package's published dependency contract.
