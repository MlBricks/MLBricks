# Native CUDA support for modern BOLT + AVQ

This patch adds a new, independent `mlbricks._bolt_modern_cuda` C++/CUDA extension. It **does not** reuse the legacy rho-normalized BOLT kernels (`_gauss_cuda`). The default remains `Bolt(384,12)`: **BOLT-32 with 384-dimensional read-gated AVQ (Identity return)**.

## Kernel coverage

| Operation | CUDA implementation | Differentiable? |
|---|---|---|
| Direct-Q BOLT, gated latent `U*(1+tanh G)` | No-P native gated-feature kernel during inference (when built), PyTorch for training | Training uses autograd through Q/U/G |
| Causal attention during training/prefill | PyTorch optimized CUDA SDPA | Yes (native PyTorch C++/CUDA) |
| AVQ recurrence `S[t]=A[t]*S[t-1]+V[t]` | Native CUDA serial-per-channel fused scan + backward | Yes |
| Cached BOLT attention with C-only cache | Native CUDA fused cache append + stable softmax decode, no RoPE | Inference only |
| AVQ recurrent one-token update | Native CUDA state kernel | Inference only |
| BOLT-16 / AVQ widths 64/96/192/384 | Same native kernels; default preset is P32 + AVQ384 | Yes |

This is **not** a fused end-to-end Transformer (linear projections, softmax prefill, AVQ projections and RMSNorm still use PyTorch operators). The native kernels are meant to reduce BOLT and recurrent-memory overhead without replacing the well-optimized SDPA prefill. The native decode attention is a basic correctness-first stable-softmax implementation, **not** a claimed performance improvement. CUDA Graph tuning/tiling should follow T4 measurements.

## Build on an NVIDIA CUDA development machine

Requirements: matching CUDA-enabled PyTorch and CUDA toolkit (`nvcc`), C++17 compiler, and compatible NVIDIA GPU (T4 supported). Example on Linux/Kaggle (with a functioning CUDA toolkit):

```bash
export MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1
export MLBRICKS_EXPECT_CUDA_NATIVE=1
export MAX_JOBS=2
python -m pip install -v --no-build-isolation .
```

An explicit native build fails if the CUDA toolkit is unavailable; it no longer silently produces a fallback-only wheel.

For development, with a CUDA toolkit but no installed extension, one can opt into JIT instead of building a wheel:

```bash
export MLBRICKS_BOLT_MODERN_ALLOW_JIT=1
```

JIT is opt-in to avoid surprise compilation at model startup. Building the extension into the wheel is preferred.

## Usage

```python
from mlbricks import Bolt

m = Bolt(embed_dim=384, num_heads=12, profile='standard', backend='auto')
print(m.resolved_backend())  # CUDA extension ready or PyTorch fallback

# Force native dispatch: raises if the new extension is missing.
m.set_backend('native')
```

- `backend='auto'`: use the modern native kernels only if the extension is installed and inputs are supported CUDA tensors; otherwise fall back to PyTorch.
- `backend='pytorch'`: use the original all-PyTorch modern BOLT equations.
- `backend='native'`: require the new modern extension; this does not call legacy BOLT native code.
- Native cached-decode currently supports `position=None` and latent widths **16/32**. `position='rope'` uses PyTorch cached-decode for `auto`; forcing `native` with RoPE decoding raises.
- The complete full-sequence attention uses PyTorch SDPA regardless of backend. During training, Q/U/G and their gradients use PyTorch autograd; the optional native feature kernel is inference-only. The native AVQ scan supports backward.
- Modern BOLT trains Q directly and does not allocate a P parameter. Use freshly trained checkpoints for the new parameterization; it is not interchangeable with checkpoints from the prior learned-P implementation.

## Verification before deployment

```bash
python -m pytest tests/test_bolt_modern_native_cuda.py tests/test_bolt_directq_native_cleanup.py -v
python examples/bolt_native_benchmark.py
```

The GPU test suite should verify the direct-Q `native` and `pytorch` **forward, backward, prefill and progressive decoding** behavior with nonzero AVQ gamma. No GPU tests could be run during patch creation in the CPU-only environment; don't treat the kernels as production-validated until the test suite passes on your target GPU.

For CUDA builds, work with `torch.autocast('cuda', dtype=torch.float16)` or FP32. BF16 takes PyTorch fallback under `auto`, and is not a supported native feature dtype in this version. Generated cache and recurrent state are separate; AVQ recurrent state uses FP32.


## Experimental BOLT V2 strict one-token AVQ decode (opt-in)

This source also includes the independently compiled `mlbricks._bolt_avq_v2_strict`
CUDA extension. **It is disabled by default** and does not replace the modern
BOLT native kernels or change the public API, training, or checkpoint format.
The V2 state update retains an FP32 recurrence with separate-rounding
`__fmul_rn`/`__fadd_rn`, and **preserves PyTorch `avq.project(o)`** rather than
using the numerically different V3 custom GEMV.

Build on a CUDA-enabled development environment:

```bash
MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 MLBRICKS_BUILD_BOLT_V2_STRICT=1 \
  python -m pip install -v --no-build-isolation .
```

Opt in for supported *inference-only* decode workloads:

```bash
MLBRICKS_BOLT_V2_STRICT_DECODE=1 python your_inference_script.py
```

Eligibility: `Bolt`/`BoltPreset` modern quality mode, Identity AVQ return,
FP16 contiguous attention output, FP32 state, batch 1, one token, widths
192/384, no gradients. Unsupported shapes use the existing code path. The
compiled extension and compatible CUDA/PyTorch stack are required when
explicitly selecting a native backend with V2 enabled. The optional speedup
has **not** been validated on every GPU or against every current release;
run the tests below before enabling in a production deployment:

```bash
python -m pytest tests/test_bolt_avq_v2_strict.py \
  tests/test_bolt_modern_native_cuda.py -v
```

The strict V2 path is *not* an end-to-end compiled Transformer, and it does
not modify native cached attention. Do not merge into a released `main` branch
without matched GPU parity and performance qualifications.
