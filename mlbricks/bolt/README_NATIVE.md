# Native CUDA support for modern BOLT + AVQ

This patch adds a new, independent `mlbricks._bolt_modern_cuda` C++/CUDA extension. It **does not** reuse the legacy rho-normalized BOLT kernels (`_gauss_cuda`). The default remains `Bolt(384,12)`: **BOLT-32 with 198-dimensional read-gated AVQ**.

## Kernel coverage

| Operation | CUDA implementation | Differentiable? |
|---|---|---|
| BOLT learned query scale `Q * P`, gated latent `U*(1+tanh G)` | Fused feature kernel + native backward | Yes (Q/U/G/P) |
| Causal attention during training/prefill | PyTorch optimized CUDA SDPA | Yes (native PyTorch C++/CUDA) |
| AVQ recurrence `S[t]=A[t]*S[t-1]+V[t]` | Native CUDA serial-per-channel fused scan + backward | Yes |
| Cached BOLT attention with C-only cache | Native CUDA fused cache append + stable softmax decode, no RoPE | Inference only |
| AVQ recurrent one-token update | Native CUDA state kernel | Inference only |
| BOLT-16 / AVQ widths 64/96/192 | Same native kernels; default preset is P32 + AVQ198 | Yes |

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

m = Bolt(embed_dim=384, num_heads=12, mode='quality', cache='normal', avq_dim=198, backend='auto').cuda()
print(m.resolved_backend())  # CUDA extension ready or PyTorch fallback

# Force native dispatch: raises if the new extension is missing.
m.set_backend('native')
```

- `backend='auto'`: use the modern native kernels only if the extension is installed and inputs are supported CUDA tensors; otherwise fall back to PyTorch.
- `backend='pytorch'`: use the original all-PyTorch modern BOLT equations.
- `backend='native'`: require the new modern extension; this does not call legacy BOLT native code.
- Native cached-decode currently supports `position=None` and latent widths **16/32**. `position='rope'` uses PyTorch cached-decode for `auto`; forcing `native` with RoPE decoding raises.
- The complete full-sequence attention uses PyTorch SDPA regardless of backend; backward still flows through the fused BOLT features and CUDA AVQ scan.
- Original BOLT state-dict keys are not changed. Rebuilding the same BOLT-32 + AVQ198 module can load its prior checkpoints. It does **not** make P16/AVQ192 checkpoints interchangeable with P32/AVQ198.

## Verification before deployment

```bash
python -m pytest tests/test_bolt_native_cuda.py -v
python examples/bolt_native_benchmark.py
```

The GPU test suite compares `native` and `pytorch` **forward, backward, prefill and progressive decoding** with nonzero AVQ gamma. No GPU tests could be run during patch creation in the CPU-only environment; don't treat the kernels as production-validated until the test suite passes on your target GPU.

For CUDA builds, work with `torch.autocast('cuda', dtype=torch.float16)` or FP32. BF16 takes PyTorch fallback under `auto`, and is not a supported native feature dtype in this version. Generated cache and recurrent state are separate; AVQ recurrent state uses FP32.
