# Fast-EsaQ → MLBricks ESA integration

Source: user-supplied `fast-EsaQ-main` v0.4.0, copied to `mlbricks/esa/fast_engine` (including CUDA source). Its half-life math is retained. An explicit multi-head initialization extension interpolates log half-lives for non-six-head MLBricks configurations while preserving the original six-head values.

## Fast-EsaQ only

- `mlbricks.ESA(embd=384, head=6)` always uses Fast-EsaQ's half-life recurrent equation. The legacy Thunder ESA class and architecture option were removed.
- `mlbricks.ESAModelConfig(...)` always creates Fast-EsaQ layers. The old `esa_architecture` selection is gone.
- `forward()` returns a Tensor; `prefill()`, `decode_step()`, and `forward_with_state()` return `(output, state)`. Initial state may be `[B,H,Dh]` or `[B,D]`, and recurrent state is FP32 by default.
- `backend='auto'|'native'|'pytorch'` chooses execution for the single architecture. CUDA compilation requires a GPU toolchain and is not validated here.
- Removed the obsolete `gate_min` and `gate_max` arguments from the ESA and VESA public interfaces. Half-life gates remain fully learned. The separate `eps` compatibility setting is unchanged.
- Fast-EsaQ base and current variants are exposed; `current` is default.
- SOUP's old packed path is disabled because it uses a different recurrence. Standard recurrent generation is available.
- Old Thunder ESA checkpoints and old-format model artifacts are rejected. Load them only using your own backup. There is no old ESA implementation in this package.
- Generic thunder-style affine scan helpers retained for other internal integrations do not provide a selectable old ESA layer or its weights.

## Example

```python
import torch
from mlbricks import ESA, ESAModel, ESAModelConfig

layer = ESA(embd=384, head=6, backend="pytorch", device="cpu")
x = torch.randn(2, 9, 384)
y = layer(x)
y2, final_state = layer.prefill(x)
assert torch.allclose(y, y2, atol=1e-5, rtol=1e-5)
print(y.shape, final_state.shape, final_state.dtype)

model = ESAModel(ESAModelConfig(vocab_size=512, embd=384, head=6, n_layer=2,
                                block=32, precision="fp32"),
                 device="cpu")
```

## Validation

See `tests/test_fast_esaq_merge.py`. CPU tests cover state/forward parity, decode continuation, gradients, non-six-head support, parameter identity, and rejecting incompatible checkpoints. CUDA kernels/latency are **not validated** without a GPU and NVCC.

## Public Fast-EsaQ scan controls (API naming update)

The following are **API-only renames**; execution, equations, and defaults are unchanged:

| Former Fast-EsaQ name | MLBricks Fast-EsaQ name | Default |
|---|---|---|
| `thunder_compass` | `cumpass` | `16` |
| `swqr_chunk` | `parallel_chunk` | `128` |
| `swqr_warps` | `cuda_warps` | `8` |

```python
from mlbricks import ESA
esa = ESA(embd=384, head=6, backend="auto", cumpass=16, parallel_chunk=128, cuda_warps=8)
```

For lower-level constructors, import `ESA`, `make_esa`, `make_model` or `ESAConfig` from `mlbricks.esa.fast_engine` and use the same three names. The old `swqr_` and `thunder_` constructor keywords are no longer accepted by these engine constructors. The MLBricks `compass` setting is otherwise retained for existing model integrations.

## Automatic backend dispatch with CUDA device default (Nov 2026)

Public `mlbricks.ESA(embd=384, head=6)` now requests `backend="auto"` and
`device="cuda"`, following the original Fast-EsaQ automatic scan and fused
decode dispatch policy while keeping the GPU device default. `ESAConfig`
defaults to `backend="auto"`; the `ESAModelConfig` and unrelated components
retain their previous defaults. Set `backend="cuda"` explicitly to require
strict native kernels. Use `backend="pytorch", device="cpu"` for explicit CPU
reference execution. Automatic dispatch does not silently change a requested
CUDA device to CPU.
