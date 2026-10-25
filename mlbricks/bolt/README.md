# BOLT presets in MLBricks

The primary BOLT constructor uses two independent controls:

| `mode` | `cache` | Attention | AVQ | Historical cache/head/token FP16 |
| --- | --- | --- | --- | --- |
| `quality` (default) | `normal` (default) | BOLT latent **32** | AVQ **198** | 64 bytes |
| `quality` | `tight` | BOLT latent **16** | AVQ **198** | 32 bytes |
| `fast` | `normal` | BOLT latent **32** | disabled | 64 bytes |
| `fast` | `tight` | BOLT latent **16** | disabled | 32 bytes |

```python
from mlbricks import Bolt, BoltModel
mixer = Bolt(embed_dim=384, num_heads=12)     # PyTorch-style, quality+normal, AVQ=198
# Equivalent: Bolt(384, 12) or Bolt(d_model=384, num_heads=12)
fast = Bolt(384, 12, mode='fast')             # BOLT-32, no AVQ
tight = Bolt(384, 12, cache='tight')          # BOLT-16 + AVQ-198
model = BoltModel(vocab_size=50257, dim=384, heads=12) # quality+normal
```

**Native status:** MLBricks ships C++/CUDA kernels for *legacy* BOLT. The modern BOLT (fast or quality) presets run PyTorch SDPA; AVQ-198 recurrence runs PyTorch. `backend="auto"` uses PyTorch for modern presets, while `backend="native"` raises an explicit unsupported-kernel error at execution. The legacy kernels are **not** mathematically interchangeable with BOLT and must not be used for these new presets.

**What the presets mean:** `quality` selects the post-softmax AVQ recurrence, not a measured guarantee of superior perplexity. `fast` removes the AVQ overhead; `tight` reduces the historical attention cache, not necessarily overall model memory. The historical cache stores *C only*; AVQ recurrent state contributes `batch * 198 * 4` bytes per layer. Unlike the old BOLT equation, BOLT uses learned scale `P` applied to the Query, and values/keys come from `C=U*(1+tanh(G))`.

The new architecture uses PyTorch SDPA and an autograd-safe O(log T) affine scan; **optimized fused AVQ-198 and BOLT-32 kernels have not been validated yet**. `mode='legacy'` (with explicit `latent_dim`) retains the original rho-normalized BOLT and native paths. Previously trained 16D/AVQ-192 models are *not checkpoint-compatible* with 32D/AVQ-198; changing either dimension requires conversion/retraining. Checkpoints from the old original BOLT equation require explicit `mode='legacy'` and matching dimensions.

## Save/Load

`state_dict` for a modern `Bolt` stores parameters under `_modern.*`; `BoltPreset` standalone stores them without that prefix. An incompatible checkpoint must raise a strict state dict load error rather than silently loading.

## Cached generation

`Bolt.prefill(x)` returns `(output, (C, state))`, where `state` is FP32 AVQ state for `quality` and `None` for `fast`. `Bolt.decode_step(token, cache)` advances both. `BoltModel.prepare_generation` uses `BoltPresetCache` for these modes and `GaussCache` for legacy mode.

## Native modern BOLT CUDA kernels

The new **modern BOLT-32/16 + AVQ-198 native extension** (`mlbricks._bolt_modern_cuda`) is optional and independent of the historical `_gauss_cuda` kernels. Build using `MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1` and a CUDA-enabled PyTorch/toolkit. It accelerates the BOLT gated-feature projection and AVQ recurrent scan (including backward), and provides fused CUDA cached decode without RoPE. Training/prefill attention still uses CUDA SDPA. See [README_NATIVE.md](README_NATIVE.md) for build commands, supported paths, parity tests and limitations. On machines without the extension, `backend='auto'` falls back to PyTorch, while `backend='native'` gives an explicit build error.
