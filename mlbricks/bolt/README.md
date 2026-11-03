# MLBricks BOLT

BOLT provides compact causal attention and learned recurrent state memory.
Its canonical public API uses architecture-oriented names and automatic
hardware dispatch. No kernel implementation names are required in user code.

```python
import torch
from mlbricks import Bolt, BoltConfig, BoltModel, BoltStateMemory, BoltCache

config = BoltConfig(
    vocab_size=50257, context=2048, layers=6, dim=384, heads=12,
    profile="standard", state_dim=198,  # state_memory=True by default
    device="auto", precision="auto", backend="auto",
)
model = BoltModel(config)  # CUDA by default when available; otherwise CPU
logits = model(torch.randint(0, 50257, (1, 32)))

compact_layer = Bolt(embed_dim=384, num_heads=12, profile="compact")

# Attention-only BOLT-32 (no AVQ allocation or recurrent state)
attention_only = BoltModel(BoltConfig(
    vocab_size=50257, context=2048, layers=6, dim=384, heads=12,
    profile="standard", state_memory=False,
))
standalone_attention_only = Bolt(embed_dim=384, num_heads=12,
                                 profile="standard", state_memory=False)
standalone_memory = BoltStateMemory(input_dim=384, width=198)
```

## Components

| Public name | Role |
| --- | --- |
| `BoltConfig` | Causal-model architecture and execution policy |
| `BoltModel` | Complete trainable autoregressive language model |
| `Bolt` | Standalone compact attention + recurrent state layer |
| `BoltStateMemory` | Standalone learnable recurrent memory |
| `BoltCache` | Preallocated compact context and memory state |

`profile="standard"` uses 32 compact dimensions per head;
`profile="compact"` uses 16. Both use 198-dimensional recurrent state by
default. The new model profile uses latent-width attention scaling and
balanced state-memory initialization (`gamma=0.25`). **`state_memory=True` is
always the default for new BOLT models.** Set `state_memory=False` on `BoltConfig`
or `Bolt` to disable AVQ; this does not instantiate AVQ projection, recurrence,
readout, or recurrent-cache tensors. The attention core stays the same, but the
result is a different model architecture with different checkpoint keys.
Existing weights remain loadable with their original `attention_scale` / profile
settings. Historical `mode="fast"` continues to request attention-only BOLT.

`device="auto"` chooses CUDA if available; explicit `device="cuda:1"` chooses
a particular GPU. CPU input tokens are moved automatically to the model device,
but placing inputs directly on the accelerator minimizes transfer overhead.
`precision="auto"` activates CUDA mixed precision while keeping FP32 weights.

Native recurrent kernels are optional at source-build time:

```bash
MLBRICKS_BUILD_BOLT_ACCEL=1 pip install -v .
```

If unavailable, automatic execution uses the original compatible reference.
No runtime JIT or Triton is needed. The legacy separate-projection checkpoint
keys are preserved, even though the default new layer trains one packed
projection parameter. Build the optimizer **after** model construction; do not
assume optimizer states are interchangeable between the layouts.

## Compatibility reference

The historical `BoltPreset`, `BoltPresetCache`, `PostSoftmaxAVQ`, `Gaussian`,
and `GaussianConfig` imports are maintained for existing code. Their older
construction options are described below; use profiles for new applications.

---

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
