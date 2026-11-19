# BOLT Architecture

## Architecture Modes

BOLT is a memory-efficient attention architecture with optional pre-attention AVQ, dynamic head dimensions, learnable head-output gates, and externally supplied positional transforms.

| Architecture | `mode` | Configuration |
|---|---|---|
| BOLT Quantised | `quantised` | One shared KV head (`kv_heads=1`, the default) |
| BOLT Grouped Quantised | `quantised` | Multiple shared KV heads (`1 < kv_heads < num_heads`) |
| BOLT Keyless | `keyless` | One shared C representation; `kv_heads` is not applicable |
| BOLT Base | `base` | Full per-head key and value representations |

Quantised and Grouped Quantised use the same implementation and differ only in `kv_heads`. Setting `kv_heads=num_heads` in `quantised` uses full KV width. Every mode supports optional AVQ with `retention="decay"` (default) or `retention="learned"`.

## BOLT API

The embedding width must be divisible by the query-head count. For Quantised and Grouped Quantised, `kv_heads` must be a positive divisor of the query-head count. The default AVQ width is the embedding width. FP32 is the default precision; mixed precision is opt-in.

```python
from mlbricks import Bolt, BoltConfig, BoltModel, RoPE

# BOLT Quantised
quantised = Bolt(384, 12, mode="quantised", kv_heads=1, position=RoPE())

# BOLT Grouped Quantised
grouped_quantised = Bolt(384, 12, mode="quantised", kv_heads=3)

# BOLT Keyless
keyless = Bolt(384, 12, mode="keyless")

# BOLT Base
base = Bolt(384, 12, mode="base")

# BOLT Base without AVQ or head-output gates
classic = Bolt(384, 12, mode="base", avq=False, head_output_gate=False)

# BOLT with learned AVQ retention
learned = Bolt(384, 12, retention="learned")

# Hybrid BOLT language model
config = BoltConfig(
    vocab_size=50257,
    context=2048,
    layers=3,
    dim=384,
    heads=12,
    layer_modes=("base", "quantised", "keyless"),
    device="auto",
)
model = BoltModel(config)
model.set_qk_transform(RoPE())
```

## Memory Efficiency

At 384D and 12 query heads (32 dimensions per head), persistent FP16 attention-cache storage is:

| BOLT architecture | Bytes/token/layer | Reduction vs Base |
|---|---:|---:|
| Quantised | 128 | 91.67% |
| Grouped Quantised (3 KV heads) | 384 | 75.00% |
| Keyless | 64 | 95.83% |
| Base | 1,536 | 0% |

Enabled 384D AVQ adds 1,536 bytes of FP32 recurrent state per sequence per layer, independent of context length. The table excludes AVQ state, model weights, temporary allocations and unused preallocated capacity. Persistent cache storage is distinct from peak GPU allocation.

## Execution and Optimization

BOLT uses PyTorch Scaled Dot-Product Attention (SDPA), with compact single-KV-head decoding and chunked AVQ inference prefill. Chunking carries recurrent state across chunks and does not alter training. FP16 and BF16 mixed-precision inference require suitable hardware. Modern native attention and CUDA graph acceleration are not implemented; optional standalone native operators are separate from the BOLT attention path.

## `Bolt` / `BoltAttention`

`BoltAttention` aliases `Bolt`. Use `mode="quantised"`, `"keyless"`, or `"base"`. Positional transforms are external callables, such as `RoPE()`, rather than strings. `state_memory=False` or `avq=False` disables AVQ; `head_output_gate=False` disables learnable head-output gates.

```python
bolt(x)  # differentiable forward
output, cache = bolt.prefill_with_cache(x)
next_output, cache = bolt.decode_step(next_x, cache)
fixed = bolt.allocate_cache(batch_size=1, capacity=2048)
y = bolt.decode_with_cache(next_x, fixed, position=0)
```

The fixed-cache example starts a separate empty session. Quantised, Grouped Quantised and Base use `(k,v,state)` dynamic caches and `EnhancedMQACache` static caches. Keyless uses `(c,state)` dynamic caches and `SharedCCache` static caches. The actual cache precision follows the model's inference precision. Cache allocation and cached decode are for inference; use ordinary forward for training.

## BOLT model and components

`BoltModel` and `BoltConfig` provide a language model; `BoltStateMemory` implements pre-AVQ. `EnhancedMQA`, `EnhancedMQACache`, `ClassicMHA`, `SharedCKeyless`, and `SharedCCache` are also exported. `BoltModel` supports forward, prefill, incremental decoding and generation. Modern mode weights retain their existing parameter and state-dict layout. Historical BOLT16/BOLT32 checkpoints and fixed-width graph caches are not supported. Historical branch and patch notes are development records, not current API specifications. Existing naming aliases such as `fast` select modern modes, not a second implementation. Modern graph acceleration is not implemented; graph policy does not select the retired kernels.
