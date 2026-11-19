# Modern BOLT

Modern BOLT has three attention modes: `quantised` (MQA/GQA), `keyless` (one shared C), and `base` (MHA). Each supports optional pre-attention AVQ with `retention="decay"` or `"learned"`. Historical fixed-width and rho-normalized architectures are retired; old checkpoints are not automatically converted. Historical patch notes are development records, not current API specifications.

## BOLT model configuration

The embedding width must be divisible by the query-head count. For MQA/GQA, `kv_heads` must divide the query-head count. The default AVQ width is the embedding width. The default precision is FP32; mixed precision is opt-in.

```python
from mlbricks import Bolt, BoltConfig, BoltModel, RoPE

mqa = Bolt(384, 12, mode="quantised", kv_heads=1, position=RoPE())
gqa = Bolt(384, 12, mode="quantised", kv_heads=3)
keyless = Bolt(384, 12, mode="keyless")
classic = Bolt(384, 12, mode="base", avq=False, head_output_gate=False)
learned = Bolt(384, 12, retention="learned")
config = BoltConfig(vocab_size=50257, context=2048, layers=3, dim=384, heads=12,
                    layer_modes=("base", "quantised", "keyless"), device="auto")
model = BoltModel(config)
model.set_qk_transform(RoPE())
```

At 384D/12 heads, FP16 persistent attention storage is 128 bytes/token/layer for MQA, 64 for shared-C Keyless, and 1536 for MHA. Enabled 384D AVQ adds 1536 bytes of FP32 recurrent state per sequence/layer. Cache storage is distinct from peak GPU allocation. Modern BOLT uses PyTorch SDPA; the retired BOLT native attention and graph routes are not available. The optimized one-KV-head decode and chunked AVQ prefill are retained; chunking does not alter training.

## Bolt attention

### `Bolt` / `BoltAttention`

`BoltAttention` aliases `Bolt`. Use `mode="quantised"`, `"keyless"`, or `"base"`. Positional transforms are external callables, such as `RoPE()`, rather than strings. `state_memory=False` or `avq=False` disables AVQ; `head_output_gate=False` disables learnable head-output gates.

```python
bolt(x)  # differentiable forward
output, cache = bolt.prefill_with_cache(x)
next_output, cache = bolt.decode_step(next_x, cache)
fixed = bolt.allocate_cache(batch_size=1, capacity=2048)
y = bolt.decode_with_cache(next_x, fixed, position=0)
```

The fixed-cache example starts a separate empty session. MQA/GQA/base use `(k,v,state)` dynamic caches and `EnhancedMQACache` static caches. Keyless uses `(c,state)` dynamic caches and `SharedCCache` static caches. The actual cache precision follows the model's inference precision. Cache allocation and cached decode are for inference; use ordinary forward for training.

### BOLT model and components

`BoltModel` and `BoltConfig` provide a language model; `BoltStateMemory` implements pre-AVQ. `EnhancedMQA`, `EnhancedMQACache`, `ClassicMHA`, `SharedCKeyless`, and `SharedCCache` are also exported. `BoltModel` supports forward, prefill, incremental decoding and generation. Modern mode weights retain their existing parameter and state-dict layout. Historical BOLT16/BOLT32 checkpoints and fixed-width graph caches are not supported. Existing naming aliases such as `fast` select modern modes, not a second implementation. Modern graph acceleration is not implemented; graph policy does not select the retired kernels.
