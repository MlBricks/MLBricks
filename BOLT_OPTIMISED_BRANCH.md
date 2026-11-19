# Optimised BOLT branch: unified pre-AVQ and shared-C Keyless

Branch: `optimised-bolt` (derived from the supplied `optimised-avq` source snapshot).

> **Status:** architecture/reference implementation. CPU tests pass. This is **not** a CUDA/Triton speed-qualified release. The old per-head-C/Post-AVQ native kernels are incompatible with the new Shared-C layout and are deliberately not dispatched.

## Architecture

All modern BOLT modes take the input `X`, optionally run the **same Pre-AVQ memory** to produce `Z`, calculate attention, apply learned per-query-head output gates, and then the output projection. The number of heads `H` is user-configured; `head_dim = d_model / H`, requiring exact divisibility. No RoPE, ALiBi, learned positions, or other positional scheme is constructed by BOLT.

**AVQ defaults:** `avq=True`, `retention="decay"`, `avq_dim=d_model`, FP32 recurrent state, learned read gate and residual contribution. `retention="learned"` switches to token-dependent sigmoid retention, initialized at `forget_gate_init=0.95`. The decay variant projects only V/R (2×AVQ width); the learned variant projects A/V/R (3×AVQ width). Fixed decay coefficients cover multiple head-group timescales and are not trainable. `avq=False` removes AVQ from all three modes.

**Attention modes:**

| `mode` | Attention core | Persistent FP16 cache/token/layer |
| --- | --- | --- |
| `quantised` **default** | H Q heads, `kv_heads` K heads and `kv_heads` V heads (1=MQA, intermediate=GQA, H=full MHA) | `4 * kv_heads * head_dim` bytes |
| `Keyless` | H Q heads, one shared C head (`head_dim` channels), formed from U and G | `2 * head_dim` bytes |
| `base` | Full QKV MHA (`kv_heads=H`) | `4 * H * head_dim` bytes |

For Keyless:

```
Z = PreAVQ(X)                 # if avq=True; otherwise Z=X
Q = reshape(Wq(Z), B,H,T,head_dim)
U = Wu(Z)                    # B,1,T,head_dim
G = Wg(Z)                    # B,1,T,head_dim
C = U * (1+tanh(G))
O_h = Softmax(Q_h C.T / sqrt(head_dim) + causal_mask) C
Y = OutProj(Concat(2*sigmoid(HeadGate(Z))_h * O_h))
```

The new Keyless C cache holds **one** `[batch, 1, capacity, head_dim]` history, not `H` such histories. It is 50% of the K+V cache of single-KV-head MQA at the same model/head dimensions; this does **not** include FP32 AVQ state, model parameters, workspace, or transient attention memory.

For example, `d_model=384,num_heads=16` produces `head_dim=24` and shared C cache 48 B/token/layer FP16. With 12 query heads, head dimension is 32 and C cache 64 B/token/layer FP16.

## Public layer API

```python
from mlbricks import Bolt

# Default: MQA with 1 KV head, Pre-AVQ, decay retention, learned head gates.
layer = Bolt(d_model=384, num_heads=12)

# Grouped-query attention: 12 queries, 4 KV heads.
gqa = Bolt(384, 12, mode="quantised", kv_heads=4)

# Full KV/MHA: either of these forms.
full = Bolt(384, 12, mode="quantised", kv_heads=12)
base = Bolt(384, 12, mode="base")

# New shared-C Keyless: 16 queries, one shared C of width 24.
keyless = Bolt(384, 16, mode="Keyless", retention="decay")

# Same Keyless attention, learned token-wise retention.
learned = Bolt(384, 16, mode="Keyless", retention="learned")

# Disable memory to create an attention-only baseline.
baseline = Bolt(384, 12, avq=False, head_output_gate=False)
```

Additional supported constructor options include `avq_dim`, `forget_gate_init`, `bias`, `dropout`, `causal`, `device`, `precision`, `backend`. Existing `avq="enable"/"disable"` strings and `state_memory` bool remain recognized for migration. `kv_heads` is **not applicable** to Keyless, and non-divisor group counts are rejected. Backend `native` is explicitly unavailable for these new attention layouts until qualified kernels exist; choose `auto` or `pytorch` to use the correct PyTorch/SDPA implementation.

## Complete-model API

```python
from mlbricks import BoltConfig, BoltModel

config = BoltConfig(
    vocab_size=50257, context=2048, layers=6, dim=384, heads=12,
    bolt_mode="Keyless",  # or quantised (default), base
    avq=True, retention="decay", head_output_gate=True,
    # kv_heads=4,         # only with bolt_mode=quantised
)
model = BoltModel(config)
logits = model(input_ids)
logits, caches = model.prefill(input_ids[:, :16])
logits, caches = model.decode_step(input_ids[:, 16:17], caches, position=16)
```

`BoltConfig.layer_modes` continues to support hybrid attention stacks; `retention` is shared across their layers. For models using `BoltModelSettings` directly, set the corresponding `bolt_mode`, `avq_dim`, and other compatible fields appropriately.

## User-supplied positional transformations only

There is no built-in positional encoding and no automatic position choice. A caller can transform token embeddings before feeding them to `Bolt`, or supply an external token-local Q/K callable:

```python
layer.set_qk_transform(transform)
model.set_qk_transform(transform)
```

The callable must have signature `transform(tensor, *, start_pos: int) -> tensor`, preserving `[B,H,T,head_dim]` shape. It must operate per token consistently so full-sequence and decode are equivalent. It is applied to Q and K-like projections, not the stored C values. Passing the old string `position="rope"` now raises an actionable error.

## Decoder, migration, and compatibility

- New Shared-C Keyless is **not** the old per-head-C `BOLT32` checkpoint: it changes C projection widths, C sharing, AVQ placement, and cache data structures. Retraining is required for research quality claims; automatic checkpoint conversion is not offered.
- Fixed retention was already present for the MQA path in `optimised-avq`; now both retention policies can be selected across **all modern modes**, including Keyless.
- Old positional-encoding expectations and old Keyless 768 B/token cache tests describe the **previous architecture**. They are intentionally obsolete and must be migrated; do not reinterpret their failures as successful new-architecture tests.
- The `BoltStateMemory` existing learned-to-fixed load behavior can drop the learned retention projection rows. This is a weight-shape migration, **not** proof of prediction equivalence.
- Native BOLT/Triton kernels from the source branch are still present for legacy implementation purposes but are not used by the new modes; they must be redesigned for shared C and qualified with forward/backward/prefill/decode parity on target GPUs before using `backend="native"`.
- `mode="legacy"` remains an explicitly selected historical path. New models should choose one of the three modern modes.

## Verification performed

`python -m pytest -q tests/test_optimised_bolt_shared_c.py` — CPU, training gradients, cache width, full/decode parity, externally supplied Q/K transform, invalid argument handling, two-layer full-model decode and preallocated caches. GPU kernel/throughput, long-context quality and cross-version checkpoint migration have **not** been validated.
