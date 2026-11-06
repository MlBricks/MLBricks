# BOLT: attention-width-matched AVQ by default

The default AVQ recurrent-memory width is the total BOLT attention width:

`avq_dim = num_heads * latent_dim`

This makes the AVQ **return map** `nn.Identity()`. It removes the additional
`avq_dim -> attention_width` matrix multiplication and its parameters. The
A/V/R input projection, recurrent update and normalization remain intact.

For the 12-head public profiles:

| Model | Attention channels | Default AVQ | Return |
|---|---:|---:|---|
| BOLT-16 (`profile="compact"`) | 192 | 192 | Identity |
| BOLT-32 (`profile="standard"`) | 384 | 384 | Identity |

```python
from mlbricks import Bolt

small = Bolt(d_model=384, num_heads=12, profile="compact")
large = Bolt(d_model=384, num_heads=12, profile="standard")
```

The same automatic sizing applies to `BoltConfig` / `BoltModel` and to
`BoltPreset`. Direct-Q is the standard training and inference math.

### Explicit width overrides

Nonmatching widths remain an **explicit research ablation**, e.g.
`Bolt(d_model=384, num_heads=12, profile="standard", avq_dim=194)`.
That configuration requires a learned linear return map; setting
`avq_return_map="identity"` with a mismatched width raises `ValueError`.

### Trade-off

Matching widths removes one projection, but AVQ-384 processes twice as many
recurrent values as AVQ-192. **End-to-end speedup is not guaranteed** and
requires GPU profiling. Native CUDA kernels must still be validated on GPU.

This patch builds on the Direct-Q patch; it does not merge to main automatically.
