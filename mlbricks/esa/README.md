# ESA

This folder contains the Entangled State Attention component implementation:

- `selector.py` — public mode selector (`retention` default, `legacy` opt-in)
- `layer.py` — legacy Thunder ESA layer
- `generation.py` — prefill/decode helpers
- `backends/` — Thunder/Lightning execution code
- `native.py` — native dispatcher/runtime integration
- `planner.py` — resource-aware execution planning
- `auto_backend.py` — performance-aware `auto` routing with a 5% switch margin
- `precision.py` / `constants.py` — ESA numeric policy
- `model.py` / `trainer.py` — ESA model and training utilities
- `compass.py`, `benchmark.py`, `boost.py` — ESA tuning and benchmark helpers

The compiled `mlbricks._C` extension remains a shared native extension under
`mlbricks/bolt` because some kernels are shared with ElasticBit. Public imports
remain backward compatible.

## Auto backend routing

`ESA(..., mode="legacy", backend="auto")` preserves explicit `native`/`pytorch` overrides for historical Thunder ESA.
For eager inference, each ESA element now uses the shared correctness-first
planner: it runs the PyTorch reference and native candidate once, rejects native
if outputs do not match within the numeric tolerance, benchmarks only valid
routes, and freezes the winner for that ESA instance. Active `torch.compile`
tracing and training keep ESA's established qualified routing path so planner
microbenchmarks never execute inside a captured graph. The existing SM 7.5
profile remains available for those compile/training routing cases.


## Current half-life ESA (FastESA 0.4.0)

The benchmarked half-life ESA is now the **default** public architecture,
selected by `mode="retention"`. Its six trainable FP32 half-life parameters are initialized
directly to `[0.125, 0.5, 2, 8, 32, 64]`, with no `/128` initialization hack.

```python
import torch
from mlbricks import ESA, LegacyESA, RetentionESA

esa = ESA(embd=384, head=6, backend="auto", device="cpu")  # mode="retention"
y, state = esa.prefill(torch.randn(2, 32, 384))
assert isinstance(esa, RetentionESA)

legacy = ESA(embd=384, head=6, mode="legacy", backend="auto", device="cpu")
legacy_y = legacy(torch.randn(2, 32, 384))   # tensor only
assert isinstance(legacy, LegacyESA)
```

**API difference:** Retention `forward(x, state=None)` returns `(y, new_state)`;
legacy `forward(x)` returns `y`. Legacy `ESAModel`, VESA and other existing
internal wrappers remain wired to `LegacyESA` to preserve their saved weights,
forward signatures and recurrent-state layouts. Those wrappers have **not**
been automatically migrated to the retention architecture.

Checkpoints are architecture-specific; do **not** load Thunder ESA weights into
retention ESA or vice versa. The selector does not introduce a `.impl.` or
`.engine.` prefix into `state_dict` keys. `ESAConfig` and the older `ESAModel`
are legacy-oriented; for retention LM models use the separate FastESA helpers.

Use `from mlbricks import CurrentESAModel, make_current_esa_model,
current_esa_forward_tokens` for the separately exported FastESA model family.
Their pretrained weights are not bundled; these are initialized from scratch.

Both ESA engines keep their own state dictionaries; FastESA's
`save_current_esa_checkpoint` and `load_current_esa_checkpoint` preserve the
new current architecture and reject mismatched historical ESA variants.
For GPU use, FastESA native CUDA kernels build lazily; validate CUDA paths
with a Kaggle T4 before production or publishing a wheel.
