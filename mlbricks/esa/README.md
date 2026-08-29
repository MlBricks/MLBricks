# ESA

This folder contains the Entangled State Attention component implementation:

- `layer.py` — public ESA layer
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

`ESA(..., backend="auto")` preserves explicit `native`/`pytorch` overrides.
For eager inference, each ESA element now uses the shared correctness-first
planner: it runs the PyTorch reference and native candidate once, rejects native
if outputs do not match within the numeric tolerance, benchmarks only valid
routes, and freezes the winner for that ESA instance. Active `torch.compile`
tracing and training keep ESA's established qualified routing path so planner
microbenchmarks never execute inside a captured graph. The existing SM 7.5
profile remains available for those compile/training routing cases.
