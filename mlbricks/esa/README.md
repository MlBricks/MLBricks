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

`ESA(..., backend="auto")` preserves explicit `native`/`pytorch` overrides and
uses qualified workload-aware routing where benchmark evidence exists. The first
qualified profile is CUDA SM 7.5 (Tesla T4-class), using a 5% hysteresis margin
so small benchmark differences do not cause route changes. Unqualified GPU
architectures keep the historical native-first behavior until separately
benchmarked.
