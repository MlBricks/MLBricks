# ESA

This folder contains the Entangled State Attention component implementation:

- `layer.py` — public ESA layer
- `generation.py` — prefill/decode helpers
- `backends/` — Thunder/Lightning execution code
- `native.py` — native dispatcher/runtime integration
- `planner.py` — resource-aware execution planning
- `precision.py` / `constants.py` — ESA numeric policy
- `model.py` / `trainer.py` — ESA model and training utilities
- `compass.py`, `benchmark.py`, `boost.py` — ESA tuning and benchmark helpers

The compiled `mlbricks._C` extension remains a shared native extension under
`mlbricks/bolt` because some kernels are shared with ElasticBit. Public imports
remain backward compatible.
