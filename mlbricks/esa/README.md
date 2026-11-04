# ESA — Fast-EsaQ only

MLBricks uses original Fast-EsaQ v0.4.0 for `ESA` and `ESA(variant="base")`.
`current` is Half-Life ESA; `base` is the historical learned-bounds variant.
Both are selected through the single direct public `ESA` class.
There is no `architecture` option, `ThunderESA` model class, or old ESA
checkpoint loader. Original Thunder ESA weights require the original repository.

- `fast_engine/` — the Fast-EsaQ implementation, scan methods, and CUDA sources
- `model.py` — ESA language model using direct native Fast-EsaQ (no adapter)
- `generation.py` — shared sampling/generation utilities (including low-level compatibility scan helpers)
- `backends/`, `native.py`, `planner.py` — internal scan/runtime utilities; not old selectable ESA architectures
- `compass.py`, `benchmark.py`, `boost.py` — tuning and benchmark helpers

For the supported usage and limitations, see [`FASTESAQ_MERGE_NOTES.md`](../../FASTESAQ_MERGE_NOTES.md).
