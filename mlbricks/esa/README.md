# ESA — Fast-EsaQ only

MLBricks uses the Fast-EsaQ v0.4.0 half-life ESA mixer as its **only** ESA architecture.
There is no `architecture` option, `ThunderESA` model class, or old ESA
checkpoint loader. Original Thunder ESA weights require the original repository.

- `layer.py` — public tensor-returning Fast-EsaQ adapter
- `fast_engine/` — the Fast-EsaQ implementation, scan methods, and CUDA sources
- `model.py` — ESA language model using Fast-EsaQ exclusively
- `generation.py` — shared sampling/generation utilities (including low-level compatibility scan helpers)
- `backends/`, `native.py`, `planner.py` — internal scan/runtime utilities; not old selectable ESA architectures
- `compass.py`, `benchmark.py`, `boost.py` — tuning and benchmark helpers

For the supported usage and limitations, see [`FASTESAQ_MERGE_NOTES.md`](../../FASTESAQ_MERGE_NOTES.md).
