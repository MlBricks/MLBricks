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

### Fused BaseESA one-token inference (v5 review patch)

`ESA(embd=384, head=6, variant="base", backend="auto")` now uses a
lazy-compiled native CUDA kernel for inference with sequence length one,
following the same design as the current Half-Life variant. The recurrence,
learned bounds, checkpoints and long-sequence/training scan path are unchanged.
The kernel combines the learned-bounds gate, FP32 recurrent update, RMSNorm
and query readout; matrix projections remain PyTorch linear operations.

For strict CUDA one-token execution use `backend="native"` within
`torch.inference_mode()`. Call
`mlbricks.esa.fast_engine.base_decode.prepare_base_decode()` once *before*
benchmarking or CUDA Graph capture to avoid compiling during the timed path.
`backend="auto"` warns and falls back if the extension cannot be compiled;
`backend="pytorch"` retains the original reference path. The new kernel
uses separate `base_decode.cu` and `base_decode_bindings.cpp`, leaving the six
original Fast-EsaQ CUDA/C++ source files unmodified.
