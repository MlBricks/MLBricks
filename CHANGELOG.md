# MLBricks Changelog

## 1.0.0 BOLT compound Stage-1 optimization

- Added a native CUDA compound Stage-1 for FP16 Bolt: packed Q/U/G projection plus fused gate, C-energy reduction and RMS normalization.
- Added the exact Stage-1 adjoint so training does not retain standalone U/G tensors; the public q/c/g Parameters and checkpoints are unchanged.
- Restored the normalized-key SDPA training identity when `use_sdpa=True`; `use_sdpa=False` remains the explicit-order reference path.
- Prefill and recurrent decode can reuse the same compound Stage-1 while preserving the compact FP16 `C + rho` cache.

## 1.0.0 unified lifecycle update

- Added architecture-agnostic `mlbricks.save()`, `mlbricks.load()`, and `mlbricks.inspect()`.
- Replaced the ESA-specific trainer with the generic `mlbricks.Trainer` and `mlbricks.train()` APIs.
- Added `Trainer.fit()`, `Trainer.evaluate()`, generic checkpointing, and `Trainer.resume()`.
- Added package-level `predict`, `generate`, `compile`, and `quantize` helpers.
- Removed `ESAModel.save()`, `ESAModel.load()`, and `mlbricks.esa.Trainer` from the public API.
- Added unified mixed-model lifecycle tests, including ESA + Bolt `Bricks` save/load and resume.

## 1.0.0

- Clean v1.0.0 public release.
- Standardized package and bundled component version metadata to `1.0.0`.
- `Bolt` and `BoltAttention` are the canonical public Bolt attention APIs.
- Consolidated the validated Bolt implementation under `mlbricks.bolt`.
- Preserved existing computation, state-dict structure, planner behavior, and native acceleration paths.
- Added `API.md` for the v1.0.0 public API.

### ESA native training

- Enabled explicit `backend="native"` Thunder ESA training when the CUDA native extension and registered autograd operators are available.
- Native training now routes differentiable Thunder scans through the existing chunked native backward path without requiring `MLBRICKS_NATIVE_TRAINING=1`.
- Fused native readout/full-forward kernels remain inference-only and are explicitly blocked while gradients are enabled.

