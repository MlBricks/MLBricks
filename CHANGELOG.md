# MLBricks Changelog

## 1.0.0 BOLT training performance restoration

- Restored the historical Gauss/BOLT 0.2 full-sequence training route: one autograd-safe packed Q/U/G projection followed by normalized-key PyTorch SDPA.
- Preserved the original `q_proj`, `c_proj`, `g_proj`, and `out_proj` Parameters/state-dict keys; no checkpoint or architecture change.
- `use_sdpa=False` remains the explicit-order reference/debug path.
- Prefill and recurrent decode/cache behavior are unchanged; compact BOLT generation continues to store `C + rho`.
- No Stream2/custom-backward path is used for this training optimization.

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

