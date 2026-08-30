# MLBricks Changelog

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
