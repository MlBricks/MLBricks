## 1.0.0b1

- PyPI distribution name finalized as `mlbricks-core`; the Python import namespace remains `mlbricks`.

- Beta packaging release with prebuilt native wheel CI for Linux CUDA, Windows CUDA, and macOS CPU-native targets.
- Adds a portable `py3-none-any` fallback so unsupported platforms do not compile native code during installation.
- Native beta ABI is pinned to PyTorch 2.10.x; CUDA release wheels use a multi-architecture fat binary.
- Source installs now default native compilation off; official wheel CI opts native extensions in explicitly.
- Fixed beta wheel CI to install modern setuptools for no-isolation native builds, use the official PyTorch CUDA 12.8 index on Linux/Windows, fail fast on CPU-only PyTorch during CUDA release builds, and restrict macOS native wheels to Apple Silicon.
- Upgraded artifact upload/download actions to Node 24-capable releases.
- Forced macOS beta native builds to emit ARM64-only wheel tags and verify every packaged native binary with `lipo` before upload.
- Fixed Windows CUDA extension compilation with PyTorch 2.10 by defining `USE_CUDA` for CUDA builds and enabling the conforming MSVC preprocessor, activating PyTorch's built-in Windows CUDA header workaround.

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

- Release consistency: MLBricks package version is `1.0.0`; experimental SOUP retains its independent component version `0.1.0a3`.
