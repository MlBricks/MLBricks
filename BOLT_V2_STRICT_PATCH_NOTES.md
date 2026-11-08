# BOLT V2 strict — patch against supplied MLBricks-main ZIP

This patch adds an **opt-in**, inference-only native CUDA AVQ postprocessing
path for the modern BOLT-16/AVQ-192 and BOLT-32/AVQ-384 modes. Existing
modern native attention, PyTorch fallbacks and training stay unchanged.

The V2 CUDA kernel is taken from the experimentally tested strict AVQ path,
retaining explicit FP32 separate multiplication/addition. The **PyTorch
projection** is intentionally retained (no V3 custom GEMV).

**Build:** `MLBRICKS_BUILD_BOLT_V2_STRICT=1 MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1 python -m pip install -v --no-build-isolation .`

**Enable:** `MLBRICKS_BOLT_V2_STRICT_DECODE=1` at process launch.

**Default:** off; unsupported shapes/dtypes and training paths unaffected.

**Validation limitation:** The uploaded ZIP is not necessarily GitHub main.
This patch is based on that ZIP and requires CUDA integration tests on T4
and other supported GPUs before being merged into main. No GitHub changes
were made. A passing 24-layer synthetic stress test does not prove trained
24-layer model quality or exact bitwise output equivalence.
