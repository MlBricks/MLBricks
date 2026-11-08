# MLBricks BOLT — width-aware native attention + autocast patch

**Target:** pinned research feature branch `feature/bolt-v2-strict-native-decode`, commit `d136f719334920e4240600911c9076ab0462f08f`. **NOT** `main` and not the published `mlbricks-kit` wheel. This ZIP contains a *patcher*, not a prebuilt CUDA wheel. The current feature-branch source itself is not bundled and must be checked out from the user's repository.

## What changes

1. `mlbricks/bolt/native_backend.py`: When explicitly opted in to the already-validated split reader, automatically select **1,536 tokens for latent R=16** and **896 tokens for latent R=32**. These are head-local cache widths, *not AVQ state widths*. The established 512-token chunk remains. Unknown widths conservatively use 8,192 tokens.
2. `mlbricks/bolt/presets.py`: In *inference-only* AVQ state-memory `step`, BOLT `decode_step`, and `decode_with_cache`, use `torch.autocast(... dtype=float16)` whenever the model's weight dtype is FP16 on CUDA/CPU. Critically: **do not manually convert the FP32 attention residual or AVQ recurrent state to FP16**. This reproduces the autocast policy used by the successful Wikipedia-trained Kaggle run. FP32 public/default models remain unchanged.
3. `tests/test_bolt_width_dispatch_policy.py`: CPU tests with a mocked native backend for crossing both thresholds, exact dispatch counters, explicit overrides, error handling, and mixed-dtype output/state preservation.

**Not included:** CUDA Graph arbitrary-position production streaming, output-head fusion, or new native kernels. The supplied `experimental_split_context_cuda.patch` is the previously tested source patch that adds the opt-in split CUDA path; the runner applies it only if missing. No model parameter layout or training path changes. The published `mlbricks-kit` API (including `Bolt`, `BoltConfig`, `BoltModel`) is not renamed.

## Install into a fresh branch (on Kaggle / Linux or Windows)

From your already cloned MLBricks repository:

```bash
# Start from the pinned V2 Strict research commit; make a NEW branch
# (Requires the commit to be available locally; fetch the feature branch if needed.)
git checkout -b feature/bolt-width-aware-decode d136f719334920e4240600911c9076ab0462f08f

# PACKAGE refers to the directory you extracted from this ZIP
python /path/to/MLBricks_BOLT_Library_Patch/scripts/patch_existing_feature_branch.py --repo .
```

The runner checks the split extension source, applies the width-aware dispatch and autocast changes, runs CPU tests, and checks `git diff --check`. It intentionally **does not** build a wheel, commit, push, or merge.

If the original feature branch is unavailable locally:

```bash
git fetch origin feature/bolt-v2-strict-native-decode
```

### Opt-in decoding policy

Set the environment before decoding (or process startup):

```python
import os
os.environ['MLBRICKS_BOLT_SPLIT_CONTEXT_READ'] = '1'
os.environ['MLBRICKS_BOLT_SPLIT_MIN_TOKENS'] = 'auto'  # IMPORTANT: remove old hardcoded 8192
os.environ['MLBRICKS_BOLT_SPLIT_CHUNK_TOKENS'] = '512'

from mlbricks import Bolt, BoltConfig, BoltModel
```

Without `MLBRICKS_BOLT_SPLIT_CONTEXT_READ=1`, the original single-block path remains the default. `MLBRICKS_BOLT_SPLIT_MIN_TOKENS=4096` overrides the per-width dispatch for reproducible tests. Setting `auto` (or leaving this variable unset) selects the per-width policy *when split reads are enabled*.

The actual native split CUDA extension must be rebuilt after adding the split patch; follow the research branch's existing native build instructions (`MLBRICKS_BUILD_BOLT_MODERN_NATIVE=1`). Avoid pip-installing the published beta over the checked-out patched source.

## Test status

- CPU dispatch and dtype unit tests: **11/11 passed** on the available source fixture.
- Compatibility with an emulated V2 Strict method signature: **AST validated** (simulated V2 Strict method; actual pinned feature branch still requires retest).
- Feature branch at the pinned commit: **not accessible for clone in this environment**. The user must run this patcher there; it checks for unmatched source anchors and aborts instead of guessing.
- CUDA compile, GPU parity, trained exact-token generation, 24L context stress, graph and QKV speed benchmark: **not yet rerun after this library patch**.

### Release gate

Do not merge into `main` until native CUDA tests, V2 Strict AVQ state and exact-token parity, error tolerance between legacy/split, both-T4 512–32K graph benchmarks, a real streaming test, and published API regression tests have passed.

### Troubleshooting

- **No `decode_append_split`:** the embedded split patch hasn't been applied or extension wasn't rebuilt.
- **No speed change:** split flag unset; old notebook environment still pins `MLBRICKS_BOLT_SPLIT_MIN_TOKENS=8192`; installed `mlbricks-kit` wheel shadows the local source; or graphs are not used in the final app.
- **Different token generation:** do not claim parity. Revert by restoring the two changed Python modules from Git and compare against the original branch.

No commit, push, library release, or `main` merge is performed by the scripts.
