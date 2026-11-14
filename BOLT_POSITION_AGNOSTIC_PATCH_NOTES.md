# BOLT position-agnostic patch

Branch (locally created from the uploaded source ZIP): `feat/bolt-position-agnostic`

## What changed

- **No built-in RoPE in BOLT.** Removed BOLT's RoPE constructors, RoPE-only dispatch logic, CUDA kernels and Python autotuner, plus native binding exports.
- **No silent position changes.** `position="rope"` and `position="rotary"` now raise an explicit error instead of silently turning the encoding off.
- **External Q/K hook.** Callers may still pass an externally supplied, token-local transform implementing `(tensor, *, start_pos=int)`. The accepted interface is the existing `position=` argument or `.set_position(external_transform)`; complete models additionally offer `.set_qk_transform(external_transform)`. This preserves autoregressive token indexing for transformed Q/K and correct key-cache behavior.
- **Default architecture unchanged.** `position=None` / `position="none"` behaves like the original default. Enhanced MQA, Classic MHA, Keyless, and legacy mode retain their untransformed attention computation.
- **Input embeddings remain distinct.** Model-level learned and sinusoidal *additive input* embeddings are still available. The broader MLBricks `mlbricks.position` module remains untouched to avoid breaking unrelated components (e.g. Bricks and vision engines).

## Migration

Previously:

```python
layer = Bolt(128, 16, mode="quantised", position="rope")
```

Now:

```python
# You implement/provide ExternalQK in your own model code.
layer = Bolt(128, 16, mode="quantised", position=ExternalQK())
```

Complete BOLT language model:

```python
model = BoltModel(vocab_size=100, context=2048, layers=6,
                  dim=128, heads=16, position="none")
model.set_qk_transform(ExternalQK())
```

**External transform contract:** input/output `[B,H,T,D]` with identical shape; `start_pos` gives the absolute token position. The transform must be deterministic and token-local to preserve full-sequence vs cached-decoding parity. Do not change it with a live cache. If it is an `nn.Module`, move it to the same device as the model or attach it through `model.set_qk_transform` (which moves the module).

**Backend:** native BOLT decode only supports the untransformed core. For external Q/K transforms use `backend="auto"` (portable PyTorch fallback where required), or `backend="pytorch"`; `backend="native"` may error. ALiBi and other score-based biases require a separate attention-score bias integration point. This patch does not claim that arbitrary score biases work through the Q/K hook.

**Compatibility:** old checkpoints that relied on built-in RoPE need their original RoPE behavior supplied externally before use. This is a deliberate API change rather than a silent removal. Existing checkpoints with no position encoding retain the same parameter tensors.

## Validation

- PyTorch CPU positional tests: 45 passed, including 16-query-head/1-KV-head MQA, Keyless and Classic cache parity, explicit string rejection, and external model-level Q/K hook.
- Full original archive: 61 failed, 313 passed, 100 skipped (existing failures from deprecated compact/BOLT16 and other unrelated tests).
- Full patched tree: 61 failed, 320 passed, 100 skipped (no new failing test functions; seven additional passing tests).
- `git diff --check` and Python compile checks pass. CUDA/native compilation and GPU execution have **not** been verified in this CPU-only runtime.

## Applying patch to an actual remote clone

The uploaded ZIP contains no `.git` remote or authenticated GitHub write capability. The new branch is a local branch based on an imported ZIP snapshot. To publish changes on your GitHub repository, start with the matching upstream commit/branch (the source snapshot must match the patch) and run:

```sh
git switch <matching-source-branch>
git switch -c feat/bolt-position-agnostic
git apply --check /path/to/bolt-remove-rope.patch
git apply /path/to/bolt-remove-rope.patch
git add -A
git commit -m "Make BOLT position-encoding agnostic"
git push -u origin feat/bolt-position-agnostic
```

Never apply to a different upstream revision without reviewing resulting conflicts. This local branch cannot itself be pushed as a continuation of an unseen upstream history; apply the patch on a genuine matching GitHub clone.
