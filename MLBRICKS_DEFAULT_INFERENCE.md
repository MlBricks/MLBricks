# MLBricks automatic inference (v3 experimental candidate)

### Public API

Use `BoltModel`, `AttentionModel`, standalone `Bolt` and `Attention` as normal.
New modules start on CUDA when available, and `model.generate(ids, 64)`
now performs **greedy** decoding by default on both model types.
Sampling remains opt-in using `temperature > 0`.

### FP32 training, FP16 inference mirror

The primary model is **not** converted to FP16. Training and its optimizer
use the original FP32 parameters. On the strictly validated T4 greedy
configuration (batch 1, no positional encoding for BOLT, standard FFN and
residual, no dropout), generation creates a version-guarded, nonregistered
FP16 inference mirror and runs Strict CUDA Graph on that mirror.

The mirror preserves parameters after FP32-to-FP16 conversion. **It does not
guarantee FP32/FP16 numerical equivalence**: graph logits and caches should
match an FP16 *eager* reference, not FP32 eager. The mirror is outside
`state_dict()` and `.parameters()`. Training mode, parameter/buffer updates,
and `.to()`/`.half()` invalidate it. CUDA OOM during mirror creation may
fall back to the original FP32 eager path. First inference incurs an extra
cast/copy and cache memory. CUDA Graph capture is still per request.

CPU, sampling, non-T4 devices and unsupported options use original eager
inference with original model dtype, without creating the mirror.

Use `model.inference_info()` to inspect actual selected execution and dtype.
No user-visible backend, graph or precision flags are needed.

**Experimental.** Full integrated T4 parity and real-trained-checkpoint
validation required before production-default merging.
