# MLBricks BOLT-16 + AVQ-192: paper math and inference-only folded Q

This is an **opt-in** paper-compatible preset. Existing default BOLT configurations, checkpoint shapes, CUDA training kernels, and trainable P are retained.

## New usage

```python
import torch
from mlbricks.bolt.presets import BoltPreset

bolt = BoltPreset.paper_bolt16_avq192(384, 12, backend="auto")

# Train/retrain normally: P is a true trainable nn.Parameter.
bolt.train()
assert bolt.p.requires_grad
# ... optimizer/training here ...

# Inference: fold P into cached Q weights, no Q * P or Q * 1 operation.
bolt.eval().fuse_for_inference()
with torch.inference_mode():
    out = bolt(torch.randn(1, 16, 384, device=next(bolt.parameters()).device))

# Training later still uses the original Q and P; caches are invalidated.
bolt.train()
# ... retrain, optimizer updates P and Q ...
bolt.eval().fuse_for_inference()
```

`fuse_for_inference()` leaves the Q weights, P parameter, optimizer state, and serialized checkpoint unchanged. A **nonpersistent** folded copy of the Q projection weights (and Q bias, when present) is cached for eval/no-gradient calculations only. The cache is invalidated on re-entering train mode, loading checkpoints, or in-place updates to the original Q or P tensors.

## Paper configuration

- `cache='tight'`: 16 latent features/head; 12 heads -> 192 attention-output features.
- `avq_dim=192`: 192 recurrent AVQ features.
- `avq_return_map='identity'`: AVQ readout adds to the 192-feature attention vector without a 192x192 return projection.
- `avq_norm_eps=torch.finfo(torch.float32).eps`: equivalent to `F.rms_norm(state_float32, ..., weight)` with the default epsilon. The FP32 recurrent state is preserved.
- `projection='packed'`, `attention_scale='head'`, `memory_init='legacy'`, no position encoding.
- Trainable `P` is preserved, with no initialization/weight remapping during training.

Note: AVQ's **A/V/R input projection** remains necessary; only its redundant **192-to-192 return map** becomes identity.

## Native CUDA support

A new `features_forward_folded(q,u,g)` entry point skips reading or multiplying by P; it computes the same C features as the old native feature kernel. Rebuild the `mlbricks._bolt_modern_cuda` extension to use it. If an older extension is installed, the inference path falls back to PyTorch gated features, while retaining native AVQ and decode routines where available. The original native training kernel and its backward path are unchanged.

## Validation

On the available CPU environment:

- Seven new paper-math/fold-inference tests passed (packed/separate, bias/no bias, nonunit P, cached decode, checkpoint stability, P gradient, FP32 RMSNorm parity).
- Existing selected BOLT suite: 38 passed / 16 skipped; other BOLT tests: 34 passed / 6 skipped.
- No CUDA GPU was available, so the new native CUDA entry point was not compiled or benchmarked here.
- Floating-point output parity has been checked on CPU; five-epoch perplexity is **not** claimed reproduced by this patch alone.

To re-run:

```bash
python -m pip install -e .
python -m pytest -q tests/test_bolt_paper_math_folding.py
```
