# MLBricks BOLT-16 + AVQ-192: paper math and inference-only folded Q

The low-level implementation preserves the tested AVQ mathematics. End users configure the existing public `Bolt` class; research-specific factories and explicit folding controls are not part of the public API. Existing checkpoint shapes and trainable P are retained.

## New usage

```python
import torch
from mlbricks import Bolt

bolt = Bolt(d_model=384, num_heads=12, profile="compact")

bolt.train()  # Q and P trainable
# ... optimizer/training here ...

bolt.eval()  # folded Q inference is automatic under no_grad/inference_mode
with torch.inference_mode():
    output = bolt(torch.randn(1, 16, 384))

bolt.train()  # original Q and P remain trainable
```

Automatic folded inference leaves the Q weights, P parameter, optimizer state, and serialized checkpoint unchanged. A **nonpersistent** folded copy of the Q projection weights is cached for eval/no-gradient calculations only.


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
