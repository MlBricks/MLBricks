# MLBricks BOLT-16 + AVQ-192: paper mathematics

Modern BOLT trains the query projection directly, exactly as in the paper
benchmark: `Q = X @ W_Q`. There is no separately trainable query scale `P`,
no multiplication by `P` and no inference-specific Q folding step.

```python
import torch
from mlbricks import Bolt

bolt = Bolt(d_model=384, num_heads=12, profile="compact")
bolt.train()  # direct Q weights train normally
# ... optimizer and training ...
bolt.eval()
with torch.inference_mode():
    result = bolt(torch.randn(1, 16, 384))
```

The validated compact 12-head configuration uses BOLT latent width 16,
AVQ-192, an Identity AVQ return map (no 192x192 back projection),
FP32-default RMSNorm epsilon and FP32 recurrent state. The AVQ A/V/R input
projection is still present.

The optional native CUDA extension provides a no-P BOLT feature kernel for
inference, where supported, and an AVQ scan. There is no native query-scale
multiplication in the modern direct-Q forward path. Native CUDA compilation
and end-to-end performance must be tested on an NVIDIA GPU.
