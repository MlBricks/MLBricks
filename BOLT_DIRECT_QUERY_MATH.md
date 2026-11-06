# BOLT direct-query training and inference

Modern `Bolt` uses the paper's direct-query equation in both training and inference:

    Q = X @ W_Q

There is **no separate query scale P** to train, multiply, fold, cache or restore.
The Q projection remains fully trainable. Inference simply uses the same learned
projection, without an extra query-scaling operation.

```python
from mlbricks import Bolt

bolt = Bolt(d_model=384, num_heads=12, profile="compact")
bolt.train()  # direct Q projection trains normally
bolt.eval()   # direct Q projection, no additional action
```

The compact 12-head model uses AVQ-192 with Identity return map and the FP32
RMSNorm default epsilon. Other explicitly selected architectures and settings
remain supported. The historical `mode="legacy"` BOLT architecture is not
changed by this patch; it is distinct from modern BOLT-AVQ.

The patch changes the model parameterization relative to the previously merged
modern learned-P version. It has no separate-P model-loading compatibility mode.
CPU parity and gradients are covered by the tests. CUDA compilation and T4/H200
performance remain to be checked before merging to main.
