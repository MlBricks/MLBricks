# BOLT public API integration (on top of commit e1682c2)

Users import the existing public class, with automatic CUDA/CPU device policy:

```python
import torch
from mlbricks import Bolt

bolt = Bolt(d_model=384, num_heads=12, profile="compact")

# Training: original Q and trainable P remain ordinary parameters.
bolt.train()
y = bolt(x)  # MLBricks handles automatic input placement
loss = criterion(y, target)
loss.backward()
optimizer.step()

# Inference: no explicit folding method or extra Q × 1 multiplication.
bolt.eval()
with torch.inference_mode():
    y = bolt(x)
```

A new compact **12-head** BOLT defaults to **AVQ-192**, an identity
memory return, FP32-default epsilon (`torch.finfo(torch.float32).eps`),
head-dimension attention scaling and legacy AVQ initialization.
The separate Q scaling `P` is still trainable; automatic folding occurs
only when evaluation and gradient computation is disabled, and never
changes the trainable `Q` and `P` tensors or checkpoint keys.

`BoltModel(BoltConfig(profile="compact", heads=12, ...))` also chooses
AVQ-192/identity by default. Explicit `state_dim`/`avq_dim` can override
this. Existing 198-wide standard presets are unchanged. To load a legacy
checkpoint created with a linear AVQ return map at matching dimensions,
construct `Bolt(..., avq_dim=192, avq_return_map="linear", avq_norm_eps=1e-6)`.

The prior `BoltPreset.paper_bolt16_avq192` and manual fusion helpers remain
available as compatibility internals, but they are not required or
recommended in the end-user API.

CUDA kernel compilation and end-to-end GPU benchmarks must be verified
separately; the patch's included regression tests are CPU-safe.
