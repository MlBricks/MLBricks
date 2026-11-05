# MLBricks BOLT professional public interface

The public interface is `from mlbricks import Bolt` with automatic device
selection. No research-named constructor, manual folding call or `query_scale`
argument is needed or supported by modern BOLT.

```python
from mlbricks import Bolt

bolt = Bolt(d_model=384, num_heads=12, profile="compact")
bolt.train()  # train direct Q weights; no P parameter
bolt.eval()   # same Q weights, no folding operation
```

The compact 12-head preset defaults to AVQ-192, Identity return,
FP32 RMSNorm epsilon, head-dimension score scaling and the paper AVQ
initialization. Other explicit AVQ state widths and return maps remain
configurable. The historical `mode="legacy"` BOLT architecture is separate
and has not been refactored by this direct-Q change.

Changes to modern BOLT parameterization require a fresh model training run.
GPU validation, CUDA extension compilation and longer quality comparisons
remain necessary before promoting the change to a release.
