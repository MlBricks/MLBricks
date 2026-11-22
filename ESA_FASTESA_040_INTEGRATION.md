# MLBricks + FastESA 0.4.0 integration

**Source inputs:** user-provided `MLBricks-main (3)(20261122-151624).zip` and
`fast-EsaQ-main (2)(1).zip` (which includes merged default half-lives).

## Scope

- Add FastESA v0.4.0 source under `mlbricks/esa/engine/` without altering its math.
- Expose `ESA(mode='retention'|'legacy')`, defaulting to retention,
  plus `LegacyESA`, `RetentionESA`, `CurrentESA`, `CurrentESAConfig`, `CurrentESAModel`,
  `make_current_esa_model`, checkpoint helpers and `make_esa`.
- Set direct defaults to `[0.125, 0.5, 2, 8, 32, 64]` as shipped in source.
- Preserve Thunder legacy behavior through `LegacyESA` and `ESA(mode='legacy')`.
  Keep historical ESAModel/VESA/SOUP wrappers on their original concrete engine;
  do not reinterpret state dictionaries.
- Include FastESA lazy CUDA `.cu`/`.cpp` source in the distribution manifest.

## Quick check

```python
import torch
from mlbricks import CurrentESA, ESA, LegacyESA, RetentionESA
new = ESA(embd=384, head=6, backend='pytorch', device='cpu')
assert isinstance(new, RetentionESA)
assert torch.allclose(new.log_half_life.detach().exp(),
    torch.tensor([.125,.5,2,8,32,64]))
y, state = new(torch.randn(1,8,384))
assert y.shape == (1,8,384) and state.shape == (1,6,64)
old = ESA(embd=32, head=4, mode='legacy', precision='fp32', backend='pytorch', device='cpu')
assert isinstance(old, LegacyESA)
assert old(torch.randn(1,8,32)).shape == (1,8,32)
```

## Limitations

- Retention `ESA` is now the default. Historical model wrappers, SOUP and VESA
  still explicitly use the legacy engine; their checkpoints are untouched.
- MLBricks's general `save`, `load`, `predict`, `Trainer` methods are not
  automatically aware of the new model family; use FastESA-specific helpers.
- CUDA kernel parity and performance need separate GPU verification.
- Git patch was produced from the user-uploaded ZIP snapshot, not verified
  against a live GitHub main checkout. Apply only after `git apply --check`.
