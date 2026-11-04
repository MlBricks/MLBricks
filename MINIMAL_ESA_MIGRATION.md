# Minimal MLBricks ESA v4 — intentional breaking API cleanup

## Scope
Only the ESA integration is simplified. BOLT, SOUP, VESA, FFNBrick,
ResController, ElasticBit and other research components remain installed.

## One public ESA API
```python
from mlbricks import ESA
mixer = ESA(embd=384, head=6, variant="current", backend="auto", device=None)
y, state = mixer(x)       # [B,T,D], [B,H,D/H] FP32
step, state = mixer.decode_step(token, state)  # [B,1,D]
```
`variant="base"` selects the separately retained historical BaseESA.

The single ESA public class is backed directly by `mlbricks.esa.fast_engine.public_api.ESA`.
No `ESAAdapter`, duplicate top-level `esa` constructor, `mlbricks.layer`,
`mlbricks.esa.layer`, or top-level adapter `ESAConfig` is exported.

`ESAModel`, VESA and SOUP construct native Fast-EsaQ instead of the adapter.
VESA retains its VESA-specific flattened-state boundary, but never wraps an
additional ESA submodule. Old saved VESA checkpoints may require key migration.

Existing ESAModel Fast-EsaQ checkpoints with `blocks.*.esa.layer.*` keys
are translated once at load time. Newly saved checkpoints use native
`blocks.*.esa.*` keys. Old Thunder (non-Fast-EsaQ) weights are rejected.

`ESA.half()`, `ESA.bfloat16()`, `ESA.to(dtype=...)` preserve Half-Life
retention and normalization controls in FP32, required by native fused CUDA.
This is a precision safety fix; no equation or CUDA kernel was changed.

SOUP's unused packed decoding path for obsolete `gate_min/gate_max` was removed.
No untested replacement of its recurrence is introduced; it uses native
Fast-EsaQ prefill and decode.

## Intentional compatibility breaks
Older code importing `ESAAdapter`, top-level `ESAConfig`, `mlbricks.layer.ESA`,
or `mlbricks.esa.layer.ESA` must switch to `from mlbricks import ESA`.
`ESA.forward()` now returns `(output, state)` instead of a bare tensor.
Old checkpoint/optimizer states should be backed up before migration.

## Validation
Run `python -m pytest -q tests/test_minimal_esa_v4.py`, other maintained
integration tests, and GPU parity/throughput benchmarks before merging.
The old wrapper-specific test suite requires migration, not suppression.
