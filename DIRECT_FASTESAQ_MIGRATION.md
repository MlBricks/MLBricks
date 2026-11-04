# MLBricks native Fast-EsaQ v0.4.0 integration

**Direct public API:** `from mlbricks import ESA` and `from mlbricks.esa import ESA`
now resolve to **the original vendored Fast-EsaQ ESA class**, not the
MLBricks `nn.Module` adapter. `mlbricks.layer.ESA` is also native.
The upstream `public_api.py` source is byte-identical to Fast-EsaQ v0.4.0.

```python
from mlbricks import ESA
m = ESA(embd=384, head=6, variant="current", backend="auto", device="cuda")
y, state = m(x)            # x [B,T,D]; y [B,T,D]
y, state = m.decode_step(token, state)  # token [B,D] or [B,1,D], y [B,1,D]
```

- `variant="current"` is Half-Life ESA. `variant="base"` is the historical learned-bounds BaseESA.
- `forward` returns `(outputs, recurrent_state)`, unlike the historical MLBricks layer.
- FP32 headed recurrent state has shape `[B,H,D/H]`.
- `state_dict()` keys are native Fast-EsaQ keys (`qgv.weight`, `log_half_life`, etc.) with **no** `layer.` prefix.
- Use `backend="native"` for strict CUDA and `backend="pytorch"` for CPU reference.
- Original Fast-EsaQ `ESA` has no MLBricks-only `precision`, `compass`, `auto_move_input`, `esa_variant`, `strict_checks`, or `auto_compile` constructor kwargs. Use PyTorch `.to(dtype=...)` for weight precision.
- `ESAAdapter` at `mlbricks.esa.ESAAdapter` or `mlbricks.layer.ESAAdapter` preserves the MLBricks layer API for older integrations. `mlbricks.esa.layer.ESA` remains that legacy adapter explicitly.
- `ESAModel`, `VESA` and built-in `SOUP` continue using that adapter to retain model checkpoint structure and single-output contracts. This **does not** claim the full-model decode hot path has been migrated to native yet.
- `Bricks` accepts native and compatibility ESA mixers. It restores a singleton time dimension at the *Bricks boundary* as needed for its residual pipeline.
- Existing user code that relied on `output = ESA(...)(x)`, `.layer.*` checkpoint names, or 2D decode output must migrate to the native output/state contract or use `ESAAdapter`.
- No original Fast-EsaQ CUDA kernels, learned gates or equations are modified by this patch.

Validate `python -m pytest -q tests/test_esa_direct_native.py` followed by the project regression suite and GPU benchmarks. Do not merge `main` until full-model migration and GPU validation.
