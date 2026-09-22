# ElasticBit Stage-2G Notebook Parity

This patch ports `ElasticBit.reduce()` from the validated notebook:

`ElasticBit_Reduce_Benchmark_Stage2G_v2.ipynb`

## Source identity

Notebook code-cell SHA-256 values:

- core compiler cell: `61c4b679c03adcea1e1d0fac46431c96827c6dc651a7b6923340f2fc49c974cd`
- transactional rollback hotfix cell: `bb70cd67ede7e5914df7e771bad7fa5300058c9c33a4f3b933e8b9e1da5c8287`
- notebook-session public reduce cell: `b321f3c4ebe3da0eb9bf9fd72b6dd983464f3c24b3a60489767e4751085b74a6`

The production module `mlbricks/elasticbit/stage2g.py` was generated from those
notebook cells. Notebook-only top-level READY/diagnostic `print(...)` statements
were removed. The 34 top-level function/class definitions used by Stage-2G,
including the transactional wrappers and `_stage2g_public_reduce`, are
byte-for-byte identical to the notebook definitions.

## Frozen behavior

`ElasticBit.reduce(model, sample)` uses the notebook's
`UniversalStage2CoreCompiler` with exactly:

- `max_param_growth=0.0`
- `max_resultant_bytes=512 * 1024**2`
- `max_passes=32`

No package-specific rewrite rule was added.

The Universal Core still performs only the notebook rules:

- right-to-left affine reduction between barriers
- affine chain composition
- additive affine-region merging
- separate resultant propagation
- consumer safety
- parameter/materialization guards
- FP32 composition followed by cast-back
- per-rewrite numerical verification and rollback
- final whole-model numerical verification
- no QKV packing
- no Gate/Up packing
- no generic same-input packing
- no Transformer-specific rewrite
- no GPU-specific rewrite

## Transactional rollback

The validated notebook v2 hotfix is included unchanged: rewrite trials are made
on deep-copied FX graphs and only accepted trials replace the active graph.
Rejected trials leave the active graph object unchanged.

## Validation performed for this patch

Functional parity was checked against code executed directly from the notebook
cells for four proof graphs:

- additive branch merge
- affine chain between nonlinear boundaries
- pure same-input fan-out
- unpacked resultant fan-out

For every case:

- rewrite reports matched
- optimized FX graphs matched
- parameter counts matched
- package-vs-notebook output relative L2 was `0.0`

Repository tests after the patch:

`242 passed, 7 skipped`

The new Stage-2G parity test file contains 7 focused tests.

## Scope

This patch adds only the notebook-derived structural reducer and its public API:

```python
model = ElasticBit.reduce(model, sample)
```

It does not add Stage-2F runtime optimization and it does not modify the
ElasticBit compression algorithm.
