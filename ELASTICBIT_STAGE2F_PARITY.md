# ElasticBit Stage-2F notebook parity

This patch intentionally ports the validated Stage-2F planner from
`notebook09b41bd228(6).ipynb` without replacing it with a generalized planner.

Reference cell SHA-256 values:

| Notebook cell | Purpose | SHA-256 |
| --- | --- | --- |
| 56 / Stage-2F cell 37 | candidate backend | `e030e966704daf292407c6916b89fed18f8e1688bffd84a4f85b7aeeaad8cc52` |
| 57 / Stage-2F cell 38 | isolated autotune | `aa57113b44217ceeb8db6aa4f5d5e3814e55259805252da70018671c2e5a8df7` |
| 58 / Stage-2F cell 39 | Granite math harness | `b1388fc640221eeaf9826d607137014d3a13bc62b9f431c43d950fe4d3662e33` |
| 59 / Stage-2F cell 40 | hierarchical full-layer autotune | `69a08ff01d43f33ceff4ea2bd9a529ed81764fa34f20cbe20ec391bf9fc9046d` |
| 60 / Stage-2F cell 41 | quality + final A/B guard | `167e5413d8963a9c988e52f12f308690ab62d5f4a1dc424a95047c3beda2c0a8` |

## Frozen algorithm values

The library port keeps the notebook values exactly:

- `M = [1, 16, 128, 512]`
- isolated iterations: `{1: 240, 16: 120, 128: 45, 512: 14}`
- isolated rounds: `5`
- group relative-L2 limit: `2e-2`
- top-K per group: `3`
- full-layer acceptance margin: `2%`
- full-layer iterations: `{1: 60, 16: 40, 128: 14, 512: 4}`
- full-layer rounds: `3`
- final iterations: `{1: 100, 16: 70, 128: 20, 512: 6}`
- final rounds: `5`
- full-layer relative-L2 limit: `2e-2`
- final deployment guard: `1%`

Candidate names are also frozen to the notebook set. No additional QKV, Gate/Up,
model, GPU, or sequence-bucket candidates are introduced by this patch.

## Deliberate library-only mechanics

Only the mechanics required to expose the notebook planner as
`ElasticBit.optimize(model, force=False)` are new:

1. explicit API entry point;
2. optional hardware-profile metadata persisted by `ElasticBit.save()`;
3. cache reuse on a matching T4/runtime fingerprint;
4. a Granite decoder-layer runtime wrapper that executes the exact notebook
   math for a final selected plan at the exact tested `M` values, while cached
   decode and unsupported call shapes fall back to the untouched original layer.

These mechanics do not alter the notebook candidate search or its numerical and
latency acceptance rules.

## Scope restriction

Stage-2F v1 intentionally accepts only the validated one-layer Granite 4.2 8B
shape on a validated Tesla T4.  Other GPUs, models, layer counts, and shape
buckets are rejected rather than silently generalized.
