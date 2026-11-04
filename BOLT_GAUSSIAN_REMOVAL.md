# BOLT-only public API migration

The retired `mlbricks/gaussian.py` file has been **deleted**, along with the
`Gaussian`, `GaussianConfig`, and `GaussCache` public import aliases.

Use these classes instead:

| Former Python API | New Python API |
|---|---|
| `Gaussian` | `BoltModel` |
| `GaussianConfig` | `BoltConfig` |
| `GaussCache` | `BoltLegacyCache` (internal legacy cache type) |

This is an intentionally **breaking public API change**. Historical state
dictionaries keep their tensor keys and may still be loaded into `BoltModel`
when you explicitly configure the matching architecture
(`attention_scale="head"`, `memory_init="legacy"`,
`projection="separate"` for the former compatibility math, and exact
widths, heads, FFN and other settings). Pickled full-model objects made with
`Gaussian` cannot be unpickled directly without a separate conversion tool;
prefer portable `state_dict` weights.

Old low-level native CUDA operator names beginning with `gauss_` remain
unchanged as internal ABI symbols, and do not affect the public Python API.
