# BOLT 72e15cf0 mode integration

Patch baseline: Git branch `feature/bolt-v2-strict-native-decode`, commit
`72e15cf0`. Source export SHA-256:
`a5dcbf61eb1764a4c99fceca57cc09b1af19e2e592d0cdc28fbf3decdd61c6a3`.

## Mode meanings

| Mode | Attention | FP16 attention cache at D=384, H=12 |
|---|---|---|
| `quantised` (default) | Shared K/V MQA, Pre-AVQ384, learned head gates | 128 B/token/layer |
| `Keyless` | Original per-head BOLT32 shared-C, AVQ, learned head gates | 768 B/token/layer |
| `base` | Full MHA QKV, AVQ, learned head gates | 1536 B/token/layer |

AVQ is enabled by default; `avq="disable"` disables **only AVQ**. The
retention sigmoid is learnable, with its bias initialized to logit(0.95).
Learnable head-output gates start neutrally at `2 sigmoid(0)=1`.
Custom `avq_dim`/`state_dim` values remain supported; a state width
unequal to `dim` uses a learned state-return projection. The default
research-matched width is still `dim` in MQA and base MHA.
The keyless AVQ updates **after attention**; Enhanced MQA AVQ updates
**before attention**, following the research model. They are distinct.

Standalone `Bolt(dim, heads)`, full `BoltModel(BoltConfig(...))`, and
`layer_modes=(...)` hybrid models are supported. `state_memory=False`
remains a legacy alias for disabling AVQ. BOLT16 (`cache="tight"`,
`profile="compact"`) is disabled. Low-level `BoltPreset` retains
`mode="quality"` and `mode="fast"` for old BOLT32 weights.

## Native backend and checkpoints

Only Keyless retains native BOLT32 / V2 Strict cached-decode kernels;
Enhanced MQA and base MHA run through portable PyTorch SDPA. Requested
`backend="native"` for an architecture lacking a corresponding native
implementation raises a clear error instead of silently changing math.

Old Keyless/BOLT32 checkpoints may load with neutral-initialized new
head-gate parameters. Mode renaming does **not** convert checkpoint
architecture, and 0.95 initialization does not overwrite loaded weights.

## Verification limits

CPU parity, gradient flow, inference cache, and hybrid tests are in
`tests/test_bolt_modes_72e15cf0.py`. CUDA native parity and T4/H100
throughput are not established by these CPU tests. The 50M research
validation loss cannot be inferred from untrained layer parity.
