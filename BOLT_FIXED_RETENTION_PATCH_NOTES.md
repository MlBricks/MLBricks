# BOLT Fixed-Retention Update

## Scope

This update targets the Enhanced-MQA AVQ path used by the `quantised` and `base` BOLT modes. Other memory paths are left unchanged.

## Algorithm

The AVQ projection now produces only the write and read components:

\[
[V_t,R_t] = W_{VR}X_t
\]

For each configured query head, the implementation derives a fixed retention coefficient and maps it across the AVQ channels assigned to that head. The recurrent update is:

\[
S_t=A\odot S_{t-1}+0.1\tanh(V_t)
\]

The read gate, normalization, return projection, residual update, attention projections, and cache behavior are unchanged.

## Projection and checkpoint handling

- The AVQ projection output width changes from `3 * avq_dim` to `2 * avq_dim`.
- The unused retention-gate projection is removed completely.
- During model construction, the active write and read rows are copied from the original projection. This preserves their initialized values and does not advance the model's random-number stream.
- Loading a legacy three-branch AVQ checkpoint into the updated Enhanced-MQA path automatically keeps the write and read rows and drops the unused gate rows.
- `forget_gate_init` remains accepted by the existing API for configuration compatibility; it does not control fixed retention in the updated Enhanced-MQA path.

## Configuration and API

No public constructor or configuration signature changes. Retention rates use the configured query-head count. AVQ channels are distributed across those heads using the configured AVQ width, including widths that are not evenly divisible by the head count.

## Validation

The added tests cover dynamic head counts and AVQ widths, two-branch projection shape, exact active-weight preservation, recurrence/reference parity, incremental-step parity, and legacy checkpoint migration.
