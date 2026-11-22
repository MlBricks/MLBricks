# MLBricks Kit 1.0.0b3 API Reference

`mlbricks-kit` is the PyPI distribution. The Python import namespace remains `mlbricks`.

```bash
pip install mlbricks-kit==1.0.0b3
```

```python
import mlbricks
print(mlbricks.__version__)  # 1.0.0b3
```

This reference is generated against the public package surface exported by `mlbricks.__all__` in the `1.0.0b3` release candidate. The supported backend policy is `auto | native | pytorch` unless a component documents a narrower behavior.

## Quick import surface

```python
from mlbricks import (
    ESA, ESAModel, ESAModelConfig,
    Bolt, BoltAttention, Attention, BoltModel, BoltConfig,
    SOUP, soup,
    Vesa, VesaConfig, VisionBolt, VisionBoltConfig,
    Bricks, Brick,
    StateAwareFFN, VirtualStateAwareFFN, MicroVirtualFFN,
    ResController,
    ElasticBit, RuntimeMatrix, BitAnalysis, BackendInfo,
    FFN, Embedding, LMHead, Linear, LayerNorm, RMSNorm, Residual,
    Trainer,
)
```

## ESA architecture modes (FastESA v0.4.0)

The public `ESA` constructor selects **retention by default**. To keep the
historical Thunder implementation, pass `mode="legacy"`. The returned object is
the concrete engine (not an additional wrapper), so checkpoint parameter keys
are unaffected. The two architectures have **different** recurrence rules and
state dictionaries; never load a checkpoint from one mode into the other.

```python
import torch
from mlbricks import ESA, RetentionESA, LegacyESA

retention = ESA(embd=384, head=6, backend="pytorch", device="cpu")
# Same as ESA(embd=384, head=6, mode="retention", ...)
assert isinstance(retention, RetentionESA)
x = torch.randn(1, 8, 384)
y, state = retention(x)           # output tensor AND recurrent state
next_out, state = retention.decode_step(torch.randn(1, 384), state)
# Initial learnable FP32 half-lives: [0.125, 0.5, 2, 8, 32, 64]

legacy = ESA(embd=32, head=4, mode="legacy", precision="fp32", device="cpu")
assert isinstance(legacy, LegacyESA)
y_only = legacy(torch.randn(1, 8, 32))   # single output tensor
```

- `RetentionESA` (also `CurrentESA`) — FastESA half-life mixer, the default ESA mode
- `LegacyESA` — previous MLBricks Thunder ESA, selected with `mode="legacy"`
- `BaseESA` — FastESA's separate historical learned-bounds variant (not Thunder)
- `CurrentESAConfig`, `CurrentESAModel`, `BaseESAModel` — FastESA-only config/model APIs
- `DEFAULT_CURRENT_HALF_LIVES` — `(0.125, 0.5, 2, 8, 32, 64)`
- `make_esa` — compatible factory with `architecture="retention"` (default), `"current"` (alias), or `"legacy"`
- `make_current_esa_model`, `current_esa_forward_tokens` — retention language model helpers
- `save_current_esa_checkpoint` / `load_current_esa_checkpoint` — retention mixer checkpoint tools
- `load_current_esa_model_checkpoint` — retention LM checkpoint loader
- `CurrentESAGraphDecoder` — CUDA graph helper (requires GPU validation)

`ESAModel` / `ESAModelConfig`, legacy `ESAConfig`, and existing SOUP/VESA models
remain on the original Thunder ESA engine for checkpoint compatibility. Their
migration to retention is a separate change. Native FastESA CUDA performance
and correctness need dedicated GPU testing.

## Backend policy and execution planner

### Backend values

- `auto` — use the MLBricks planner. Backend-aware elements qualify available routes independently; a composed model can therefore mix native and PyTorch routes.
- `native` — require the component's supported native implementation.
- `pytorch` — force the PyTorch/reference implementation.

### Package-level backend helpers

```python
normalize_backend(value="auto", *, warn_legacy=False) -> str
set_module_backend(module, backend, *, recursive=True)
backend_report(module) -> list[dict[str, str]]
build_execution_plan(module) -> ExecutionPlan
prepare_module_execution(
    module, *sample_args,
    sample_kwargs=None,
    warmup=5,
    trials=20,
    candidates=("operator", "native", "pytorch"),
    force=False,
) -> ExecutionPlan
predict_module(
    module, *args,
    device="auto",
    dtype="auto",
    calibrate=True,
    calibration_warmup=1,
    calibration_trials=3,
    candidates=("operator", "native", "pytorch"),
    **kwargs,
)
apply_execution_route(module, route)
reset_execution_route(module)
```

`EXECUTION_PLANNER` is the shared `MLBricksExecutionPlanner` instance. `AUTO_OPERATOR_DEFAULTS` contains the planner's built-in default route hints. `ExecutionPlan` is the model-level diagnostic summary returned by the execution helpers.

Example:

```python
from mlbricks import Bolt, backend_report, build_execution_plan

layer = Bolt(384, 6, backend="auto")
print(backend_report(layer))
print(build_execution_plan(layer))
```

## Unified lifecycle API

Lifecycle operations are package-level and architecture-agnostic.

```python
save(model, path, *, metadata=None) -> pathlib.Path
load(path, *, device="auto", strict=True) -> torch.nn.Module
inspect(model_or_path) -> dict
predict(model, *args, **kwargs)
generate(model, *args, **kwargs)
compile(model, *, mode="default", dynamic=None, fullgraph=False, strict=False)
```

Example:

```python
import mlbricks as mlb

mlb.save(model, "model")
model = mlb.load("model", device="auto")
metadata = mlb.inspect("model")
y = mlb.predict(model, inputs)
```

`ESAModel.save()`, `ESAModel.load()`, `mlbricks.esa.Trainer`, and `mlbricks.esa.trainer` are not public lifecycle APIs in this release.

## Training API

### `Trainer`

```python
Trainer(
    model,
    *,
    optimizer="adamw",
    lr=3e-4,
    optimizer_kwargs=None,
    scheduler=None,
    scaler=None,
    loss_fn=None,
    checkpoint_dir="checkpoints",
    save_every=None,
    save_at=None,
    save_best=True,
    save_last=True,
    keep_last_n=3,
    grad_clip=None,
)
```

Main methods:

```python
trainer.fit(
    train_loader,
    *,
    steps=None,
    epochs=None,
    val_loader=None,
    validate_every=None,
    log_every=None,
    callback=None,
) -> dict

trainer.evaluate(data_loader) -> dict[str, float]
trainer.save(name="last", *, extra=None) -> pathlib.Path
trainer.save_checkpoint(*, step=None, name=None, protected=False, extra=None)
trainer.load_checkpoint(path, *, device=None, restore_rng=True)
trainer.resume_from(value, *, device=None)
Trainer.resume(
    path,
    *,
    device="auto",
    loss_fn=None,
    scheduler=None,
    scaler=None,
    restore_rng=True,
)
```

One-call training:

```python
train(
    model,
    train_loader,
    *,
    steps=None,
    epochs=None,
    val_loader=None,
    validate_every=None,
    **trainer_kwargs,
) -> Trainer
```

For `(inputs, targets)` batches, `Trainer` passes targets to the model. A model may return a scalar loss, `(logits, loss)`, a mapping containing `loss`, or an object exposing `.loss`. Use `loss_fn=` when the model returns predictions only.

### Optimizers

MLBricks exports `Adam` and `AdamW` wrappers plus:

```python
stabilize_optimizer(optimizer, *, min_fp16_eps=1e-5, warn=True) -> bool
FP16_ADAM_MIN_EPS
```

## ESA

### `ESA` (retention default)

```python
ESA(embd=384, head=None, backend="auto", device=None, *, mode="retention", **kwargs)
```

The default number of heads is 6 for retention and 4 for legacy. Retention
accepts FastESA-specific keywords such as `chunk_size`; legacy accepts
`precision`, `compass`, `gate_min`, `gate_max`, etc. Each mode validates its
own arguments. `mode` accepts only `"retention"` and `"legacy"`.

```python
retention(x, state=None) -> (output, new_state)
retention.prefill(x, state=None) -> (output, new_state)
retention.decode_step(x, state) -> (output, new_state)

legacy(x) -> torch.Tensor
legacy.forward_with_state(x, state=None) -> (output, new_state)
legacy.prefill(x, state=None) -> (output, new_state)
legacy.decode_step(x, state) -> (output, new_state)
```

`esa` is the lowercase alias for the public `ESA` selector. `LegacyESA`
preserves the original layer class for existing applications and checkpoints.
`ESAConfig` is the legacy layer configuration dataclass.

### `ESAModel`

```python
ESAModel(config=None, *, device="cuda", **kwargs)
```

`ESAModelConfig` fields include `vocab_size`, `block`, `n_layer`, `head`, `embd`, backend/precision settings, FFN/residual choices, training compile settings, and embedding tying.

Core model methods:

```python
model(input_ids, targets=None) -> (logits, loss_or_none)
model.set_backend(backend, *, recursive=True)
model.backend_report()
model.execution_plan()
model.prepare_execution(...)
model.predict(...)
model.reset_execution()
model.compile(*, mode=None, fullgraph=None, strict=False)
model.compile_training(*, mode=None, fullgraph=None)
model.prepare_generation(*, compile_decode=True, mode="default", fullgraph=False)
model.prefill(
    input_ids,
    *,
    engine="thunder",
    compile_mode="default",
    fullgraph=False,
    dynamic=True,
)
model.compile_generation(*, mode="default", fullgraph=False)
model.model_info() -> dict
```

Generation:

```python
model.generate(
    prompt=None,
    *,
    tokenizer=None,
    input_ids=None,
    seek=128,
    prefill="thunder_16",
    runtime="lightning",
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
    seed=None,
    compile=True,
    compile_mode="default",
    progress_interval=None,
    stats=False,
    max_new_tokens=None,
)
```

When `stats=True`, generation can return `GenerationResult`; statistics are represented by `GenerationStats`.

### ESA Compass / benchmark helpers

```python
compass(
    *,
    evaluate_fn,
    c_candidates=(8, 16, 32, 64),
    precision="fp16",
    quality_tolerance=0.02,
    metric="ppl",
    speed_metric="tok_per_sec",
    **evaluate_kwargs,
) -> CompassResult
```

`ThunderESA`, `thunderBoost`, `ESABenchmarkConfig`, `TrainingIntervalTimer`, `cuda_telemetry`, and the benchmark preset constants are also exported from the package root.

## BOLT Architecture

### Architecture Modes

BOLT is a memory-efficient attention architecture with optional pre-attention AVQ, dynamic head dimensions, learnable head-output gates, and externally supplied positional transforms.

| Architecture | `mode` | Configuration |
|---|---|---|
| BOLT Quantised | `quantised` | One shared KV head (`kv_heads=1`, the default) |
| BOLT Grouped Quantised | `quantised` | Multiple shared KV heads (`1 < kv_heads < num_heads`) |
| BOLT Keyless | `keyless` | One shared C representation; `kv_heads` is not applicable |
| BOLT Base | `base` | Full per-head key and value representations |

Quantised and Grouped Quantised use the same implementation and differ only in `kv_heads`. Setting `kv_heads=num_heads` in `quantised` uses full KV width. Every mode supports optional AVQ with `retention="decay"` (default) or `retention="learned"`.

### BOLT API

The embedding width must be divisible by the query-head count. For Quantised and Grouped Quantised, `kv_heads` must be a positive divisor of the query-head count. The default AVQ width is the embedding width. FP32 is the default precision; mixed precision is opt-in.

```python
from mlbricks import Bolt, BoltConfig, BoltModel, RoPE

# BOLT Quantised
quantised = Bolt(384, 12, mode="quantised", kv_heads=1, position=RoPE())

# BOLT Grouped Quantised
grouped_quantised = Bolt(384, 12, mode="quantised", kv_heads=3)

# BOLT Keyless
keyless = Bolt(384, 12, mode="keyless")

# BOLT Base
base = Bolt(384, 12, mode="base")

# BOLT Base without AVQ or head-output gates
classic = Bolt(384, 12, mode="base", avq=False, head_output_gate=False)

# BOLT with learned AVQ retention
learned = Bolt(384, 12, retention="learned")

# Hybrid BOLT language model
config = BoltConfig(
    vocab_size=50257,
    context=2048,
    layers=3,
    dim=384,
    heads=12,
    layer_modes=("base", "quantised", "keyless"),
    device="auto",
)
model = BoltModel(config)
model.set_qk_transform(RoPE())
```

### Memory Efficiency

At 384D and 12 query heads (32 dimensions per head), persistent FP16 attention-cache storage is:

| BOLT architecture | Bytes/token/layer | Reduction vs Base |
|---|---:|---:|
| Quantised | 128 | 91.67% |
| Grouped Quantised (3 KV heads) | 384 | 75.00% |
| Keyless | 64 | 95.83% |
| Base | 1,536 | 0% |

Enabled 384D AVQ adds 1,536 bytes of FP32 recurrent state per sequence per layer, independent of context length. The table excludes AVQ state, model weights, temporary allocations and unused preallocated capacity. Persistent cache storage is distinct from peak GPU allocation.

### Execution and Optimization

BOLT uses PyTorch Scaled Dot-Product Attention (SDPA), with compact single-KV-head decoding and chunked AVQ inference prefill. Chunking carries recurrent state across chunks and does not alter training. FP16 and BF16 mixed-precision inference require suitable hardware. Modern native attention and CUDA graph acceleration are not implemented; optional standalone native operators are separate from the BOLT attention path.

### `Bolt` / `BoltAttention`

`BoltAttention` aliases `Bolt`. Use `mode="quantised"`, `"keyless"`, or `"base"`. Positional transforms are external callables, such as `RoPE()`, rather than strings. `state_memory=False` or `avq=False` disables AVQ; `head_output_gate=False` disables learnable head-output gates.

```python
bolt(x)  # differentiable forward
output, cache = bolt.prefill_with_cache(x)
next_output, cache = bolt.decode_step(next_x, cache)
fixed = bolt.allocate_cache(batch_size=1, capacity=2048)
y = bolt.decode_with_cache(next_x, fixed, position=0)
```

The fixed-cache example starts a separate empty session. Quantised, Grouped Quantised and Base use `(k,v,state)` dynamic caches and `EnhancedMQACache` static caches. Keyless uses `(c,state)` dynamic caches and `SharedCCache` static caches. The actual cache precision follows the model's inference precision. Cache allocation and cached decode are for inference; use ordinary forward for training.

### `Attention`

`Attention` is the standard causal attention implementation under the same backend policy.

```python
Attention(
    d_model,
    num_heads,
    *,
    bias=False,
    dropout=0.0,
    causal=True,
    backend="auto",
    autotune_kernels=True,
    position=None,
)
```

### BOLT model and components

`BoltModel` and `BoltConfig` provide a language model; `BoltStateMemory` implements pre-AVQ. `EnhancedMQA`, `EnhancedMQACache`, `ClassicMHA`, `SharedCKeyless`, and `SharedCCache` are also exported. `BoltModel` supports forward, prefill, incremental decoding and generation. Modern mode weights retain their existing parameter and state-dict layout. Historical BOLT16/BOLT32 checkpoints and fixed-width graph caches are not supported. Historical branch and patch notes are development records, not current API specifications. Existing naming aliases such as `fast` select modern modes, not a second implementation. Modern graph acceleration is not implemented; graph policy does not select the retired kernels.

## SOUP

```python
SOUP(
    dim=512,
    width=1116,
    depth=2,
    mixer="esa",
    ffn="saffn",
    mixer_config=None,
    ffn_config=None,
    backend="auto",
    precision="fp16",
    memory_dim=128,
    fusion_hidden=768,
)
```

`width`, `mixer`, `ffn`, and their configuration values may be specified per layer where supported by the SOUP configuration helpers. Built-in mixer names include `esa` and `bolt`; built-in FFN names include `saffn` and `ffn`.

Main methods:

```python
model(x) -> torch.Tensor
model.set_backend(backend)
model.element_backends()
model.resolved_backend()
model.validate(x) -> dict
model.prepare_generation(fast=True)
model.clear_generation_plan()
model.prefill(x) -> (output, cache)
model.decode_step(x, cache) -> (output, cache)
model.parameter_count  # property
```

For recurrent generation, call `model.eval()` before `prepare_generation()`.

`soup(...)` is the lowercase constructor helper.

## Bricks composition API

### `Brick`

```python
Brick(
    *,
    mixer,
    ffn=None,
    norm1="rmsnorm",
    norm2="rmsnorm",
    residual=None,
    residual2=None,
    position=None,
    pre=None,
    post=None,
    dim=None,
)
```

### `Bricks`

```python
Bricks(
    *,
    vocab_size,
    dim,
    context,
    layers,
    embedding="standard",
    position=None,
    final_norm="rmsnorm",
    lm_head="tied",
    dropout=0.0,
    backend="auto",
)
```

`Bricks` supports the common model execution methods: `set_backend`, `backend_report`, `execution_plan`, `prepare_execution`, `predict`, `reset_execution`, `compile`, `prepare_generation`, `prefill`, `decode_step`, and `generate`.

When composing an ESA layer inside a parent module, use `ESA(..., device=None)` and place the complete parent with `.to(device)`.

## FFNBrick

```python
StateAwareFFN(
    d_model,
    state_dim=256,
    depth_embedding_dim=64,
    layer_index=0,
    total_layers=1,
    use_native=None,
    fused_cuda=True,
    backend="auto",
)

VirtualStateAwareFFN(
    d_model,
    state_dim=256,
    depth_embedding_dim=64,
    layer_index=0,
    total_layers=1,
    virtual_refinements=2,
    virtual_hidden_dim=128,
    use_native=None,
    fused_cuda=True,
    backend="auto",
)

MicroVirtualFFN(
    d_model,
    hidden_dim=64,
    refinements=1,
    use_native=None,
    fused_cuda=True,
    backend="auto",
)
```

State-aware forward contract:

```python
update, next_state = ffn(x, esa_update, previous_esa, previous_state)
```

Native capability helpers:

```python
ffnbrick_native_backend_available() -> bool
ffnbrick_native_backend_name() -> str
```

The `ffnbrick` submodule is also exported.

## ResidualBrick

```python
ResController(
    update_ratio,
    stream_ratio=1.08,
    update_softness=8.0,
    stream_softness=8.0,
    eps=1e-12,
    *,
    use_native=None,
    fused_cuda=True,
    backend="auto",
)
```

```python
out = controller(residual, update)
controller.set_backend("auto")
controller.resolved_backend()
```

Capability helpers:

```python
residualbrick_native_backend_available() -> bool
residualbrick_native_backend_name() -> str
```

The `residualbrick` submodule is also exported.

## PyTorch-native building blocks

These components preserve ordinary PyTorch autograd, serialization, device, and dtype behavior.

```python
Linear(in_features, out_features, bias=True, device=None, dtype=None)
Embedding(vocab_size, embedding_dim, **torch_embedding_kwargs)
LMHead(hidden_size, vocab_size, *, bias=False, tie_to=None, device=None, dtype=None)
FFN(
    hidden_size,
    intermediate_size=None,
    *,
    activation="gelu",
    dropout=0.0,
    bias=True,
    gated=False,
    device=None,
    dtype=None,
)
LayerNorm(
    normalized_shape,
    *,
    eps=1e-5,
    elementwise_affine=True,
    bias=True,
    device=None,
    dtype=None,
)
RMSNorm(
    normalized_shape,
    *,
    eps=1e-6,
    elementwise_affine=True,
    device=None,
    dtype=None,
)
Residual(dropout=0.0)
```

Lowercase aliases are exported: `ffn`, `embedding`, `embeddings`, `lmhead`, `linear`, `layernorm`, `rmsnorm`, `residual`.

## Position encodings

```python
RoPE(dim=None, *, base=10000.0)
LearnedPosition(dim, max_seq_len)
SinusoidalPosition(dim, max_seq_len=65536, base=10000.0)
```

All support `forward(x, *, start_pos=0)`.

## VESA

```python
Vesa(config=None, **kwargs)
VesaConfig(..., backend="auto")
```

Common configuration fields:

```text
image_size=32
patch_size=4
in_channels=3
num_classes=10
dim=192
depth=6
backend="auto"
engine="Serpentine"
position="auto"
scan="cross"
heads=6
latent_dim=32
```

Supported high-level engine names include `Serpentine`, `ViT` / `VisionTransformer`, `CNN`, `Diffusion`, and `AR`.

Model helpers include `set_backend`, `backend_report`, `execution_plan`, `prepare_execution`, `predict`, and `reset_execution`. `generate()` is available for `engine="AR"`; diffusion exposes `benchmark_sample_loop()`.

Advanced VESA families remain available through the exported `mlbricks.vesa` module.

## VisionBolt / VisualBolt

VisualBolt is exposed through `VisionBolt`.

```python
VisionBolt(config=None, **kwargs)
VisionBoltConfig(..., backend="auto")
```

`VisionBoltConfig` shares the VESA high-level vision configuration fields and adds/uses Bolt-specific `heads` and `latent_dim` values. It supports the same high-level engine family names.

Capability helpers:

```python
vision_native_available() -> bool
vision_native_cuda_built() -> bool
```

## ElasticBit

ElasticBit is threshold-driven and adaptive. There is no fixed-bit public quantization API.

### Whole-model compression

```python
model = ElasticBit.compress(
    model,
    calibrationData=calibrationData,
    threshold=0.01,
    decodePolicy="hardwareNative",
)
```

`decodePolicy` accepts only `"hardwareNative"` or `"fullPrecision"`. Both keep FP16 activations. Prefill always uses native framework/vendor GEMM after on-device FP16 materialization. `hardwareNative` preserves the validated T4 execution map and auto-tunes legal W4A16/W8A16/W16A16 candidates per matrix shape on other CUDA GPUs.

```python
model.elasticbit.summary()
model.elasticbit.setDecodePolicy("fullPrecision")
model.elasticbit.setDecodePolicy("hardwareNative")
```

### Explicit Stage-2F hardware optimization

```python
model = ElasticBit.optimize(model)
# Re-run the benchmark even when this hardware profile is cached:
model = ElasticBit.optimize(model, force=True)
```

`ElasticBit.optimize()` is explicit: normal inference does not benchmark candidate paths.
This first Stage-2F library port is deliberately frozen to the validated one-layer
Granite 4.2 8B / Tesla T4 notebook. It reuses a matching saved hardware profile
and otherwise executes the notebook candidate search once. Unsupported models,
shapes, or GPUs are rejected rather than silently generalized. Saving the model
after optimization persists the profile metadata in the `.elasticbit` artifact.

### Analysis and matrix API

```python
analysis = ElasticBit.analyze(weights, calibrationData, threshold=0.01)
analysis.selectedBits
analysis.selectedError
analysis.candidates

matrix = ElasticBit.compressMatrix(weights, calibrationData, threshold=0.01)
matrix.storageBits
matrix.executionWidth
matrix.executionPlanner
matrix.decodePolicy
matrix.activateDType
matrix.memoryReduction
matrix.forward(x)
matrix.setDecodePolicy("fullPrecision")
matrix.save("projection.mlb")
loaded = ElasticBit.loadMatrix("projection.mlb")
```

### Model artifacts

```python
ElasticBit.save(model, "model.elasticbit")
model = ElasticBit.load(modelArchitecture, "model.elasticbit")
```

`ElasticBit.load()` receives a model architecture instance because generic PyTorch module classes cannot be safely reconstructed from a weight artifact alone.

## Benchmark exports

The following ESA benchmark helpers are exported:

- `ESABenchmarkConfig`
- `DEFAULT_BENCHMARK_CONFIG`
- `FAST_BENCHMARK_CONFIG`
- `PAPER_BENCHMARK_CONFIG`
- `BENCHMARK_DEFAULTS`
- `FAST_BENCHMARK_DEFAULTS`
- `PAPER_BENCHMARK_DEFAULTS`
- `TrainingIntervalTimer`
- `cuda_telemetry`

## Complete package-root export index

The following names are exported by `mlbricks.__all__` in `1.0.0b3`:

| Group | Exports |
| --- | --- |
| Attention / Bolt | `Attention`, `Bolt`, `BoltAttention`, `BoltModel`, `BoltConfig`, `BoltStateMemory`, `EnhancedMQA`, `EnhancedMQACache`, `ClassicMHA`, `SharedCKeyless`, `SharedCCache` |
| SOUP | `SOUP`, `soup` |
| Composition | `Bricks`, `Brick` |
| Vision | `VisionBolt`, `VisionBoltConfig`, `vision_native_available`, `vision_native_cuda_built`, `vesa`, `Vesa`, `VesaConfig` |
| Position | `RoPE`, `LearnedPosition`, `SinusoidalPosition` |
| Runtime / planner | `normalize_backend`, `set_module_backend`, `backend_report`, `ExecutionPlan`, `build_execution_plan`, `predict_module`, `prepare_module_execution`, `apply_execution_route`, `reset_execution_route`, `EXECUTION_PLANNER`, `MLBricksExecutionPlanner`, `AUTO_OPERATOR_DEFAULTS` |
| FFNBrick | `ffnbrick`, `MicroVirtualFFN`, `StateAwareFFN`, `VirtualStateAwareFFN`, `ffnbrick_native_backend_available`, `ffnbrick_native_backend_name` |
| ResidualBrick | `residualbrick`, `ResController`, `residualbrick_native_backend_available`, `residualbrick_native_backend_name` |
| ESA | `ESA`, `esa`, `ESAConfig`, `ESAModel`, `ESAModelConfig`, `GenerationResult`, `GenerationStats`, `compass`, `CompassResult`, `ThunderESA`, `thunderBoost` |
| Lifecycle / training | `Trainer`, `TrainerState`, `save`, `load`, `inspect`, `predict`, `generate`, `compile`, `train`, `Adam`, `AdamW`, `FP16_ADAM_MIN_EPS`, `stabilize_optimizer` |
| Basic blocks | `FFN`, `Embedding`, `LMHead`, `Linear`, `LayerNorm`, `RMSNorm`, `Residual`, `ffn`, `embedding`, `embeddings`, `lmhead`, `linear`, `layernorm`, `rmsnorm`, `residual` |
| ElasticBit | `ElasticBit`, `RuntimeMatrix`, `BitAnalysis`, `BitCandidate`, `BackendInfo` |
| ESA benchmark | `ESABenchmarkConfig`, `DEFAULT_BENCHMARK_CONFIG`, `FAST_BENCHMARK_CONFIG`, `PAPER_BENCHMARK_CONFIG`, `BENCHMARK_DEFAULTS`, `FAST_BENCHMARK_DEFAULTS`, `PAPER_BENCHMARK_DEFAULTS`, `TrainingIntervalTimer`, `cuda_telemetry` |

## Licensing

MLBricks Kit is distributed under the repository-level PolyForm Noncommercial License 1.0.0 terms plus component-specific notices shipped with the relevant components. Commercial use requires a separate written commercial license. See `LICENSE.md`, `LICENSING_NOTICE.md`, `COMMERCIAL_LICENSE.md`, and the component notices listed in `LICENSE.md`.


## ESA mode selector (default retention)

`ESA(embd=384, head=6, backend="auto", mode="retention")` returns a concrete
`RetentionESA` with learnable FP32 half-lives initialized to
`[0.125, 0.5, 2, 8, 32, 64]`. The mode argument is optional.
Its forward and prefill calls return `(output, state)`.

`ESA(embd=384, head=4, mode="legacy")` returns the original `LegacyESA`
Thunder-based mixer; `forward(x)` returns only an output tensor.
Both instances keep native state dict parameter paths (no wrapper prefix).
`CurrentESA` remains as a compatible alias to `RetentionESA`.
Older `ESAModel` and VESA remain wired to the legacy engine and require
migration before they can use retention. Keep architecture-specific checkpoints
separate; do not swap the weight files between modes.
