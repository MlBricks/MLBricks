# MLBricks Kit 1.0.0b1 API Reference

`mlbricks-kit` is the PyPI distribution. The Python import namespace remains `mlbricks`.

```bash
pip install mlbricks-kit==1.0.0b1
```

```python
import mlbricks
print(mlbricks.__version__)  # 1.0.0b1
```

This reference is generated against the public package surface exported by `mlbricks.__all__` in the `1.0.0b1` release candidate. The supported backend policy is `auto | native | pytorch` unless a component documents a narrower behavior.

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
    ElasticBit, ElasticBitConfig,
    FFN, Embedding, LMHead, Linear, LayerNorm, RMSNorm, Residual,
    Trainer,
)
```

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
quantize(model, *, method="elasticbit", bits=4, include_embeddings=False, **kwargs)
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

### `ESA`

```python
ESA(
    embd,
    head=4,
    batch=None,
    block=None,
    backend="auto",
    precision="fp16",
    *,
    compass="auto",
    dropout=0.0,
    gate_min=0.8,
    gate_max=0.995,
    eps=1e-5,
    device="auto",
    auto_compile=False,
    compile_mode="default",
    auto_move_input=True,
    strict_checks=False,
)
```

`esa` is a lowercase constructor alias for `ESA`.

Main methods:

```python
layer(x) -> torch.Tensor
layer.set_backend(backend, *, recursive=True)
layer.resolved_backend() -> str
layer.prefill(x, state=None, *, backend=None, compass=None)
    -> (output, state)
layer.decode_step(x, state) -> (output, state)
layer.compile(mode="default")
```

`ESAConfig` is the standalone ESA layer configuration dataclass.

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

## Bolt attention

### `Bolt` / `BoltAttention`

`BoltAttention` is an alias of the canonical `Bolt` implementation.

```python
Bolt(
    d_model,
    num_heads,
    *,
    latent_dim=32,
    bias=False,
    dropout=0.0,
    causal=True,
    backend="auto",
    autotune_kernels=True,
    eps=1e-6,
    use_sdpa=True,
    position=None,
    native_full_sequence=False,
)
```

Main methods:

```python
bolt(x) -> torch.Tensor
bolt.set_backend(backend, *, recursive=True)
bolt.resolved_backend() -> str
bolt.prefill(x, *, start_pos=0) -> (output, cache)
bolt.decode_step(x, cache) -> (output, cache)
bolt.project_cache_state(x, *, start_pos=0)
bolt.prefill_with_cache(x, *, start_pos=0)
bolt.decode(x, c_cache, rho_cache, *, force_retune=False, position=None, used_length=None)
```

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

### Ready-made Bolt language model

`BoltModel` is the canonical ready-made Bolt LM name. `Gaussian` is retained as the compatibility name; `BoltConfig` is an alias of `GaussianConfig`.

```python
BoltModel(
    vocab_size=50257,
    context=2048,
    layers=6,
    dim=384,
    heads=6,
    latent_dim=32,
    position="rope",
    backend="auto",
)
```

The full configuration also supports FFN, residual, normalization, dropout, embedding tying, and FFNBrick/ResController tuning fields.

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

### PyTorch-compatible quantization surface

```python
ElasticBit(
    bits=4,
    group_size=128,
    *,
    scale_dtype=torch.float16,
    compute_dtype=None,
    cache_dequantized=True,
    runtime="auto",
    backend="auto",
)

ElasticBitConfig(
    bits=4,
    group_size=128,
    scale_dtype=torch.float16,
    compute_dtype=None,
    cache_dequantized=True,
    runtime="auto",
    backend="auto",
)
```

Main methods:

```python
packed = elastic.quantize(tensor)
tensor = elastic.dequantize(packed, device=None, dtype=torch.float32)
elastic_linear = elastic.linear(torch_linear)
elastic_embedding = elastic.embedding(torch_embedding)
elastic.quantize_module(model, include_embeddings=False, skip_names=())
```

Package-level helpers:

```python
quantize_tensor(tensor, config=None) -> PackedElasticBit
dequantize_tensor(packed, *, device=None, dtype=torch.float32) -> torch.Tensor
quantize_module(module, config=None, *, include_embeddings=False, skip_names=())
```

`ElasticLinear` and `ElasticEmbedding` are the packed module wrappers.

### Optional native 4–32-bit runtime

When the ElasticBit native CUDA extension is packaged and available:

```python
analysis = ElasticBit.bitsAnaliser(weights, calibration, threshold=0.01)
matrix = ElasticBit.RuntimeMatrix(weights, analysis["selected_bits"], "compact")
y = matrix.forward(x)
```

`ElasticBit.RuntimeMatrix`, `ElasticBit.NativeFP16Matrix`, and `ElasticBit.bitsAnaliser` require the native ElasticBit runtime; the portable wheel can still use the PyTorch-compatible ElasticBit surface above.

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

The following names are exported by `mlbricks.__all__` in `1.0.0b1`:

| Group | Exports |
| --- | --- |
| Attention / Bolt | `Attention`, `Bolt`, `BoltAttention`, `BoltModel`, `BoltConfig`, `Gaussian`, `GaussianConfig` |
| SOUP | `SOUP`, `soup` |
| Composition | `Bricks`, `Brick` |
| Vision | `VisionBolt`, `VisionBoltConfig`, `vision_native_available`, `vision_native_cuda_built`, `vesa`, `Vesa`, `VesaConfig` |
| Position | `RoPE`, `LearnedPosition`, `SinusoidalPosition` |
| Runtime / planner | `normalize_backend`, `set_module_backend`, `backend_report`, `ExecutionPlan`, `build_execution_plan`, `predict_module`, `prepare_module_execution`, `apply_execution_route`, `reset_execution_route`, `EXECUTION_PLANNER`, `MLBricksExecutionPlanner`, `AUTO_OPERATOR_DEFAULTS` |
| FFNBrick | `ffnbrick`, `MicroVirtualFFN`, `StateAwareFFN`, `VirtualStateAwareFFN`, `ffnbrick_native_backend_available`, `ffnbrick_native_backend_name` |
| ResidualBrick | `residualbrick`, `ResController`, `residualbrick_native_backend_available`, `residualbrick_native_backend_name` |
| ESA | `ESA`, `esa`, `ESAConfig`, `ESAModel`, `ESAModelConfig`, `GenerationResult`, `GenerationStats`, `compass`, `CompassResult`, `ThunderESA`, `thunderBoost` |
| Lifecycle / training | `Trainer`, `TrainerState`, `save`, `load`, `inspect`, `predict`, `generate`, `compile`, `quantize`, `train`, `Adam`, `AdamW`, `FP16_ADAM_MIN_EPS`, `stabilize_optimizer` |
| Basic blocks | `FFN`, `Embedding`, `LMHead`, `Linear`, `LayerNorm`, `RMSNorm`, `Residual`, `ffn`, `embedding`, `embeddings`, `lmhead`, `linear`, `layernorm`, `rmsnorm`, `residual` |
| ElasticBit | `ElasticBit`, `ElasticBitConfig`, `PackedElasticBit`, `ElasticLinear`, `ElasticEmbedding`, `quantize_tensor`, `dequantize_tensor`, `quantize_module` |
| ESA benchmark | `ESABenchmarkConfig`, `DEFAULT_BENCHMARK_CONFIG`, `FAST_BENCHMARK_CONFIG`, `PAPER_BENCHMARK_CONFIG`, `BENCHMARK_DEFAULTS`, `FAST_BENCHMARK_DEFAULTS`, `PAPER_BENCHMARK_DEFAULTS`, `TrainingIntervalTimer`, `cuda_telemetry` |

## Licensing

MLBricks Kit is distributed under the repository-level PolyForm Noncommercial License 1.0.0 terms plus component-specific notices shipped with the relevant components. Commercial use requires a separate written commercial license. See `LICENSE.md`, `LICENSING_NOTICE.md`, `COMMERCIAL_LICENSE.md`, and the component notices listed in `LICENSE.md`.
