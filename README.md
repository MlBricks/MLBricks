# MLBricks Kit 1.0.0b1

**MLBricks** is a modular machine-learning library for building efficient language and vision models from reusable components, with PyTorch reference paths and optional native C++/CUDA acceleration.

## Core components

- **ESA** — Entangled State Attention for recurrent/state-based sequence processing.
- **Bolt / BoltAttention** — optimized causal attention component.
- **BoltModel / BoltConfig** — ready-made Bolt language-model interface.
- **FFNBricks** — `StateAwareFFN`, `VirtualStateAwareFFN`, and `MicroVirtualFFN`.
- **ResController** — adaptive residual control.
- **SOUP** — state-oriented sequence processing with layerwise ESA/Bolt mixer selection.
- **ElasticBit** — adaptive 4–32-bit CUDA runtime plus PyTorch-compatible quantization helpers.
- **VESA** — ESA-based vision models.
- **VisualBolt** — Bolt-based vision models through `VisionBolt`.
- **Bricks / Brick** — heterogeneous model construction from MLBricks components.
- **Execution planner** — automatic backend and execution-route planning.
- **Native acceleration** — optional C++/CUDA kernels where supported.

The PyPI distribution is `mlbricks-kit`; the public Python import package remains `mlbricks`.

## Version

```python
import mlbricks
print(mlbricks.__version__)
# 1.0.0b1
```

## Installation

> Install `mlbricks-kit` from PyPI; use `import mlbricks` in Python.

From PyPI:

```bash
pip install mlbricks-kit
```

For development or installation directly from the repository root:

```bash
pip install -e .
```

For a CPU-only native build:

```bash
MLBRICKS_FORCE_CPU=1 pip install -e .
```




## Quick start

```python
import torch
from mlbricks import ESA, Bolt, StateAwareFFN, ResController

x = torch.randn(2, 128, 384)

esa = ESA(embd=384, head=6)
bolt = Bolt(d_model=384, num_heads=6, latent_dim=32)
ffn = StateAwareFFN(d_model=384)
residual = ResController(update_ratio=0.18)
```

MLBricks components use `backend="auto"` by default. The public backend choices are:

- `auto` — qualify **each element independently** (for example ESA, Bolt, SAFFN, a vision scan, or ElasticLinear). PyTorch is used as the one-time correctness reference; native must match it before both routes are benchmarked. The fastest valid route is then frozen for that element. Composite models can mix routes.
- `native` — require a supported MLBricks native implementation.
- `pytorch` — force the PyTorch/reference path.

Example:

```python
bolt.set_backend("native")
bolt.set_backend("pytorch")
bolt.set_backend("auto")
```

## Unified model lifecycle

Saving, loading, training, resume, inference, compilation, and quantization are package-level MLBricks APIs. They are not owned by ESA or any other individual architecture.

```python
import mlbricks as mlb

mlb.save(model, "my_model")
model = mlb.load("my_model", device="auto")
info = mlb.inspect("my_model")
```

The same calls work for ESA models, Bolt models, SOUP, VESA, `Bricks`, and mixed models built from multiple MLBricks components.

Train with the generic trainer:

```python
trainer = mlb.Trainer(
    model,
    optimizer="adamw",
    lr=3e-4,
    checkpoint_dir="checkpoints",
    save_every=1000,
)

trainer.fit(
    train_loader,
    steps=10_000,
    val_loader=val_loader,
    validate_every=500,
)
```

Or use the one-call convenience API:

```python
trainer = mlb.train(
    model,
    train_loader,
    steps=10_000,
    optimizer="adamw",
    lr=3e-4,
)
```

Resume an interrupted run with model weights, optimizer state, training step, scaler state, scheduler state (when supplied), and RNG state:

```python
trainer = mlb.Trainer.resume("checkpoints/last", device="auto")
trainer.fit(train_loader, steps=20_000)
```

Unified inference and optimization helpers are also available:

```python
y = mlb.predict(model, x)
text_or_tokens = mlb.generate(model, prompt, max_new_tokens=128)
model = mlb.compile(model)
model = mlb.quantize(model, method="elasticbit", bits=4)
```

`ESAModel.save()`, `ESAModel.load()`, and `mlbricks.esa.Trainer` are no longer public APIs.

## ESA

```python
from mlbricks import ESA

esa = ESA(
    embd=384,
    head=6,
)
```

For the ready-made ESA language-model architecture:

```python
from mlbricks import ESAModel, ESAModelConfig
```

## Bolt

`Bolt` and `BoltAttention` are the public attention names in MLBricks Kit 1.0.0b1.

CUDA FP16 Bolt now uses a compound Stage-1 execution path when the native extension is available: one packed Q/U/G GEMM followed by one fused gate/RMS postprocess emits only `Q`, `C`, and FP32 `rho`. Training keeps the same parameters/equations and uses normalized-key PyTorch SDPA; `use_sdpa=False` remains the explicit reference route.

```python
from mlbricks import Bolt, BoltAttention

bolt = Bolt(
    d_model=384,
    num_heads=6,
    latent_dim=32,
)
```

You can also import directly from the component package:

```python
from mlbricks.bolt import Bolt, BoltAttention
```

### Bolt language model

```python
from mlbricks import BoltModel

model = BoltModel(
    vocab_size=50_257,
    context=2048,
    layers=12,
    dim=768,
    heads=12,
    latent_dim=32,
    position="rope",
    ffn="saffn",
    residual="rescontroller",
    norm="rmsnorm",
)
```

## FFNBricks

```python
from mlbricks import StateAwareFFN, VirtualStateAwareFFN, MicroVirtualFFN

ffn = StateAwareFFN(
    d_model=384,
    state_dim=128,
    depth_embedding_dim=32,
    layer_index=0,
    total_layers=6,
)
```

## ResController

```python
from mlbricks import ResController

controller = ResController(update_ratio=0.18)
```

## SOUP

```python
from mlbricks import SOUP

model = SOUP(
    dim=512,
    width=[1116, 1116],
    depth=2,
    mixer=["esa", "bolt"],
    ffn=["saffn", "saffn"],
    backend="auto",
)
```

SOUP keeps `backend="auto"` **element-wise**. It does not force one backend for
the whole SOUP model: an ESA layer can freeze to native while a Bolt layer
freezes to PyTorch, and backend-aware FFN/residual elements make their own
one-time decisions. Each native candidate must first match its PyTorch
reference output; only parity-qualified routes are benchmarked for speed.
Per-layer mixer and FFN choices remain supported.

## ElasticBit

```python
from mlbricks import (
    ElasticBit,
    ElasticBitConfig,
    ElasticLinear,
    ElasticEmbedding,
    quantize_tensor,
    dequantize_tensor,
)
```

ElasticBit now also exposes the standalone 0.2 adaptive 4–32-bit CUDA API through the MLBricks namespace:

```python
from mlbricks import ElasticBit

analysis = ElasticBit.bitsAnaliser(weights, calibration, threshold=0.01)
matrix = ElasticBit.RuntimeMatrix(weights, analysis["selected_bits"], "compact")
y = matrix.forward(x)
```

The existing `ElasticLinear`, tensor quantization, and module-conversion helpers remain available as a PyTorch-compatible fallback surface.

## VESA

```python
from mlbricks import Vesa

model = Vesa(
    image_size=224,
    patch_size=16,
    dim=384,
    depth=12,
    engine="Serpentine",
    scan="cross",
    position=None,
    backend="auto",
)
```

Supported vision engine names include:

- `Serpentine`
- `ViT` / `VisionTransformer`
- `CNN`
- `Diffusion`
- `AR`

## VisualBolt

VisualBolt is exposed through the `VisionBolt` API:

```python
from mlbricks import VisionBolt

model = VisionBolt(
    image_size=224,
    patch_size=16,
    dim=384,
    depth=12,
    heads=6,
    latent_dim=32,
    engine="Serpentine",
    scan="cross",
    position=None,
    backend="auto",
)
```

VESA and VisualBolt share the same high-level vision-engine choices while keeping their respective ESA and Bolt mixers.

## Build a custom model with Bricks

```python
import torch
from mlbricks import (
    Bricks,
    Brick,
    ESA,
    Bolt,
    Attention,
    StateAwareFFN,
    FFN,
    ResController,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Bricks(
    vocab_size=50_257,
    dim=384,
    context=2048,
    position="sinusoidal",
    layers=[
        Brick(
            # In a composed model, let Bricks own placement and move the parent once.
            mixer=ESA(embd=384, head=6, device=None),
            ffn=StateAwareFFN(
                d_model=384,
                state_dim=128,
                depth_embedding_dim=32,
                layer_index=0,
                total_layers=2,
            ),
            residual=ResController(update_ratio=0.18),
            dim=384,
        ),
        Brick(
            mixer=Bolt(384, 6, latent_dim=32),
            ffn=FFN(384, 1536, activation="gelu"),
            dim=384,
        ),
    ],
).to(device)

# Place inputs on the same device as the parent model.
input_ids = torch.randint(0, 50_257, (1, 32), device=device)
logits = model(input_ids)
```

`ESA(device="auto")` is convenient when ESA is used by itself. Inside `Bricks` or another parent `nn.Module`, prefer `ESA(..., device=None)` and move the complete parent model with `.to(device)` so embeddings, residual paths, mixers, FFNs, and inputs remain on one device.

## Native vision runtime

VESA and VisualBolt can use the shared MLBricks vision runtime when the compiled backend is available. The public backend contract remains `auto | native | pytorch`.

Native-supported operations include selected scan/reorder paths, positional operations, normalization/dataflow helpers, recurrent ESA updates, and Bolt sequence paths. Operations already efficiently provided by PyTorch libraries such as GEMM, convolution, and standard tensor operations continue to use the corresponding PyTorch/ATen vendor implementations.

## API documentation

See [`API.md`](API.md) for the public API reference and [`examples/`](examples/) for runnable examples.

## Component licenses

MLBricks Kit 1.0.0b1 contains component-specific license notices in addition to the repository-level licensing documents:

- ESA — `mlbricks/esa/LICENSE_ESA.txt`
- Bolt — `mlbricks/bolt/LICENSE_BOLT.txt`
- ElasticBit — `mlbricks/elasticbit/LICENSE_ELASTICBIT.txt`
- VESA — `mlbricks/vesa/LICENSE_VESA.txt`
- VisualBolt — `mlbricks/LICENSE_VISUALBOLT.txt`
- FFNBricks — `mlbricks/ffnbrick/LICENSE_FFNBRICK.txt`
- ResController — `mlbricks/residualbrick/LICENSE_RESIDUALBRICK.txt`
- SOUP — `mlbricks/soup/LICENSE_SOUP.txt`

Also see the repository-level [`LICENSE.md`](LICENSE.md), [`LICENSING_NOTICE.md`](LICENSING_NOTICE.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).

## Licensing

MLBricks Kit 1.0.0b1 is source-available software distributed under the **PolyForm Noncommercial License 1.0.0**.

The public license permits noncommercial use, including personal use, education, academic study, noncommercial research, experimentation, benchmarking, and hobby projects, subject to the complete PolyForm license terms.

**Commercial use is not granted by the public license and requires a separate written commercial license.** This includes commercial products, paid software, SaaS/cloud services, enterprise deployment, paid client work, and other commercial or revenue-generating use where applicable.

Commercial licensing inquiries: **licensing@mlbricks.io**

See [`LICENSE.md`](LICENSE.md), [`LICENSING_NOTICE.md`](LICENSING_NOTICE.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).

## Release

**MLBricks Kit 1.0.0b1** is the beta release line for the package and its current component APIs.


## Training compilation

The default remains `training_compile_mode="default"`.

### Bolt native build control

To install MLBricks without compiling the optional Bolt native extension, set:

`MLBRICKS_BUILD_BOLT_NATIVE=0`

Bolt remains available through its PyTorch/fallback implementation; this flag only disables native extension compilation during installation.

### SOUP component license

The bundled SOUP component license is shipped at `mlbricks/soup/LICENSE_SOUP.txt`.

### Versioning note

MLBricks Kit is released as `1.0.0b1`. Experimental SOUP retains its independent component version `0.1.0a3`.

## Beta native wheel distribution

MLBricks `1.0.0b1` is distributed with prebuilt native wheels for supported Linux, Windows, and macOS targets plus a `py3-none-any` fallback wheel. On a matching platform, pip selects the platform-specific native wheel; otherwise it installs the portable PyTorch fallback instead of compiling C++/CUDA locally.

For release builds, GitHub CI compiles CUDA wheels with a fat architecture list covering NVIDIA compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, 10.0, and 12.0+PTX. The beta native ABI is validated against PyTorch `2.10.x`; the package therefore requires `torch>=2.10,<2.11` until the native bindings migrate to the PyTorch stable ABI.

Source installations default native compilation **off**. Developers who intentionally want to compile from source can set the relevant `MLBRICKS_BUILD_*_NATIVE=1` flags. The corresponding `=0` values explicitly disable each extension.

### Beta native platform support

For `1.0.0b1`, official prebuilt native wheels target Linux x86_64 with CUDA 12.8, Windows x86_64 with CUDA 12.8, and macOS Apple Silicon (`arm64`) for CPU-native acceleration. Intel macOS is not included because the PyTorch 2.10 binary line used by this beta does not provide the required current x86_64 macOS wheels.

Windows CUDA-native users must use the CUDA 12.8 PyTorch build from the official PyTorch `cu128` index. The MLBricks wheel itself remains precompiled, so MLBricks does not compile native code on the user's machine. If CUDA-native prerequisites are unavailable, MLBricks' PyTorch implementation remains the portable fallback.
