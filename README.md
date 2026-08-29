# MLBricks v1.0.0

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

The public Python package is `mlbricks`.

## Version

```python
import mlbricks
print(mlbricks.__version__)
# 1.0.0
```

## Installation

From PyPI:

```bash
pip install mlbricks
```

For development or installation directly from the repository root:

```bash
pip install -e .
```

For a CPU-only native build:

```bash
MLBRICKS_FORCE_CPU=1 pip install -e .
```

Optional native components can be disabled during installation when needed:

```text
MLBRICKS_BUILD_CORE_NATIVE=0
MLBRICKS_BUILD_VISION_NATIVE=0
MLBRICKS_BUILD_VESA_NATIVE=0
MLBRICKS_BUILD_FFNBRICK_NATIVE=0
MLBRICKS_BUILD_RESIDUALBRICK_NATIVE=0
MLBRICKS_BUILD_ELASTICBIT_NATIVE=0
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

- `auto` — qualify **each element independently** (for example ESA, Bolt, SAFFN, a vision scan, or ElasticLinear), choose native or PyTorch once, and keep that element's route stable during execution. Composite models can mix routes.
- `native` — require a supported MLBricks native implementation.
- `pytorch` — force the PyTorch/reference path.

Example:

```python
bolt.set_backend("native")
bolt.set_backend("pytorch")
bolt.set_backend("auto")
```

## ESA

```python
from mlbricks import ESA

esa = ESA(
    embd=384,
    head=6,
)
```

For the ready-made ESA model lifecycle APIs:

```python
from mlbricks import ESAModel, ESAModelConfig
```

## Bolt

`Bolt` and `BoltAttention` are the public attention names in MLBricks v1.0.0.

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
one-time decisions. Per-layer mixer and FFN choices remain supported.

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

model = Bricks(
    vocab_size=50_257,
    dim=384,
    context=2048,
    position="sinusoidal",
    layers=[
        Brick(
            mixer=ESA(embd=384, head=6),
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
)
```

## Native vision runtime

VESA and VisualBolt can use the shared MLBricks vision runtime when the compiled backend is available. The public backend contract remains `auto | native | pytorch`.

Native-supported operations include selected scan/reorder paths, positional operations, normalization/dataflow helpers, recurrent ESA updates, and Bolt sequence paths. Operations already efficiently provided by PyTorch libraries such as GEMM, convolution, and standard tensor operations continue to use the corresponding PyTorch/ATen vendor implementations.

## API documentation

See [`API.md`](API.md) for the public API reference and [`examples/`](examples/) for runnable examples.

## Component licenses

MLBricks v1.0.0 contains component-specific license notices in addition to the repository-level licensing documents:

- ESA — `mlbricks/esa/LICENSE_ESA.txt`
- Bolt — `mlbricks/bolt/LICENSE_BOLT.txt`
- ElasticBit — `mlbricks/elasticbit/LICENSE_ELASTICBIT.txt`
- VESA — `mlbricks/vesa/LICENSE_VESA.txt`
- VisualBolt — `mlbricks/LICENSE_VISUALBOLT.txt`
- FFNBricks — `mlbricks/ffnbrick/LICENSE_FFNBRICK.txt`
- ResController — `mlbricks/residualbrick/LICENSE_RESIDUALBRICK.txt`

Also see the repository-level [`LICENSE.md`](LICENSE.md), [`LICENSING_NOTICE.md`](LICENSING_NOTICE.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).

## Licensing

MLBricks v1.0.0 is source-available software distributed under the **PolyForm Noncommercial License 1.0.0**.

The public license permits noncommercial use, including personal use, education, academic study, noncommercial research, experimentation, benchmarking, and hobby projects, subject to the complete PolyForm license terms.

**Commercial use is not granted by the public license and requires a separate written commercial license.** This includes commercial products, paid software, SaaS/cloud services, enterprise deployment, paid client work, and other commercial or revenue-generating use where applicable.

Commercial licensing inquiries: **licensing@mlbricks.io**

See [`LICENSE.md`](LICENSE.md), [`LICENSING_NOTICE.md`](LICENSING_NOTICE.md), and [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md).

## Release

**MLBricks v1.0.0** is the clean public release line for the package and its current component APIs.


## Training compilation

The default remains `training_compile_mode="default"`.
