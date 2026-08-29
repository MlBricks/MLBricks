# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE_VESA.txt in this directory for the VESA component terms.

"""MLBricks VESA: native VisionESA models and C++/CUDA ESA kernels.

Simple public usage::

    from mlbricks import Vesa

    model = Vesa(image_size=32, num_classes=10, dim=192, depth=4)

The convenience ``VesaConfig`` uses MLBricks backend="auto" by default.
Advanced VESA model families remain available from ``mlbricks.vesa``.
"""

from dataclasses import dataclass, replace

import torch.nn as nn

from .backends.native import native_available, native_cuda_built
from .config import (
    AutoregressiveConfig,
    DiffusionConfig,
    ESAConfig,
    FullBackend,
    VisionConfig,
)
from .layers.mixer import ESAMixer
from .models.autoregressive import ESAARModel
from .models.classifier import VisionESAClassifier
from .models.diffusion import ESADiffusionModel
from .utils import count_parameters
from ..runtime import (normalize_backend, backend_report as collect_backend_report, build_execution_plan, prepare_module_execution, reset_execution_route, predict_module)
from ..vision_models import ImageDiffusionEngine, VisualAREngine


@dataclass(frozen=True)
class VesaConfig(VisionConfig):
    """Simple VESA image-model configuration.

    It has the same fields as :class:`VisionConfig` and uses the uniform
    MLBricks ``auto`` backend policy by default.
    """

    backend: FullBackend = "auto"


class Vesa(VisionESAClassifier):
    """Unified ESA vision family selected by ``engine=``.

    Examples::

        Vesa(engine="Serpentine", position=None, scan="cross")
        Vesa(engine="ViT", position="auto")
        Vesa(engine="CNN", position=None)
        Vesa(engine="Diffusion", position=None, scan="cross")
        Vesa(engine="AR", position=None, scan="cross")

    ``position="auto"`` resolves to ``None`` for Serpentine/CNN/Diffusion/AR
    and to ``"2d_sincos"`` for ViT. Explicit ``None`` always disables spatial
    positional embeddings.
    """

    def __init__(
        self,
        config: VesaConfig | VisionConfig | None = None,
        **kwargs,
    ):
        if config is not None and kwargs:
            raise TypeError(
                "Vesa accepts either a config object or configuration keyword "
                "arguments, not both"
            )

        if config is None:
            config = VesaConfig(**kwargs)

        engine = config.engine
        if engine in {"serpentine", "vit", "cnn"}:
            # Preserve the historical inheritance/API for classifier engines.
            super().__init__(config)
            self.engine_model = None
        else:
            # Vesa remains a VisionESAClassifier subclass for compatibility,
            # but Diffusion/AR have different forward signatures and therefore
            # use a dedicated delegate module.
            nn.Module.__init__(self)
            self.config = config
            self.engine = engine
            self.engine_model = None
            if self.engine == "diffusion":
                self.engine_model = ImageDiffusionEngine("esa", config)
            elif self.engine == "ar":
                self.engine_model = VisualAREngine("esa", config)
            else:  # pragma: no cover - config validation handles this
                raise RuntimeError(f"Unhandled VESA engine: {self.engine}")

        self.backend = normalize_backend(config.backend, warn_legacy=True)
        self.engine = engine

    def forward(self, *args, **kwargs):
        if self.engine_model is not None:
            return self.engine_model(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self.engine != "ar" or self.engine_model is None:
            raise AttributeError("generate is available only when engine='AR'")
        return self.engine_model.generate(*args, **kwargs)

    def benchmark_sample_loop(self, *args, **kwargs):
        if self.engine != "diffusion" or self.engine_model is None:
            raise AttributeError(
                "benchmark_sample_loop is available only when engine='Diffusion'"
            )
        return self.engine_model.benchmark_sample_loop(*args, **kwargs)

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        self._mlbricks_requested_backend = value
        self._mlbricks_model_route = "operator" if value == "auto" else value
        self._mlbricks_model_route_reason = f"explicit:{value}"
        self._mlbricks_model_timings = None
        self.backend = value
        if hasattr(self, "config"):
            try:
                self.config = replace(self.config, backend=value)
            except Exception:
                pass
        if recursive:
            for module in self.modules():
                if module is self:
                    continue
                setter = getattr(module, "set_backend", None)
                if callable(setter):
                    try:
                        setter(value, recursive=False)
                    except TypeError:
                        setter(value)
                elif hasattr(module, "backend"):
                    try:
                        module.backend = value
                        from .planner import EXECUTION_PLANNER
                        EXECUTION_PLANNER.clear_owner_routes(module)
                    except ImportError:
                        try:
                            from ..planner import EXECUTION_PLANNER
                            EXECUTION_PLANNER.clear_owner_routes(module)
                        except Exception:
                            pass
                    except Exception:
                        pass
        return self

    def backend_report(self):
        return collect_backend_report(self)

    def execution_plan(self):
        """Return the current heterogeneous model + operator planner summary."""
        return build_execution_plan(self)

    def prepare_execution(self, *sample_args, sample_kwargs=None, warmup=5, trials=20, candidates=("operator", "native", "pytorch"), force=False):
        """Calibrate the hierarchical inference planner on representative input."""
        return prepare_module_execution(
            self, *sample_args, sample_kwargs=sample_kwargs, warmup=warmup,
            trials=trials, candidates=tuple(candidates), force=force,
        )

    def predict(self, *args, device="auto", dtype="auto", calibrate=True, **kwargs):
        """One-call optimized inference with automatic placement and planning."""
        return predict_module(
            self, *args, device=device, dtype=dtype, calibrate=calibrate, **kwargs
        )

    def reset_execution(self):
        reset_execution_route(self)
        return self


# Advanced aliases for the three VESA model families.
AR = ESAARModel
Classifier = VisionESAClassifier
Diffusion = ESADiffusionModel

__all__ = [
    "Vesa",
    "VesaConfig",
    "AR",
    "Classifier",
    "Diffusion",
    "AutoregressiveConfig",
    "DiffusionConfig",
    "ESAARModel",
    "ESAConfig",
    "ESADiffusionModel",
    "ESAMixer",
    "VisionConfig",
    "VisionESAClassifier",
    "count_parameters",
    "native_available",
    "native_cuda_built",
]

__version__ = "1.0.0"
