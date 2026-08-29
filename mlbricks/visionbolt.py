"""VisionBolt: image models using Bolt with selectable spatial engines."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch import Tensor

from .bolt import Bolt
from .ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from .residualbrick import ResController
from .runtime import (backend_report as collect_backend_report, normalize_backend, build_execution_plan, prepare_module_execution, reset_execution_route, predict_module)
from .vesa.config import FullBackend, VisionConfig
from .vesa.layers.local import LocalDepthwiseConv
from .vesa.layers.normalization import PerspectiveNorm
from .vesa.models.common import MLP
from .vision_common import (
    scan_indices,
    sinusoidal_2d_positions,
    apply_scan_native_or_pytorch,
    restore_scan_native_or_pytorch,
    add_sinusoidal_2d_native_or_pytorch,
)
from .vision_models import ImageDiffusionEngine, VisualAREngine


@dataclass(frozen=True)
class VisionBoltConfig(VisionConfig):
    """Configuration shared by all VisionBolt engines."""

    backend: FullBackend = "auto"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dim % self.heads:
            raise ValueError("VisionBolt dim must be divisible by heads")


class VisionBoltBlock(nn.Module):
    def __init__(self, config: VisionBoltConfig, *, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.ffn_kind = config.ffn
        self.residual_kind = config.residual
        self.use_local = config.engine in {"serpentine", "cnn"}
        self.scan = config.scan
        self.backend = config.backend

        self.local_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        self.local = LocalDepthwiseConv(config.dim, config.local_kernel_size)
        self.cnn_pointwise = (
            nn.Sequential(nn.Linear(config.dim, config.dim), nn.GELU())
            if config.engine == "cnn"
            else nn.Identity()
        )

        self.mixer_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        directional = config.engine in {"serpentine", "cnn"}
        self.mixer = Bolt(
            config.dim,
            config.heads,
            latent_dim=config.latent_dim,
            causal=directional,
            position=None,
            backend=config.backend,
            native_full_sequence=True,
        )
        self.mlp_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)

        if self.ffn_kind == "standard":
            self.mlp = MLP(config.dim, config.mlp_mult)
        elif self.ffn_kind == "ffnbrick":
            self.mlp = StateAwareFFN(
                d_model=config.dim,
                state_dim=config.ffn_state_dim,
                depth_embedding_dim=config.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=config.depth,
                backend=config.backend,
            )
        elif self.ffn_kind == "virtual_ffnbrick":
            self.mlp = VirtualStateAwareFFN(
                d_model=config.dim,
                state_dim=config.ffn_state_dim,
                depth_embedding_dim=config.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=config.depth,
                virtual_refinements=config.ffn_virtual_refinements,
                virtual_hidden_dim=config.ffn_virtual_hidden_dim,
                backend=config.backend,
            )
        elif self.ffn_kind == "micro_ffnbrick":
            self.mlp = MicroVirtualFFN(
                d_model=config.dim,
                hidden_dim=config.ffn_micro_hidden_dim,
                refinements=config.ffn_micro_refinements,
                backend=config.backend,
            )
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled VisionBolt FFN component: {self.ffn_kind}")

        if config.residual == "rescontroller":
            kwargs = dict(
                update_ratio=config.residual_update_ratio,
                stream_ratio=config.residual_stream_ratio,
                update_softness=config.residual_update_softness,
                stream_softness=config.residual_stream_softness,
                backend=config.backend,
            )
            self.local_residual = ResController(**kwargs)
            self.bolt_residual = ResController(**kwargs)
            self.ffn_residual = ResController(**kwargs)
        else:
            self.local_residual = None
            self.bolt_residual = None
            self.ffn_residual = None

    @staticmethod
    def _combine(controller, residual: Tensor, update: Tensor) -> Tensor:
        return residual + update if controller is None else controller(residual, update)

    def _ffn_update(
        self,
        normalized: Tensor,
        bolt_update: Tensor,
        previous_bolt: Tensor | None,
        previous_ffn_state: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.ffn_kind == "standard":
            return self.mlp(normalized), previous_ffn_state
        if self.ffn_kind == "micro_ffnbrick":
            hidden = normalized
            total = torch.zeros_like(normalized)
            for i in range(self.mlp.refinements):
                update = self.mlp(hidden, i)
                total = total + update
                hidden = hidden + update
            return total, previous_ffn_state
        if previous_bolt is None:
            previous_bolt = torch.zeros_like(bolt_update)
        if previous_ffn_state is None:
            previous_ffn_state = self.mlp.initial_state(normalized)
        return self.mlp(normalized, bolt_update, previous_bolt, previous_ffn_state)

    def forward(
        self,
        x: Tensor,
        grid: tuple[int, int],
        order: Tensor | None,
        previous_bolt: Tensor | None,
        previous_ffn_state: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.use_local:
            local = self.local(self.local_norm(x), grid)
            local = self.cnn_pointwise(local)
            x = self._combine(self.local_residual, x, local)

        mixer_in = self.mixer_norm(x)
        use_scan = self.config.engine in {"serpentine", "cnn"}
        inverse = None
        if use_scan:
            mixer_in = apply_scan_native_or_pytorch(
                mixer_in, *grid, scan=self.scan, layer_index=self.layer_index,
                backend=self.backend, owner=self,
            )
        elif order is not None:
            inverse = torch.argsort(order)
            mixer_in = mixer_in.index_select(1, order)
        mixed = self.mixer(mixer_in)
        if use_scan:
            mixed = restore_scan_native_or_pytorch(
                mixed, *grid, scan=self.scan, layer_index=self.layer_index,
                backend=self.backend, owner=self,
            )
        elif inverse is not None:
            mixed = mixed.index_select(1, inverse)
        x = self._combine(self.bolt_residual, x, mixed)

        ffn, next_state = self._ffn_update(
            self.mlp_norm(x), mixed, previous_bolt, previous_ffn_state
        )
        x = self._combine(self.ffn_residual, x, ffn)
        return x, mixed, next_state


class VisionBoltClassifier(nn.Module):
    """Bolt image classifier for Serpentine, ViT, and CNN engines."""

    def __init__(self, config: VisionBoltConfig | None = None):
        super().__init__()
        config = config or VisionBoltConfig()
        if config.engine not in {"serpentine", "vit", "cnn"}:
            raise ValueError("VisionBoltClassifier supports Serpentine, ViT, CNN")
        self.config = config
        self.engine = config.engine
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        g = config.image_size // config.patch_size
        self.grid = (g, g)
        tokens = g * g
        if config.position == "learned":
            self.learned_position = nn.Parameter(torch.zeros(1, tokens, config.dim))
            nn.init.trunc_normal_(self.learned_position, std=0.02)
        else:
            self.register_parameter("learned_position", None)
        self.blocks = nn.ModuleList(
            VisionBoltBlock(config, layer_index=i) for i in range(config.depth)
        )
        self.final_norm = PerspectiveNorm(config.dim, config.perspective_groups, backend=config.backend)
        self.head = nn.Linear(config.dim, config.num_classes)

    def _add_position(self, x: Tensor) -> Tensor:
        if self.config.position is None:
            return x
        if self.config.position == "learned":
            return x + self.learned_position.to(x.dtype)
        if self.config.position == "2d_sincos":
            return add_sinusoidal_2d_native_or_pytorch(
                x, *self.grid, backend=self.config.backend, owner=self
            )
        raise RuntimeError(f"Unhandled position policy: {self.config.position}")

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[1] != self.config.in_channels:
            raise ValueError(f"images must have {self.config.in_channels} channels")
        if images.shape[2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image size does not match configuration")
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = self._add_position(x)
        previous_bolt = None
        ffn_state = None
        for i, block in enumerate(self.blocks):
            x, previous_bolt, ffn_state = block(
                x, self.grid, None, previous_bolt, ffn_state
            )
        return self.head(self.final_norm(x).mean(dim=1))


class VisionBolt(nn.Module):
    """Unified Bolt image family selected by ``engine=``.

    Engines:
      - ``Serpentine``: directional cross-scan classifier, no position by default.
      - ``ViT``: bidirectional patch classifier, 2-D sin/cos position by default.
      - ``CNN``: local-convolution + directional Bolt classifier.
      - ``Diffusion``: image denoiser ``forward(images, timesteps)``.
      - ``AR``: visual-token autoregressive model with ``generate``.
    """

    def __init__(
        self,
        config: VisionBoltConfig | VisionConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is not None and kwargs:
            raise TypeError(
                "VisionBolt accepts either a config object or configuration keyword arguments, not both"
            )
        if config is None:
            config = VisionBoltConfig(**kwargs)
        elif not isinstance(config, VisionBoltConfig):
            config = VisionBoltConfig(**config.__dict__)
        self.config = config
        self.engine = config.engine
        self.backend = normalize_backend(config.backend, warn_legacy=True)

        if self.engine in {"serpentine", "vit", "cnn"}:
            self.engine_model = VisionBoltClassifier(config)
        elif self.engine == "diffusion":
            self.engine_model = ImageDiffusionEngine("bolt", config)
        elif self.engine == "ar":
            self.engine_model = VisualAREngine("bolt", config)
        else:  # pragma: no cover
            raise RuntimeError(f"Unhandled VisionBolt engine: {self.engine}")

    def forward(self, *args, **kwargs):
        return self.engine_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        if self.engine != "ar":
            raise AttributeError("generate is available only when engine='AR'")
        return self.engine_model.generate(*args, **kwargs)

    def benchmark_sample_loop(self, *args, **kwargs):
        if self.engine != "diffusion":
            raise AttributeError("benchmark_sample_loop is available only when engine='Diffusion'")
        return self.engine_model.benchmark_sample_loop(*args, **kwargs)

    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        self._mlbricks_requested_backend = value
        self._mlbricks_model_route = "operator" if value == "auto" else value
        self._mlbricks_model_route_reason = f"explicit:{value}"
        self._mlbricks_model_timings = None
        self.backend = value
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


__all__ = ["VisionBolt", "VisionBoltConfig", "VisionBoltClassifier", "VisionBoltBlock"]
