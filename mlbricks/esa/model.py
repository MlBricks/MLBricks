# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any


import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import warnings

from .generation import (
    GenerationResult,
    GenerationStats,
    sample_next_token,
    parse_engine_spec,
)
from .layer import ESA
from ..components import Embedding, FFN, LayerNorm, LMHead
from ..ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from ..residualbrick import ResController
from ..runtime import (
    normalize_backend, backend_report as collect_backend_report, build_execution_plan, prepare_module_execution, reset_execution_route, predict_module
)

def available_devices():
    """
    Return all devices supported by ESA.
    """

    return {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "xla": importlib.util.find_spec("torch_xla") is not None,
        "npu": importlib.util.find_spec("torch_npu") is not None,
    }


def print_available_devices():
    devices = available_devices()

    print("\nAvailable Devices")
    print("-----------------------------")

    for name, enabled in devices.items():
        print(f"{'✓' if enabled else '✗'} {name}")

    if devices["cuda"]:
        print(f"\nCUDA GPU : {torch.cuda.get_device_name(0)}")


def _resolve_model_device(device: str | torch.device) -> torch.device:
    """Resolve a user device while preserving indexed CUDA devices."""
    if isinstance(device, torch.device):
        requested = device
    else:
        name = str(device).strip().lower()
        if name in {"tpu", "xla"}:
            try:
                import torch_xla.core.xla_model as xm
            except ImportError as exc:
                raise RuntimeError("TPU/XLA requested but torch_xla is not installed.") from exc
            return xm.xla_device()
        requested = torch.device(name)

    if requested.type == "cuda":
        if torch.cuda.is_available():
            return requested
        warnings.warn(
            "CUDA requested but not available. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.device("cpu")
    if requested.type == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return requested
        raise RuntimeError("MPS requested but Apple MPS is not available.")
    if requested.type == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested but torch_npu is not installed.") from exc
        return requested
    if requested.type == "cpu":
        return requested
    raise ValueError(
        f"Unsupported ESA device: {device!r}. "
        "Choose from: 'cuda', 'cuda:N', 'cpu', 'mps', 'tpu', 'xla', or 'npu'."
    )


def _resolve_model_compute_dtype(config: "ESAModelConfig", device: torch.device) -> torch.dtype:
    name = config.compute_dtype
    if name == "auto":
        # Keep CPU reference/training numerically safe. CUDA/MPS use the model's
        # requested precision so Linear/FFN/LM-head computation matches Thunder.
        if device.type not in {"cuda", "mps", "npu"}:
            return torch.float32
        name = config.precision
        if name == "fp8":
            name = "fp16"
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    return mapping.get(name, torch.float32)


@dataclass
class ESAModelConfig:
    vocab_size: int
    block: int = 512
    n_layer: int = 6
    head: int = 6
    embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    # ESA has one production full-sequence backend. Kept in serialized
    # configs for backward checkpoint compatibility; normal users do not need
    # to pass this option.
    backend: str = "auto"
    precision: str = "fp16"
    compute_dtype: str = "auto"
    compass: int | str | None = "auto"

    # Selectable MLBricks components.
    ffn: str = "standard"
    residual: str = "standard"
    ffn_state_dim: int = 256
    ffn_depth_embedding_dim: int = 64
    ffn_virtual_refinements: int = 2
    ffn_virtual_hidden_dim: int = 128
    ffn_micro_hidden_dim: int = 64
    ffn_micro_refinements: int = 1
    residual_update_ratio: float = 0.18
    residual_stream_ratio: float = 1.08
    residual_update_softness: float = 8.0
    residual_stream_softness: float = 8.0

    # Standard embedding / LM-head initialization.
    init_std: float = 0.02

    training_compile: bool = True
    training_compile_mode: str = "default"
    training_compile_fullgraph: bool = False
    gate_min: float = 0.80
    gate_max: float = 0.995
    eps: float = 1e-5
    tie_embeddings: bool = True
    format_version: int = 2

    def __post_init__(self) -> None:
        self.precision = str(self.precision).strip().lower()
        self.compute_dtype = str(self.compute_dtype).strip().lower()
        if self.compute_dtype not in {"auto", "fp16", "bf16", "fp32", "fp64"}:
            raise ValueError(
                "compute_dtype must be one of: auto, fp16, bf16, fp32, fp64."
            )
        self.backend = normalize_backend(self.backend, warn_legacy=True)
        if self.compass is None:
            self.compass = "auto"
        elif isinstance(self.compass, str):
            if self.compass.strip().lower() != "auto":
                raise ValueError("compass must be a positive integer or 'auto'.")
            self.compass = "auto"
        else:
            self.compass = int(self.compass)
            if self.compass <= 0:
                raise ValueError(f"compass must be positive, got {self.compass}.")

        ffn_name = str(self.ffn).strip().lower().replace("-", "_")
        ffn_aliases = {
            "standard": "standard",
            "default": "standard",
            "ffn": "standard",
            "ffnbrick": "ffnbrick",
            "state_aware": "ffnbrick",
            "stateaware": "ffnbrick",
            "virtual_ffnbrick": "virtual_ffnbrick",
            "virtual_state_aware": "virtual_ffnbrick",
            "virtualstateaware": "virtual_ffnbrick",
            "micro_ffnbrick": "micro_ffnbrick",
            "micro_virtual": "micro_ffnbrick",
            "microvirtual": "micro_ffnbrick",
        }
        if ffn_name not in ffn_aliases:
            raise ValueError(
                "ffn must be one of: standard, ffnbrick, virtual_ffnbrick, "
                "micro_ffnbrick."
            )
        self.ffn = ffn_aliases[ffn_name]

        residual_name = str(self.residual).strip().lower().replace("-", "_")
        residual_aliases = {
            "standard": "standard",
            "default": "standard",
            "residual": "standard",
            "rescontroller": "rescontroller",
            "res_controller": "rescontroller",
            "controller": "rescontroller",
        }
        if residual_name not in residual_aliases:
            raise ValueError("residual must be one of: standard, rescontroller.")
        self.residual = residual_aliases[residual_name]

        positive_ints = {
            "ffn_state_dim": self.ffn_state_dim,
            "ffn_depth_embedding_dim": self.ffn_depth_embedding_dim,
            "ffn_virtual_refinements": self.ffn_virtual_refinements,
            "ffn_virtual_hidden_dim": self.ffn_virtual_hidden_dim,
            "ffn_micro_hidden_dim": self.ffn_micro_hidden_dim,
            "ffn_micro_refinements": self.ffn_micro_refinements,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if float(self.init_std) <= 0.0:
            raise ValueError("init_std must be greater than zero.")
        for name in (
            "residual_update_ratio",
            "residual_stream_ratio",
            "residual_update_softness",
            "residual_stream_softness",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ESAModelConfig":
        """Construct a model configuration from serialized dictionary values."""
        allowed = cls.__dataclass_fields__.keys()
        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in allowed
            }
        )


class _ESABlock(nn.Module):
    def __init__(
        self,
        cfg: ESAModelConfig,
        *,
        layer_index: int,
    ):
        super().__init__()
        self.ffn_kind = cfg.ffn
        self.residual_kind = cfg.residual

        self.ln1 = LayerNorm(
            cfg.embd,
            bias=cfg.bias,
        )

        self.esa = ESA(
            embd=cfg.embd,
            head=cfg.head,
            block=cfg.block,
            backend=cfg.backend,
            precision=cfg.precision,
            compass=cfg.compass,
            dropout=cfg.dropout,
            gate_min=cfg.gate_min,
            gate_max=cfg.gate_max,
            eps=cfg.eps,
            device=None,
        )

        self.ln2 = LayerNorm(
            cfg.embd,
            bias=cfg.bias,
        )

        if self.ffn_kind == "standard":
            self.mlp = FFN(
                cfg.embd,
                4 * cfg.embd,
                activation="gelu",
                dropout=cfg.dropout,
                bias=cfg.bias,
            )
        elif self.ffn_kind == "ffnbrick":
            self.mlp = StateAwareFFN(
                d_model=cfg.embd,
                state_dim=cfg.ffn_state_dim,
                depth_embedding_dim=cfg.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=cfg.n_layer,
                backend=cfg.backend,
            )
        elif self.ffn_kind == "virtual_ffnbrick":
            self.mlp = VirtualStateAwareFFN(
                d_model=cfg.embd,
                state_dim=cfg.ffn_state_dim,
                depth_embedding_dim=cfg.ffn_depth_embedding_dim,
                layer_index=layer_index,
                total_layers=cfg.n_layer,
                virtual_refinements=cfg.ffn_virtual_refinements,
                virtual_hidden_dim=cfg.ffn_virtual_hidden_dim,
                backend=cfg.backend,
            )
        elif self.ffn_kind == "micro_ffnbrick":
            self.mlp = MicroVirtualFFN(
                d_model=cfg.embd,
                hidden_dim=cfg.ffn_micro_hidden_dim,
                refinements=cfg.ffn_micro_refinements,
                backend=cfg.backend,
            )
        else:  # guarded by ESAModelConfig
            raise RuntimeError(f"Unhandled FFN component: {self.ffn_kind}")

        if self.residual_kind == "rescontroller":
            controller_kwargs = dict(
                update_ratio=cfg.residual_update_ratio,
                stream_ratio=cfg.residual_stream_ratio,
                update_softness=cfg.residual_update_softness,
                stream_softness=cfg.residual_stream_softness,
            )
            self.esa_residual = ResController(**controller_kwargs, backend=cfg.backend)
            self.ffn_residual = ResController(**controller_kwargs, backend=cfg.backend)
        else:
            self.esa_residual = None
            self.ffn_residual = None

    def _combine_esa_residual(
        self,
        x: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if self.esa_residual is not None:
            return self.esa_residual(x, update)
        return x + update

    def _combine_ffn_residual(
        self,
        x: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if self.ffn_residual is not None:
            return self.ffn_residual(x, update)
        return x + update

    def _residual_and_ln2(
        self,
        x: torch.Tensor,
        update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine the ESA update and prepare the FFN input."""
        if self.esa_residual is not None:
            residual = self.esa_residual(x, update)
            return residual, self.ln2(residual)

        # Preserve the existing fused standard residual + LayerNorm fast path.
        if (
            not torch.is_grad_enabled()
            and x.is_cuda
            and self.ln2.weight is not None
            and self.ln2.bias is not None
            and x.dtype == update.dtype == self.ln2.weight.dtype == self.ln2.bias.dtype
        ):
            try:
                from .native import residual_layer_norm as _native_residual_ln
                from .native import fused_enabled_for as _native_fused_enabled
                if _native_fused_enabled(x):
                    return _native_residual_ln(
                        x, update, self.ln2.weight, self.ln2.bias, self.ln2.eps
                    )
            except (ImportError, AttributeError, RuntimeError):
                import os
                if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                    raise
        residual = x + update
        return residual, self.ln2(residual)

    def _ffn_update(
        self,
        normalized: torch.Tensor,
        esa_update: torch.Tensor,
        previous_esa: torch.Tensor | None,
        previous_ffn_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.ffn_kind == "standard":
            return self.mlp(normalized), previous_ffn_state

        if self.ffn_kind == "micro_ffnbrick":
            # ``refine`` preserves the original sequential residual semantics.
            # The optimized FFNBrick backend can execute all refinement passes
            # inside one native call during eager no-grad inference, while
            # training/torch.compile keep the original PyTorch graph.
            refined = self.mlp.refine(normalized)
            return refined - normalized, previous_ffn_state

        if previous_esa is None:
            previous_esa = torch.zeros_like(esa_update)
        if previous_ffn_state is None:
            previous_ffn_state = self.mlp.initial_state(normalized)
        return self.mlp(
            normalized,
            esa_update,
            previous_esa,
            previous_ffn_state,
        )

    def _mlp_residual(
        self,
        residual: torch.Tensor,
        normalized: torch.Tensor,
        esa_update: torch.Tensor,
        previous_esa: torch.Tensor | None,
        previous_ffn_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the selected FFN component and combine its residual update."""
        if (
            self.ffn_kind == "standard"
            and self.ffn_residual is None
            and not torch.is_grad_enabled()
            and residual.is_cuda
            and not self.mlp.gated
            and self.mlp.activation_name == "gelu"
            and (not self.training or self.mlp.drop.p == 0.0)
            and self.mlp.fc.bias is not None
            and self.mlp.proj.bias is not None
        ):
            try:
                from .native import ffn_gelu_residual as _native_ffn
                from .native import custom_ops_registered as _custom_ops_registered
                if _custom_ops_registered():
                    return (
                        _native_ffn(
                            normalized, residual,
                            self.mlp.fc.weight, self.mlp.fc.bias,
                            self.mlp.proj.weight, self.mlp.proj.bias,
                        ),
                        previous_ffn_state,
                    )
            except (ImportError, AttributeError, RuntimeError):
                import os
                if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                    raise

        update, next_ffn_state = self._ffn_update(
            normalized,
            esa_update,
            previous_esa,
            previous_ffn_state,
        )
        return self._combine_ffn_residual(residual, update), next_ffn_state

    def forward(
        self,
        x: torch.Tensor,
        previous_esa: torch.Tensor | None = None,
        previous_ffn_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        esa_update = self.esa(self.ln1(x))
        x, ln2_x = self._residual_and_ln2(x, esa_update)
        x, next_ffn_state = self._mlp_residual(
            x, ln2_x, esa_update, previous_esa, previous_ffn_state
        )
        return x, esa_update, next_ffn_state

    @torch.no_grad()
    def prefill(
        self,
        x: torch.Tensor,
        *,
        backend: str | None = None,
        compass: int | str | None = None,
        previous_esa: torch.Tensor | None = None,
        previous_ffn_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        esa_update, state = self.esa.prefill(
            self.ln1(x),
            backend=backend,
            compass=compass,
        )
        x, ln2_x = self._residual_and_ln2(x, esa_update)
        x, next_ffn_state = self._mlp_residual(
            x, ln2_x, esa_update, previous_esa, previous_ffn_state
        )
        return x, state, esa_update, next_ffn_state

    lightning_prefill = prefill

    def lightning_step_standard(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fast standard-FFN Lightning path matching MLBricks 0.1.0.

        This deliberately avoids the newer residual/FFN custom-op
        orchestration so ``torch.compile`` sees the same compact fixed-shape
        graph that produced the strong original Lightning decode results.
        Advanced FFNBrick/ResidualBrick models continue to use
        :meth:`lightning_step` below.
        """
        y, new_state = self.esa.decode_step(
            self.ln1(x),
            state,
        )
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, new_state

    def lightning_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        previous_esa: torch.Tensor | None = None,
        previous_ffn_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        esa_update, new_state = self.esa.decode_step(
            self.ln1(x),
            state,
        )
        x, ln2_x = self._residual_and_ln2(x, esa_update)
        x, next_ffn_state = self._mlp_residual(
            x, ln2_x, esa_update, previous_esa, previous_ffn_state
        )
        return x, new_state, esa_update, next_ffn_state


class ESAModel(nn.Module):
    """
    Complete causal language model built from ESA layers.

    Public lifecycle:
        model(...)
        model.generate(...)
        model.save(...)
        ESAModel.load(...)
    """

    def __init__(
        self,
        config: ESAModelConfig | None = None,
        *,
        device: str = "cuda",
        **kwargs: Any,
    ):
        super().__init__()

        if config is None:
            config = ESAModelConfig(
                **kwargs
            )
        elif kwargs:
            raise TypeError(
                "Pass either config=... or keyword configuration, not both."
            )

        self.config = config

        # ==========================================================
        # Device / compute dtype selection
        # ==========================================================
        target_device = _resolve_model_device(device)
        target_dtype = _resolve_model_compute_dtype(config, target_device)

        self.wte = Embedding(
            config.vocab_size,
            config.embd,
        )

        self.wpe = Embedding(
            config.block,
            config.embd,
        )

        self.drop = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList(
            [
                _ESABlock(config, layer_index=index)
                for index in range(config.n_layer)
            ]
        )

        self.ln_f = LayerNorm(
            config.embd,
            bias=config.bias,
        )

        self.lm_head = LMHead(
            config.embd,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            self.wte.weight = self.lm_head.weight

        self.apply(
            self._init_weights
        )

        # FFNBrick virtual paths intentionally start as zero-update refiners.
        # Restore that identity initialization after the model-wide Linear init.
        for block in self.blocks:
            if isinstance(block.mlp, VirtualStateAwareFFN):
                block.mlp.reset_virtual_identity()
            elif isinstance(block.mlp, MicroVirtualFFN):
                block.mlp.reset_identity()

        for name, parameter in self.named_parameters():
            if (
                name.endswith(
                    (
                        "proj.weight",
                        "out_proj.weight",
                    )
                )
                and parameter.ndim >= 2
            ):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=config.init_std
                    / math.sqrt(
                        2 * config.n_layer
                    ),
                )

        self._compiled_lightning_step = None

        self._compiled_lightning_key = None
        self._compiled_training_forward = None
        self._compiled_training_key = None
        self._training_compile_failed = False
        self._compile_warnings_emitted: set[str] = set()
        self.to(device=target_device, dtype=target_dtype)





    def set_backend(self, backend: str, *, recursive: bool = True):
        value = normalize_backend(backend, warn_legacy=True)
        self._mlbricks_requested_backend = value
        self._mlbricks_model_route = "operator" if value == "auto" else value
        self._mlbricks_model_route_reason = f"explicit:{value}"
        self._mlbricks_model_timings = None
        self.config.backend = value
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
        return getattr(self, "_mlbricks_execution_plan", build_execution_plan(self))

    def prepare_execution(self, *sample_args, sample_kwargs=None, warmup=5, trials=20, candidates=("operator", "native", "pytorch"), force=False):
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

    def compile(self, *, mode: str | None = None, fullgraph: bool | None = None, strict: bool = False):
        """Uniform MLBricks compile entrypoint for the ESA training/forward graph."""
        try:
            self.compile_training(mode=mode, fullgraph=fullgraph)
            plan = build_execution_plan(self)
            plan.compiled = self._compiled_training_forward is not None
            plan.compile_mode = mode or self.config.training_compile_mode
            plan.fullgraph = (
                self.config.training_compile_fullgraph if fullgraph is None else bool(fullgraph)
            )
            self._mlbricks_execution_plan = plan
        except Exception:
            if strict:
                raise
        return self

    def prepare_generation(self, *, compile_decode: bool = True, mode: str = "default", fullgraph: bool = False):
        """Prepare ESA's fixed-state decode path; ESA needs no growing KV cache."""
        if compile_decode and self.device.type == "cuda":
            self.compile_generation(mode=mode, fullgraph=fullgraph)
        return self

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

    @property
    def device(self) -> torch.device:
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    @property
    def compute_dtype(self) -> torch.dtype:
        for parameter in self.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        for buffer in self.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        return torch.get_default_dtype()

    def _forward_eager(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = input_ids.shape
        if T > self.config.block:
            raise ValueError(
                f"Sequence length {T} exceeds block {self.config.block}."
            )
        pos = torch.arange(
            T,
            dtype=torch.long,
            device=input_ids.device,
        )
        x = self.drop(self.wte(input_ids) + self.wpe(pos)[None, :, :])
        previous_esa = None
        ffn_state = None
        for block in self.blocks:
            x, previous_esa, ffn_state = block(
                x,
                previous_esa=previous_esa,
                previous_ffn_state=ffn_state,
            )
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def compile_training(
        self,
        *,
        mode: str | None = None,
        fullgraph: bool | None = None,
    ) -> "ESAModel":
        """Compile the full training forward path and cache it."""
        if not hasattr(torch, "compile"):
            return self
        mode = mode or self.config.training_compile_mode
        fullgraph = (
            self.config.training_compile_fullgraph
            if fullgraph is None
            else bool(fullgraph)
        )
        key = (mode, fullgraph)
        if self._compiled_training_key == key and self._compiled_training_forward is not None:
            return self
        try:
            self._compiled_training_forward = torch.compile(
                self._forward_eager,
                mode=mode,
                fullgraph=fullgraph,
            )
            self._compiled_training_key = key
            self._training_compile_failed = False
        except Exception as exc:
            self._compiled_training_forward = None
            self._training_compile_failed = True
            self._warn_compile_fallback("training", exc)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        use_compiled_training = (
            bool(self.config.training_compile)
            and self.device.type == "cuda"
            and self.training
            and targets is not None
            and not self._training_compile_failed
        )
        if use_compiled_training:
            if self._compiled_training_forward is None:
                self.compile_training()
            if self._compiled_training_forward is not None:
                return self._compiled_training_forward(input_ids, targets)
        return self._forward_eager(input_ids, targets)

    @torch.no_grad()
    def _prefill_eager(
        self,
        input_ids: torch.Tensor,
        *,
        backend: str | None = None,
        compass: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be [B,T], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.size(1) > self.config.block:
            input_ids = input_ids[:, -self.config.block:]
        T = input_ids.size(1)
        if T <= 0:
            raise ValueError("Prefill requires at least one token.")
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.wte(input_ids) + self.wpe(pos)[None, :, :])
        states = []
        previous_esa = None
        ffn_state = None
        for block in self.blocks:
            x, state, previous_esa, ffn_state = block.prefill(
                x,
                backend=backend,
                compass=compass,
                previous_esa=previous_esa,
                previous_ffn_state=ffn_state,
            )
            states.append(state)
        states_out = torch.stack(states, dim=0).contiguous()
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return logits, states_out, T

    def _warn_compile_fallback(self, component: str, exc: Exception) -> None:
        import warnings

        if component in self._compile_warnings_emitted:
            return
        self._compile_warnings_emitted.add(component)
        warnings.warn(
            f"ESA {component} compilation failed; falling back to eager execution: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _cudagraph_mark_step_begin(self) -> None:
        """Mark a new CUDA-graph iteration when the PyTorch API is available.

        ``torch.compile(mode="reduce-overhead")`` may use CUDA graphs. ESA
        Lightning carries recurrent state from one compiled invocation into the
        next, so explicitly marking decode-step boundaries prevents PyTorch from
        treating successive autoregressive steps as one graph iteration.
        """
        if self.device.type != "cuda":
            return

        compiler = getattr(torch, "compiler", None)
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)

        if marker is not None:
            marker()

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        engine: str = "thunder",
        compile_mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Run prompt prefill directly through the selected ESA backend.

        Thunder prefill intentionally does not use ``torch.compile``. The
        native C++/CUDA Thunder scan is faster in steady-state benchmarks, so
        wrapping the dynamic prefill graph in Inductor only adds overhead.

        ``compile_mode``, ``fullgraph``, and ``dynamic`` are retained as
        backward-compatible no-op keyword arguments so code written against
        MLBricks 1.0.x does not break. Legacy engine names such as
        ``thunder_compiled_16`` are also accepted by the parser, but prefill
        still executes the same direct native Thunder path.

        Lightning decode compilation is independent and remains controlled by
        :meth:`compile_generation` / ``generate(..., compile=...)``.
        """

        # Backward-compatible API only. Prefill compilation was removed after
        # native Thunder consistently outperformed the compiled wrapper.
        _ = (compile_mode, fullgraph, dynamic)

        spec = parse_engine_spec(engine)

        return self._prefill_eager(
            input_ids,
            backend=spec.backend,
            compass=spec.compass,
        )

    @torch.no_grad()
    def lightning_prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Backward-compatible v2.1 prefill using the model's configured backend."""
        return self._prefill_eager(input_ids)

    def lightning_step(
        self,
        token: torch.Tensor,
        states: torch.Tensor,
        pos_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one recurrent ESA-Lightning decoding step.

        The default ``standard`` FFN + residual configuration uses the exact
        compact decode graph from MLBricks 0.1.0.  Component-selected models
        retain the state-aware generalized path.
        """
        x = self.wte(token) + self.wpe(pos_tensor)[None, :]
        x = self.drop(x)

        new_states = []

        if self.config.ffn == "standard" and self.config.residual == "standard":
            for index, block in enumerate(self.blocks):
                x, state_i = block.lightning_step_standard(
                    x,
                    states[index],
                )
                new_states.append(state_i)
        else:
            previous_esa = None
            ffn_state = None
            for index, block in enumerate(self.blocks):
                x, state_i, previous_esa, ffn_state = block.lightning_step(
                    x,
                    states[index],
                    previous_esa=previous_esa,
                    previous_ffn_state=ffn_state,
                )
                new_states.append(state_i)

        states_out = torch.stack(new_states, dim=0).contiguous()
        logits = self.lm_head(self.ln_f(x))
        return logits, states_out


    def compile_generation(
        self,
        *,
        mode: str = "default",
        fullgraph: bool = False,
    ) -> "ESAModel":
        """
        Compile the fixed-shape ESA-Lightning decode step.

        Lightning decode always processes one token at a time, so its input
        shape is stable. The production default is ``mode="default"`` because
        that is the proven fast path from the public MLBricks 0.1.0 Lightning
        implementation. ``reduce-overhead`` remains available explicitly.

        Prefill and decode intentionally use different compilation policies:

            Prefill:
                dynamic=True
                CUDA Graphs disabled

            Decode:
                dynamic=False
                mode="reduce-overhead"
                CUDA Graph acceleration available
        """

        if (
            not hasattr(
                torch,
                "compile",
            )
            or self.device.type != "cuda"
        ):
            return self


        key = (
            mode,
            bool(
                fullgraph
            ),
        )


        # ==============================================================================================
        # REUSE EXISTING COMPILED LIGHTNING STEP
        # ==============================================================================================

        if (
            self._compiled_lightning_step
            is not None
            and self._compiled_lightning_key
            == key
        ):
            return self


        # ==============================================================================================
        # COMPILE LIGHTNING
        # ==============================================================================================

        try:

            self._compiled_lightning_step = (
                torch.compile(
                    self.lightning_step,

                    mode=mode,

                    fullgraph=fullgraph,

                    # Lightning is fixed-shape:
                    # one token + fixed ESA state.
                    dynamic=False,
                )
            )


            self._compiled_lightning_key = (
                key
            )


        except Exception as exc:

            self._compiled_lightning_step = (
                None
            )


            self._compiled_lightning_key = (
                None
            )


            self._warn_compile_fallback(
                "runtime",
                exc,
            )


        return self




    @torch.inference_mode()
    def generate(
        self,
        prompt: str | torch.Tensor | None = None,
        *,
        tokenizer: Any | None = None,
        input_ids: torch.Tensor | None = None,
        seek: int = 128,
        prefill: str = "thunder_16",
        
        runtime: str = "lightning",
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        seed: int | None = None,
        compile: bool | str = True,
        compile_mode: str = "default",
        progress_interval: int | None = None,
        stats: bool = False,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor | str | GenerationResult:
        """Generate text with optimized ESA defaults.

        CUDA generation compiles the fixed-shape Lightning decode step with the
        proven ``default`` compile mode. Prompt prefill runs native Thunder
        directly without a ``torch.compile`` wrapper. Normal users can pass raw
        text positionally; ``input_ids`` and ``max_new_tokens`` remain available
        for backward compatibility.
        """
        if max_new_tokens is not None:
            if seek != 128 and int(seek) != int(max_new_tokens):
                raise ValueError(
                    "Pass either seek or max_new_tokens, not conflicting values for both."
                )
            seek = int(max_new_tokens)
        seek = int(seek)
        if seek <= 0:
            raise ValueError("seek must be positive.")

        # Backward compatibility: model.generate(tensor, ...).
        if torch.is_tensor(prompt):
            if input_ids is not None:
                raise ValueError("Provide token IDs either positionally or via input_ids, not both.")
            input_ids = prompt
            prompt = None

        if input_ids is None:
            if prompt is None or tokenizer is None:
                raise ValueError(
                    "Provide a text prompt with tokenizer, or use the advanced input_ids API."
                )
            if hasattr(tokenizer, "encode_ordinary"):
                ids = tokenizer.encode_ordinary(prompt)
            elif hasattr(tokenizer, "encode"):
                ids = tokenizer.encode(prompt)
            else:
                raise TypeError(
                    "tokenizer must expose encode_ordinary() or encode()."
                )
            input_ids = torch.tensor(
                ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
        else:
            input_ids = input_ids.to(self.device)

        runtime_spec = parse_engine_spec(runtime)
        if runtime_spec.backend != "lightning":
            raise ValueError(
                "Autoregressive decode currently supports runtime='lightning' only. "
                "Thunder is the selectable prefill engine."
            )
        if isinstance(compile, str):
            compile_policy = compile.strip().lower()
            if compile_policy not in {"auto", "always", "never"}:
                raise ValueError("compile must be bool or one of: auto, always, never")
            if compile_policy == "always":
                compile_runtime = True
            elif compile_policy == "never":
                compile_runtime = False
            else:
                # Autoregressive Lightning has a fixed one-token shape. On CUDA,
                # compiling the whole decode step lets the runtime optimize the
                # embedding, normalization, ESA update, FFN, LM head, and state
                # hand-off together. Native custom ops remain available inside
                # the compiled graph, but they are no longer used as a reason to
                # disable whole-step compilation.
                compile_runtime = bool(self.device.type == "cuda")
        else:
            compile_runtime = bool(compile)
        compile_runtime = bool(compile_runtime or runtime_spec.compiled)

        was_training = self.training
        self.eval()
        try:
            if seed is not None:
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))

            def sync() -> None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

            prompt_tokens = int(input_ids.size(1))
            sync()
            total_start = time.perf_counter()
            prefill_start = total_start
            logits, states, prefill_len = self.prefill(
                input_ids,
                engine=prefill,
            )

            sync()
            prefill_seconds = time.perf_counter() - prefill_start

            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = [next_token]
            finished = torch.zeros(
                input_ids.size(0),
                dtype=torch.bool,
                device=self.device,
            )
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(int(eos_token_id))

            step_fn = self.lightning_step
            using_compiled_runtime = False

            if compile_runtime and self.device.type == "cuda":
                key = (
                    compile_mode,
                    False,
                   
                )

                if (
                    self._compiled_lightning_step is None
                    or self._compiled_lightning_key != key
                ):
                    self.compile_generation(
                        mode=compile_mode,
                        fullgraph=False,
                    )

                if self._compiled_lightning_step is not None:
                    step_fn = self._compiled_lightning_step
                    using_compiled_runtime = True

            decode_target = seek - 1
            positions = (
                torch.arange(decode_target, device=self.device, dtype=torch.long)
                + int(prefill_len)
            ) % int(self.config.block)
            graph_managed_state = bool(
                using_compiled_runtime and str(compile_mode) == "reduce-overhead"
            )
            stable_state = torch.empty_like(states) if graph_managed_state else None

            sync()
            decode_start = time.perf_counter()
            for step in range(decode_target):
                if eos_token_id is not None and bool(finished.all()):
                    break
                pos_tensor = positions[step]
                if graph_managed_state:
                    self._cudagraph_mark_step_begin()
                try:
                    logits, states_out = step_fn(
                        next_token.squeeze(1),
                        states,
                        pos_tensor,
                    )
                except Exception as exc:
                    if not using_compiled_runtime:
                        raise

                    # torch.compile is lazy; graph lowering/capture errors can
                    # appear on the first real invocation. Fall back once.
                    self._compiled_lightning_step = None
                    self._compiled_lightning_key = None
                    self._warn_compile_fallback("runtime-execution", exc)
                    step_fn = self.lightning_step
                    using_compiled_runtime = False
                    graph_managed_state = False
                    stable_state = None
                    logits, states_out = step_fn(
                        next_token.squeeze(1),
                        states,
                        pos_tensor,
                    )

                # Default/eager compilation returns normal tensor ownership, so
                # carry the state directly. reduce-overhead may use CUDA Graph
                # managed output buffers; copy into one persistent buffer there
                # to avoid per-token clone allocations.
                if graph_managed_state:
                    assert stable_state is not None
                    stable_state.copy_(states_out)
                    states = stable_state
                else:
                    states = states_out

                sampled = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                if eos_token_id is not None:
                    eos_value = int(eos_token_id)
                    sampled = torch.where(
                        finished[:, None],
                        torch.full_like(sampled, eos_value),
                        sampled,
                    )
                    finished |= sampled.squeeze(1).eq(eos_value)
                next_token = sampled
                generated.append(next_token)

                if progress_interval and (step + 1) % int(progress_interval) == 0:
                    sync()
                    elapsed = time.perf_counter() - decode_start
                    done = step + 1
                    print(
                        f"ESA-Lightning {done:,}/{decode_target:,} | "
                        f"{done/max(elapsed, 1e-9):,.2f} tok/s"
                    )

            sync()
            decode_seconds = time.perf_counter() - decode_start
            total_seconds = time.perf_counter() - total_start
            generated_ids = torch.cat(generated, dim=1)
            sequences = torch.cat([input_ids, generated_ids], dim=1)
            state_bytes = int(states.numel() * states.element_size())
            generation_stats = GenerationStats(
                prompt_tokens=prompt_tokens,
                prefill_tokens=int(prefill_len),
                generated_tokens=int(generated_ids.size(1)),
                decode_steps=max(0, int(generated_ids.size(1)) - 1),
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                decode_tok_s=max(0, int(generated_ids.size(1)) - 1)
                / max(decode_seconds, 1e-9),
                total_seconds=total_seconds,
                state_bytes=state_bytes,
                state_mb=state_bytes / 1024**2,
            )
            result = GenerationResult(
                sequences=sequences,
                generated_ids=generated_ids,
                stats=generation_stats,
            )
            if tokenizer is not None:
                if sequences.size(0) == 1:
                    result.text = tokenizer.decode(
                        sequences[0].detach().cpu().tolist()
                    )
                else:
                    result.text = [
                        tokenizer.decode(row.detach().cpu().tolist())
                        for row in sequences
                    ]
            if stats:
                return result
            if result.text is not None:
                return result.text
            return sequences
        finally:
            self.train(was_training)



    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        seek: int = 128,
        **kwargs: Any,
    ) -> torch.Tensor | GenerationResult:
        """Advanced token-level generation API."""
        return self.generate(
            input_ids=input_ids,
            seek=seek,
            **kwargs,
        )

    def model_info(
        self,
    ) -> dict[str, Any]:
        """Return architecture, parameter, device, and runtime information."""
        return {
            **asdict(
                self.config
            ),
            "parameters": sum(
                parameter.numel()
                for parameter
                in self.parameters()
            ),
            "device": str(
                self.device
            ),
            "compute_dtype_resolved": str(self.compute_dtype).replace("torch.", ""),
            "generation_engine": (
                "ESA-Lightning"
            ),
        }

    def save(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save the configuration, weights, and optional metadata to a model directory."""
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            path
            / "config.json"
        ).write_text(
            json.dumps(
                asdict(
                    self.config
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        torch.save(
            self.state_dict(),
            path / "model.pt",
        )

        meta = {
            "format_version": self.config.format_version,
            "architecture": "ESAModel",
            "generation_engine": "ESA-Lightning",
            "backend": self.config.backend,
        }
        from ..elasticbit import elasticbit_manifest
        quantization = elasticbit_manifest(self)
        if quantization is not None:
            meta["quantization"] = quantization

        if metadata:
            meta.update(
                metadata
            )

        (
            path
            / "metadata.json"
        ).write_text(
            json.dumps(
                meta,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "ESAModel":
        """Load an ESA model directory and restore its configuration and weights."""
        path = Path(path)

        config = ESAModelConfig.from_dict(
            json.loads(
                (
                    path
                    / "config.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
        )

        target_device = _resolve_model_device(device)
        model = cls(config, device=target_device)

        metadata_path = path / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        quantization = metadata.get("quantization")
        if quantization is not None:
            from ..elasticbit import restore_elasticbit_modules
            restore_elasticbit_modules(model, quantization)
            model.to(device=target_device)

        state = torch.load(
            path / "model.pt",
            map_location=target_device,
            weights_only=True,
        )
        model.load_state_dict(state, strict=strict)
        return model

