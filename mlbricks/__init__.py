# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE.md and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from .bolt import Attention, Bolt, BoltAttention
from .soup import SOUP, soup
from .gaussian import Gaussian, GaussianConfig
# Canonical ready-made Bolt model names; Gaussian names remain compatibility aliases.
BoltModel = Gaussian
BoltConfig = GaussianConfig
from .bricks import Bricks, Brick
from .visionbolt import VisionBolt, VisionBoltConfig
from .vision_native import available as vision_native_available, cuda_built as vision_native_cuda_built
from .position import RoPE, LearnedPosition, SinusoidalPosition
from .runtime import (
    normalize_backend, set_module_backend, backend_report,
    ExecutionPlan, build_execution_plan, prepare_module_execution, apply_execution_route, reset_execution_route, predict_module,
)
from .planner import (
    EXECUTION_PLANNER, MLBricksExecutionPlanner, AUTO_OPERATOR_DEFAULTS,
)

# ``import mlbricks.attention`` resolves to the canonical Bolt implementation.
import sys as _sys
from .bolt import attention as _attention_module
_sys.modules.setdefault(__name__ + ".attention", _attention_module)
from . import vesa
from .vesa import Vesa, VesaConfig
from . import ffnbrick
from .ffnbrick import MicroVirtualFFN, StateAwareFFN, VirtualStateAwareFFN
from .ffnbrick import native_backend_available as ffnbrick_native_backend_available
from .ffnbrick import native_backend_name as ffnbrick_native_backend_name
from . import residualbrick
from .residualbrick import ResController
from .residualbrick import native_backend_available as residualbrick_native_backend_available
from .residualbrick import native_backend_name as residualbrick_native_backend_name
from .esa import (
    ESA,
    ESAConfig,
    ESAModel,
    ESAModelConfig,
)
from .lifecycle import save, load, inspect, predict, generate, compile, quantize
from .trainer import Trainer, TrainerState, train
from .optim import Adam, AdamW, FP16_ADAM_MIN_EPS, stabilize_optimizer
from .esa import GenerationResult, GenerationStats
from .esa import compass, CompassResult
from .esa import ThunderESA
from .esa import (
    ESABenchmarkConfig,
    DEFAULT_BENCHMARK_CONFIG,
    FAST_BENCHMARK_CONFIG,
    PAPER_BENCHMARK_CONFIG,
    BENCHMARK_DEFAULTS,
    FAST_BENCHMARK_DEFAULTS,
    PAPER_BENCHMARK_DEFAULTS,
    TrainingIntervalTimer,
    cuda_telemetry,
)
from .esa import thunderBoost
from .components import (
    FFN,
    Embedding,
    LMHead,
    Linear,
    LayerNorm,
    RMSNorm,
    Residual,
    ffn,
    embedding,
    embeddings,
    lmhead,
    linear,
    layernorm,
    rmsnorm,
    residual,
)
from .elasticbit import (
    ElasticBit,
    ElasticBitConfig,
    PackedElasticBit,
    ElasticLinear,
    ElasticEmbedding,
    quantize_tensor,
    dequantize_tensor,
    quantize_module,
)

# Friendly lowercase constructor alias.
esa = ESA

__version__ = "1.0.0"

__all__ = [
    "Attention",
    "Bolt",
    "BoltAttention",
    "SOUP",
    "soup",
    "BoltModel",
    "BoltConfig",
    "Gaussian",
    "GaussianConfig",
    "Bricks",
    "Brick",
    "VisionBolt",
    "VisionBoltConfig",
    "vision_native_available",
    "vision_native_cuda_built",
    "RoPE",
    "LearnedPosition",
    "SinusoidalPosition",
    "normalize_backend",
    "set_module_backend",
    "backend_report",
    "ExecutionPlan",
    "build_execution_plan",
    "predict_module",
    "prepare_module_execution",
    "apply_execution_route",
    "reset_execution_route",
    "EXECUTION_PLANNER",
    "MLBricksExecutionPlanner",
    "AUTO_OPERATOR_DEFAULTS",
    "vesa",
    "Vesa",
    "VesaConfig",
    "ffnbrick",
    "MicroVirtualFFN",
    "StateAwareFFN",
    "VirtualStateAwareFFN",
    "ffnbrick_native_backend_available",
    "ffnbrick_native_backend_name",
    "residualbrick",
    "ResController",
    "residualbrick_native_backend_available",
    "residualbrick_native_backend_name",
    "ESA",
    "esa",
    "ESAConfig",
    "ESAModel",
    "ESAModelConfig",
    "Trainer",
    "TrainerState",
    "save",
    "load",
    "inspect",
    "predict",
    "generate",
    "compile",
    "quantize",
    "train",
    "Adam",
    "AdamW",
    "FP16_ADAM_MIN_EPS",
    "stabilize_optimizer",
    "GenerationResult",
    "GenerationStats",
    "compass",
    "CompassResult",
    "ThunderESA",
    "thunderBoost",
    "FFN",
    "Embedding",
    "LMHead",
    "Linear",
    "LayerNorm",
    "RMSNorm",
    "Residual",
    "ffn",
    "embedding",
    "embeddings",
    "lmhead",
    "linear",
    "layernorm",
    "rmsnorm",
    "residual",
    "ElasticBit",
    "ElasticBitConfig",
    "PackedElasticBit",
    "ElasticLinear",
    "ElasticEmbedding",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_module",
    "ESABenchmarkConfig",
    "DEFAULT_BENCHMARK_CONFIG",
    "FAST_BENCHMARK_CONFIG",
    "PAPER_BENCHMARK_CONFIG",
    "BENCHMARK_DEFAULTS",
    "FAST_BENCHMARK_DEFAULTS",
    "PAPER_BENCHMARK_DEFAULTS",
    "TrainingIntervalTimer",
    "cuda_telemetry",
]
