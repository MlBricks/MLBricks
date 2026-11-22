"""Entangled State Attention (ESA) component package."""

from .config import ESAConfig
from .selector import ESA, LegacyESA, RetentionESA
from .model import ESAModel, ESAModelConfig
from .generation import GenerationResult, GenerationStats
from .compass import compass, CompassResult
from .backends import ThunderESA
from .benchmark import (
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
from .boost import thunderBoost

__all__ = [
    "ESA",
    "LegacyESA",
    "RetentionESA",
    "CurrentESA",
    "BaseESA",
    "CurrentESAConfig",
    "CurrentESAModel",
    "BaseESAModel",
    "DEFAULT_CURRENT_HALF_LIVES",
    "make_esa",
    "make_current_esa_model",
    "current_esa_forward_tokens",
    "save_current_esa_checkpoint",
    "load_current_esa_checkpoint",
    "load_current_esa_model_checkpoint",
    "CurrentESAGraphDecoder",
    "ESAConfig",
    "ESAModel",
    "ESAModelConfig",
    "GenerationResult",
    "GenerationStats",
    "compass",
    "CompassResult",
    "ThunderESA",
    "ESABenchmarkConfig",
    "DEFAULT_BENCHMARK_CONFIG",
    "FAST_BENCHMARK_CONFIG",
    "PAPER_BENCHMARK_CONFIG",
    "BENCHMARK_DEFAULTS",
    "FAST_BENCHMARK_DEFAULTS",
    "PAPER_BENCHMARK_DEFAULTS",
    "TrainingIntervalTimer",
    "cuda_telemetry",
    "thunderBoost",
]

# Retention ESA is the public default; legacy implementation stays explicitly available.
from .half_life import (CurrentESA, BaseESA, CurrentESAConfig, CurrentESAModel,
    BaseESAModel, DEFAULT_CURRENT_HALF_LIVES, make_esa, make_current_esa_model,
    current_esa_forward_tokens, save_current_esa_checkpoint, load_current_esa_checkpoint,
    load_current_esa_model_checkpoint, CurrentESAGraphDecoder)
