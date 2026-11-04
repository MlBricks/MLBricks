"""Entangled State Attention (ESA) component package."""

# Primary ESA API: the original Fast-EsaQ concrete module, no per-token MLBricks wrapper.
from .fast_engine.public_api import ESA, save_esa_checkpoint, load_esa_checkpoint

from .model import ESAModel, ESAModelConfig
from .generation import GenerationResult, GenerationStats
from .compass import compass, CompassResult
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
    "save_esa_checkpoint",
    "load_esa_checkpoint",
    "ESAModel",
    "ESAModelConfig",
    "GenerationResult",
    "GenerationStats",
    "compass",
    "CompassResult",
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
