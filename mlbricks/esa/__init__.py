"""Entangled State Attention (ESA) component package."""

from .config import ESAConfig
from .layer import ESA
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
