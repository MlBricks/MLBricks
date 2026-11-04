"""Entangled State Attention (ESA) component package."""

from .config import ESAConfig
from .layer import ESA
from .fast_engine import BaseESA as FastBaseESA, CurrentESA as FastCurrentESA
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
    "FastCurrentESA",
    "FastBaseESA",
    "ESAConfig",
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
