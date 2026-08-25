# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ESABenchmarkConfig:
    """
    Benchmark settings for ESA experiments.

    These settings are separate from the normal ESA layer.
    ESA layers do not automatically warm up, compile, train, or benchmark.
    """

    compile_warmup_steps: int = 2
    speed_warmup_steps: int = 2
    speed_bench_steps: int = 10
    compile_mode: str = "reduce-overhead"
    reset_seed_after_compile_warmup: bool = True


DEFAULT_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=2,
    speed_warmup_steps=2,
    speed_bench_steps=10,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


FAST_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=1,
    speed_warmup_steps=1,
    speed_bench_steps=6,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


PAPER_BENCHMARK_CONFIG = ESABenchmarkConfig(
    compile_warmup_steps=3,
    speed_warmup_steps=5,
    speed_bench_steps=30,
    compile_mode="reduce-overhead",
    reset_seed_after_compile_warmup=True,
)


# Dict aliases for users who prefer dictionary-style access.
BENCHMARK_DEFAULTS = asdict(DEFAULT_BENCHMARK_CONFIG)
FAST_BENCHMARK_DEFAULTS = asdict(FAST_BENCHMARK_CONFIG)
PAPER_BENCHMARK_DEFAULTS = asdict(PAPER_BENCHMARK_CONFIG)
# -----------------------------------------------------------------------------
# Measurement helpers for reproducible GPU benchmarks.
# -----------------------------------------------------------------------------
import subprocess
import time
from typing import Any


class TrainingIntervalTimer:
    """Measure training-only wall time while explicitly excluding validation.

    Call ``pause()`` before evaluation/checkpointing and ``resume()`` before the
    next training step. ``tokens_per_second(tokens)`` then reports only active
    training time, fixing the common validation-contamination artifact.
    """

    def __init__(self) -> None:
        self._elapsed = 0.0
        self._start = time.perf_counter()
        self._running = True

    def pause(self) -> None:
        if self._running:
            self._elapsed += time.perf_counter() - self._start
            self._running = False

    def resume(self) -> None:
        if not self._running:
            self._start = time.perf_counter()
            self._running = True

    def reset(self) -> None:
        self._elapsed = 0.0
        self._start = time.perf_counter()
        self._running = True

    @property
    def elapsed(self) -> float:
        value = self._elapsed
        if self._running:
            value += time.perf_counter() - self._start
        return float(value)

    def tokens_per_second(self, tokens: int) -> float:
        return float(tokens) / max(self.elapsed, 1e-12)


def cuda_telemetry(device: int = 0) -> dict[str, Any]:
    """Return NVIDIA GPU clocks/utilization/temp/power for benchmark logs.

    The function is deliberately optional: if ``nvidia-smi`` is unavailable it
    returns an empty dictionary rather than affecting model execution.
    """
    fields = [
        "utilization.gpu",
        "clocks.sm",
        "clocks.mem",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "memory.used",
        "memory.total",
    ]
    cmd = [
        "nvidia-smi",
        f"--id={int(device)}",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        row = [value.strip() for value in output.splitlines()[0].split(",")]
        if len(row) != len(fields):
            return {}
        result: dict[str, Any] = {}
        for key, value in zip(fields, row):
            clean = key.replace(".", "_")
            try:
                result[clean] = float(value)
            except ValueError:
                result[clean] = value
        return result
    except Exception:
        return {}
