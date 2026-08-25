# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""Unified MLBricks execution planner.

The original ESA planner is now the library-wide planner.  It keeps ESA's
Compass/hierarchy resource model intact and adds operation-level routing for
Bolt, vision operators, FFNBrick, ResidualBrick/ResController and ElasticBit.

The planner never dispatches by marketing GPU name.  Keys are derived from
CUDA capability/resources, tensor shape/dtype, execution mode and operator
metadata.  ``backend='native'`` and ``backend='pytorch'`` remain strict user
choices; only ``backend='auto'`` is planner-controlled.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Final, Mapping

import torch

from .runtime import normalize_backend

AUTO_COMPASS_CANDIDATES: Final[tuple[int, ...]] = (8, 16, 32, 64)
DIRECT_PREFIX_KERNEL_MAX: Final[int] = 1024

# Conservative initial routing.  Benchmark results always override these
# defaults.  Two T4 qualification findings are intentionally reflected here:
# generic index_select beat the custom scan kernel for the tested image shape,
# and SDPA/ATen beat the fused full-sequence Bolt kernel at B32/T196/D384.
# Native is still available via backend='native', and benchmark calibration can
# promote it for shapes/devices where it wins.
AUTO_OPERATOR_DEFAULTS: Final[dict[str, str]] = {
    "esa_scan": "native",
    "bolt_decode": "native",
    "attention_decode": "native",
    "bolt_full": "pytorch",
    "vision_scan": "pytorch",
    "vision_position_2d": "native",
    "vision_unpatchify": "native",
    "vision_patchify_layout": "native",
    "perspective_norm": "native",
    "ffnbrick_state": "native",
    "ffnbrick_virtual": "native",
    "ffnbrick_micro": "native",
    "rescontroller": "native",
    "elastic_linear": "pytorch",
}


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def floor_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value.bit_length() - 1)


def ceil_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << ((value - 1).bit_length())


def sequence_bucket(length: int) -> int:
    """Bucket nearby prompt lengths so auto-tuning is reusable interactively."""
    return max(64, ceil_power_of_two(int(length)))


@dataclass(frozen=True)
class HardwareProfile:
    device_index: int
    compute_capability: tuple[int, int]
    multiprocessors: int
    warp_size: int
    max_threads_per_sm: int

    @property
    def resident_threads(self) -> int:
        return self.multiprocessors * self.max_threads_per_sm


class MLBricksExecutionPlanner:
    """One resource/route cache shared by every MLBricks component."""

    def __init__(self, direct_kernel_max: int = DIRECT_PREFIX_KERNEL_MAX) -> None:
        self.direct_kernel_max = int(direct_kernel_max)

        # Existing ESA planner state (kept byte-for-byte compatible in spirit).
        self.group_cache: dict[tuple[object, ...], int] = {}
        self.compass_cache: dict[tuple[object, ...], int] = {}
        self.group_benchmarks: dict[tuple[object, ...], list[tuple[int, float, int]]] = {}
        self.compass_benchmarks: dict[tuple[object, ...], list[tuple[int, float]]] = {}
        self.route_cache: dict[tuple[object, ...], object] = {}

        # Library-wide operation routing.
        self.operator_cache: dict[tuple[object, ...], str] = {}
        self.operator_benchmarks: dict[tuple[object, ...], dict[str, float]] = {}
        self.operator_reasons: dict[tuple[object, ...], str] = {}
        # Monotonic revision used by component-local hot-path route caches.
        # It changes only when an externally visible operator decision changes.
        self._operator_revision: int = 0

        # Hierarchical/model-level routing.  These entries select a composed
        # execution policy after operator calibration has already happened.
        # Routes are: operator (keep per-op auto decisions), native, pytorch.
        self.model_cache: dict[tuple[object, ...], str] = {}
        self.model_benchmarks: dict[tuple[object, ...], dict[str, float]] = {}
        self.model_reasons: dict[tuple[object, ...], str] = {}

    @staticmethod
    def _device_index(device: torch.device | str) -> int:
        resolved = torch.device(device)
        if resolved.index is not None:
            return int(resolved.index)
        return int(torch.cuda.current_device())

    def hardware_profile(self, device: torch.device | str) -> HardwareProfile:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA hardware profile requested without CUDA")
        index = self._device_index(device)
        props = torch.cuda.get_device_properties(index)
        return HardwareProfile(
            device_index=index,
            compute_capability=(int(props.major), int(props.minor)),
            multiprocessors=int(props.multi_processor_count),
            warp_size=int(getattr(props, "warp_size", 32)),
            max_threads_per_sm=int(
                getattr(props, "max_threads_per_multi_processor", 2048)
            ),
        )

    # ------------------------------------------------------------------
    # Existing ESA resource model
    # ------------------------------------------------------------------
    def direct_summary_budget(
        self, device: torch.device | str, batch: int, channels: int
    ) -> int:
        hw = self.hardware_profile(device)
        workers = max(1, int(batch) * int(channels))
        concurrency = max(1, hw.resident_threads // workers)
        resource_budget = floor_power_of_two(concurrency * hw.warp_size)
        resource_budget = max(hw.warp_size, resource_budget)
        return min(self.direct_kernel_max, resource_budget)

    def group_candidates(
        self, summary_count: int, direct_budget: int, device: torch.device | str
    ) -> list[int]:
        hw = self.hardware_profile(device)
        minimum_needed = ceil_power_of_two(
            ceil_div(int(summary_count), int(direct_budget))
        )
        start = max(hw.warp_size, minimum_needed)
        end = min(int(direct_budget), int(summary_count))
        candidates: list[int] = []
        group_size = ceil_power_of_two(start)
        while group_size <= end:
            candidates.append(group_size)
            group_size *= 2
        if not candidates:
            candidates = [min(int(direct_budget), int(summary_count))]
        return candidates

    def group_key(self, tensor: torch.Tensor, direct_budget: int) -> tuple[object, ...]:
        hw = self.hardware_profile(tensor.device)
        return (
            hw.device_index,
            hw.compute_capability,
            hw.multiprocessors,
            int(tensor.shape[0]),
            int(tensor.shape[1]),
            int(tensor.shape[2]),
            str(tensor.dtype),
            int(direct_budget),
        )

    def compass_key_values(
        self,
        device: torch.device | str,
        *,
        batch: int,
        time: int,
        channels: int,
        dtype: torch.dtype,
        training: bool,
    ) -> tuple[object, ...]:
        hw = self.hardware_profile(device)
        return (
            hw.device_index,
            hw.compute_capability,
            hw.multiprocessors,
            int(batch),
            sequence_bucket(int(time)),
            int(channels),
            str(dtype),
            bool(training),
        )

    def compass_key(self, tensor: torch.Tensor, *, training: bool) -> tuple[object, ...]:
        channels = int(tensor.shape[2] * tensor.shape[3])
        return self.compass_key_values(
            tensor.device,
            batch=int(tensor.shape[0]),
            time=int(tensor.shape[1]),
            channels=channels,
            dtype=tensor.dtype,
            training=training,
        )

    # ------------------------------------------------------------------
    # Library-wide operation routing
    # ------------------------------------------------------------------
    def _device_signature(self, tensor: torch.Tensor) -> tuple[object, ...]:
        if tensor.is_cuda and torch.cuda.is_available():
            hw = self.hardware_profile(tensor.device)
            return (
                "cuda",
                hw.device_index,
                hw.compute_capability,
                hw.multiprocessors,
                hw.warp_size,
            )
        return (tensor.device.type, tensor.device.index)

    def operator_key(
        self,
        op: str,
        tensor: torch.Tensor,
        *,
        training: bool,
        extra: tuple[object, ...] = (),
    ) -> tuple[object, ...]:
        shape = tuple(int(v) for v in tensor.shape)
        # Exact B/D with a reusable power-of-two sequence bucket where a
        # sequence dimension exists. This avoids a new plan for T=193 vs 196.
        if len(shape) >= 3:
            bucketed_shape = (shape[0], sequence_bucket(shape[-2]), shape[-1])
            if len(shape) > 3:
                bucketed_shape = shape[:-2] + bucketed_shape[-2:]
        else:
            bucketed_shape = shape
        return (
            "op",
            str(op),
            *self._device_signature(tensor),
            str(tensor.dtype),
            bucketed_shape,
            bool(training),
            *tuple(extra),
        )

    @property
    def operator_revision(self) -> int:
        """Revision of the operator routing table.

        Components use this to keep a one-entry local route cache without
        becoming stale after explicit calibration, manual overrides, or a
        planner reset. Merely reading a cached decision does not change it.
        """
        return int(self._operator_revision)

    @staticmethod
    def _local_route_signature(
        op: str,
        tensor: torch.Tensor,
        *,
        requested_backend: str,
        training: bool,
        extra: tuple[object, ...],
        native_available: bool,
        native_supports_training: bool,
    ) -> tuple[object, ...]:
        # Exact shape is deliberate for the component-local single-entry cache.
        # The global planner still uses reusable sequence buckets.
        return (
            str(op),
            str(requested_backend),
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
            tuple(tensor.shape),
            bool(training),
            bool(native_available),
            bool(native_supports_training),
            tuple(extra),
        )

    def select_operator_cached(
        self,
        owner: object,
        op: str,
        tensor: torch.Tensor,
        *,
        requested_backend: str = "auto",
        native_available: bool,
        native_supports_training: bool = False,
        training: bool | None = None,
        extra: tuple[object, ...] = (),
        default_auto: str | None = None,
    ) -> str:
        """Resolve an operator using a component-local hot-path cache.

        ``select_operator`` remains the authoritative global shape/device cache.
        This helper adds a one-entry cache on the owning module so steady-state
        inference avoids rebuilding the global planner key and re-running
        availability/dispatch logic on every forward. The entry is invalidated
        whenever the planner revision changes.
        """
        raw_policy = str(requested_backend).strip().lower()
        policy = (
            raw_policy
            if raw_policy in {"auto", "native", "pytorch"}
            else normalize_backend(requested_backend, warn_legacy=True)
        )
        is_training = bool(torch.is_grad_enabled()) if training is None else bool(training)

        # Explicit policies are intentionally direct and never memoized as auto.
        if policy != "auto":
            return self.select_operator(
                op, tensor, requested_backend=policy,
                native_available=native_available,
                native_supports_training=native_supports_training,
                training=is_training, extra=extra, default_auto=default_auto,
            )

        signature = self._local_route_signature(
            op, tensor, requested_backend=policy, training=is_training,
            extra=extra, native_available=native_available,
            native_supports_training=native_supports_training,
        )
        cached = getattr(owner, "_mlbricks_auto_route_cache", None)
        if (
            isinstance(cached, tuple)
            and len(cached) == 3
            and cached[0] == self._operator_revision
            and cached[1] == signature
            and cached[2] in {"native", "pytorch"}
        ):
            return cached[2]

        route = self.select_operator(
            op, tensor, requested_backend="auto",
            native_available=native_available,
            native_supports_training=native_supports_training,
            training=is_training, extra=extra, default_auto=default_auto,
        )
        try:
            setattr(
                owner, "_mlbricks_auto_route_cache",
                (self._operator_revision, signature, route),
            )
        except Exception:
            pass
        return route

    def select_operator(
        self,
        op: str,
        tensor: torch.Tensor,
        *,
        requested_backend: str = "auto",
        native_available: bool,
        native_supports_training: bool = False,
        training: bool | None = None,
        extra: tuple[object, ...] = (),
        default_auto: str | None = None,
    ) -> str:
        """Resolve one operation to ``native`` or ``pytorch``.

        The decision is cached for auto mode. Explicit native/pytorch requests
        are never overridden by the planner.
        """
        policy = normalize_backend(requested_backend, warn_legacy=True)
        is_training = bool(torch.is_grad_enabled()) if training is None else bool(training)

        if policy == "pytorch":
            return "pytorch"
        if policy == "native":
            if not native_available:
                raise RuntimeError(
                    f"backend='native' requested but native route for {op!r} is unavailable"
                )
            if is_training and not native_supports_training:
                raise RuntimeError(
                    f"backend='native' requested for {op!r}, but its native route is "
                    "inference-only for this call"
                )
            return "native"

        # Auto: training safety beats every cached/heuristic choice.
        if is_training and not native_supports_training:
            return "pytorch"
        if not native_available:
            return "pytorch"

        key = self.operator_key(op, tensor, training=is_training, extra=extra)
        cached = self.operator_cache.get(key)
        if cached in {"native", "pytorch"}:
            return cached

        default = str(
            default_auto or AUTO_OPERATOR_DEFAULTS.get(str(op), "pytorch")
        ).lower()
        if default not in {"native", "pytorch"}:
            default = "pytorch"
        route = default if (default != "native" or native_available) else "pytorch"
        self.operator_cache[key] = route
        self.operator_reasons[key] = f"heuristic:{route}"
        return route

    def set_operator_route(
        self,
        op: str,
        tensor: torch.Tensor,
        route: str,
        *,
        training: bool = False,
        extra: tuple[object, ...] = (),
        reason: str = "manual",
    ) -> str:
        value = str(route).strip().lower()
        if value not in {"native", "pytorch"}:
            raise ValueError("route must be 'native' or 'pytorch'")
        key = self.operator_key(op, tensor, training=training, extra=extra)
        previous = self.operator_cache.get(key)
        self.operator_cache[key] = value
        self.operator_reasons[key] = str(reason)
        if previous != value:
            self._operator_revision += 1
        return value

    def record_operator_benchmark(
        self,
        op: str,
        tensor: torch.Tensor,
        timings_ms: Mapping[str, float],
        *,
        training: bool = False,
        extra: tuple[object, ...] = (),
    ) -> str:
        valid = {
            str(name): float(ms)
            for name, ms in timings_ms.items()
            if str(name) in {"native", "pytorch"} and float(ms) > 0.0
        }
        if not valid:
            raise ValueError("timings_ms must contain positive native/pytorch timings")
        winner = min(valid, key=valid.get)
        key = self.operator_key(op, tensor, training=training, extra=extra)
        previous = self.operator_cache.get(key)
        self.operator_benchmarks[key] = valid
        self.operator_cache[key] = winner
        if previous != winner:
            self._operator_revision += 1
        self.operator_reasons[key] = "benchmark:" + ",".join(
            f"{k}={v:.6f}ms" for k, v in sorted(valid.items())
        )
        return winner

    def benchmark_candidates(
        self,
        op: str,
        tensor: torch.Tensor,
        candidates: Mapping[str, Callable[[], object]],
        *,
        warmup: int = 5,
        trials: int = 20,
        training: bool = False,
        extra: tuple[object, ...] = (),
    ) -> tuple[str, dict[str, float]]:
        """Microbenchmark candidate callables and cache the fastest route.

        Intended for explicit calibration (e.g. Kaggle qualification or model
        preparation), not hidden synchronization inside normal forward passes.
        """
        usable = {k: v for k, v in candidates.items() if k in {"native", "pytorch"}}
        if not usable:
            raise ValueError("candidates must include 'native' and/or 'pytorch'")
        warmup = max(0, int(warmup))
        trials = max(1, int(trials))
        timings: dict[str, float] = {}

        for name, fn in usable.items():
            for _ in range(warmup):
                fn()
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                samples = []
                for _ in range(trials):
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
            else:
                samples = []
                for _ in range(trials):
                    t0 = time.perf_counter()
                    fn()
                    samples.append((time.perf_counter() - t0) * 1000.0)
            samples.sort()
            timings[name] = samples[len(samples) // 2]

        winner = self.record_operator_benchmark(
            op, tensor, timings, training=training, extra=extra
        )
        return winner, timings


    # ------------------------------------------------------------------
    # Hierarchical model-level routing
    # ------------------------------------------------------------------
    @staticmethod
    def _model_signature(module) -> tuple[object, ...]:
        config = getattr(module, "config", None)
        fields = []
        for name in (
            "engine", "dim", "depth", "layers", "n_layer", "embd",
            "heads", "head", "latent_dim", "ffn", "residual",
            "image_size", "patch_size", "context", "block", "vocab_size",
        ):
            value = getattr(config, name, None) if config is not None else None
            if value is None:
                value = getattr(module, name, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                fields.append((name, value))
        try:
            params = sum(int(p.numel()) for p in module.parameters())
        except Exception:
            params = -1
        return (module.__class__.__module__, module.__class__.__qualname__, params, *fields)

    def model_key(
        self, module, tensor: torch.Tensor, *, training: bool = False,
        extra: tuple[object, ...] = (),
    ) -> tuple[object, ...]:
        return (
            "model",
            *self._device_signature(tensor),
            str(tensor.dtype),
            self.operator_key("__model_input__", tensor, training=training)[-2],
            bool(training),
            self._model_signature(module),
            *tuple(extra),
        )

    def select_model_route(
        self, module, tensor: torch.Tensor, *, training: bool = False,
        extra: tuple[object, ...] = (), default: str = "operator",
    ) -> str:
        if training:
            return "operator"
        key = self.model_key(module, tensor, training=training, extra=extra)
        route = self.model_cache.get(key)
        if route in {"operator", "native", "pytorch"}:
            return route
        route = str(default).strip().lower()
        if route not in {"operator", "native", "pytorch"}:
            route = "operator"
        self.model_cache[key] = route
        self.model_reasons[key] = f"heuristic:{route}"
        return route

    def record_model_benchmark(
        self, module, tensor: torch.Tensor, timings_ms: Mapping[str, float], *,
        training: bool = False, extra: tuple[object, ...] = (),
    ) -> str:
        valid = {
            str(name): float(ms)
            for name, ms in timings_ms.items()
            if str(name) in {"operator", "native", "pytorch"} and float(ms) > 0.0
        }
        if not valid:
            raise ValueError("model timings must contain positive operator/native/pytorch values")
        winner = min(valid, key=valid.get)
        key = self.model_key(module, tensor, training=training, extra=extra)
        self.model_benchmarks[key] = valid
        self.model_cache[key] = winner
        self.model_reasons[key] = "benchmark:" + ",".join(
            f"{k}={v:.6f}ms" for k, v in sorted(valid.items())
        )
        return winner

    def model_decisions(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key, route in self.model_cache.items():
            rows.append({
                "key": key,
                "route": route,
                "reason": self.model_reasons.get(key, "cached"),
                "benchmarks_ms": self.model_benchmarks.get(key),
            })
        return rows

    def operator_decisions(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for key, route in self.operator_cache.items():
            rows.append(
                {
                    "key": key,
                    "route": route,
                    "reason": self.operator_reasons.get(key, "cached"),
                    "benchmarks_ms": self.operator_benchmarks.get(key),
                }
            )
        return rows

    def clear(self) -> None:
        self.group_cache.clear()
        self.compass_cache.clear()
        self.group_benchmarks.clear()
        self.compass_benchmarks.clear()
        self.route_cache.clear()
        self.operator_cache.clear()
        self.operator_benchmarks.clear()
        self.operator_reasons.clear()
        self._operator_revision += 1
        self.model_cache.clear()
        self.model_benchmarks.clear()
        self.model_reasons.clear()


# Old public name remains an alias so existing imports/checkpoints/scripts work.
ESAExecutionPlanner = MLBricksExecutionPlanner
EXECUTION_PLANNER = MLBricksExecutionPlanner()

__all__ = [
    "AUTO_COMPASS_CANDIDATES",
    "AUTO_OPERATOR_DEFAULTS",
    "DIRECT_PREFIX_KERNEL_MAX",
    "HardwareProfile",
    "MLBricksExecutionPlanner",
    "ESAExecutionPlanner",
    "EXECUTION_PLANNER",
    "ceil_div",
    "floor_power_of_two",
    "ceil_power_of_two",
    "sequence_bucket",
]
