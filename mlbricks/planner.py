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
    "esa_element": "native",
    "esa_scan": "native",
    "vesa_scan": "native",
    "vesa_decode": "native",
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

    def select_operator_once(
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
        """Resolve ``backend='auto'`` once for an owning component.

        Unlike the shape-local hot-path cache, this is a *stable execution
        decision*: after the first auto resolution for an operation/workload,
        the owner keeps that native/PyTorch route for its lifetime. It is not
        reconsidered because utilization, temperature, transient load, planner
        calibration revisions, or later calls change. Explicit ``native`` and
        ``pytorch`` requests always bypass the frozen auto route.

        The cache is separated by operation and training/eval workload because
        those can require fundamentally different implementations. Components
        should only call this helper after checking that their native path is
        valid for the current input.
        """
        policy = normalize_backend(requested_backend, warn_legacy=True)
        is_training = bool(torch.is_grad_enabled()) if training is None else bool(training)
        if policy != "auto":
            return self.select_operator(
                op, tensor, requested_backend=policy,
                native_available=native_available,
                native_supports_training=native_supports_training,
                training=is_training, extra=extra, default_auto=default_auto,
            )

        cache = getattr(owner, "_mlbricks_frozen_auto_routes", None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(owner, "_mlbricks_frozen_auto_routes", cache)
            except Exception:
                pass
        key = (str(op), bool(is_training))
        route = cache.get(key)
        if route in {"native", "pytorch"}:
            return route

        route = self.select_operator(
            op, tensor, requested_backend="auto",
            native_available=native_available,
            native_supports_training=native_supports_training,
            training=is_training, extra=extra, default_auto=default_auto,
        )
        cache[key] = route
        return route

    @staticmethod
    def clear_owner_routes(owner: object) -> None:
        """Forget one element's frozen auto decisions after an explicit reset."""
        try:
            setattr(owner, "_mlbricks_frozen_auto_routes", {})
        except Exception:
            pass
        try:
            setattr(owner, "_mlbricks_auto_route_cache", None)
        except Exception:
            pass
        try:
            setattr(owner, "_mlbricks_frozen_auto_details", {})
        except Exception:
            pass

    @staticmethod
    def owner_routes(owner: object) -> dict[tuple[object, ...], str]:
        """Return a copy of one element's frozen auto routes for reporting."""
        cache = getattr(owner, "_mlbricks_frozen_auto_routes", None)
        if not isinstance(cache, dict):
            return {}
        return {k: v for k, v in cache.items() if v in {"native", "pytorch"}}

    def _auto_default(
        self,
        op: str,
        *,
        fallback: str | None = None,
    ) -> str:
        value = str(fallback or AUTO_OPERATOR_DEFAULTS.get(str(op), "pytorch")).lower()
        return value if value in {"native", "pytorch"} else "pytorch"

    @staticmethod
    def _parity_tolerances(
        tensor: torch.Tensor,
        *,
        rtol: float | None,
        atol: float | None,
    ) -> tuple[float, float]:
        """Return conservative one-time native-vs-PyTorch parity tolerances."""
        if rtol is None:
            rtol = 2.0e-2 if tensor.dtype in {torch.float16, torch.bfloat16} else 1.0e-4
        if atol is None:
            atol = 2.0e-2 if tensor.dtype in {torch.float16, torch.bfloat16} else 1.0e-5
        return max(0.0, float(rtol)), max(0.0, float(atol))

    @classmethod
    def _compare_candidate_outputs(
        cls,
        reference: object,
        candidate: object,
        *,
        rtol: float,
        atol: float,
        path: str = "output",
    ) -> tuple[bool, dict[str, object]]:
        """Recursively compare candidate output against the PyTorch reference.

        Tensor leaves are checked with ``torch.allclose`` while tuple/list/dict
        structure must match exactly.  The returned details are intentionally
        compact so they can be stored on the owning element for diagnostics.
        """
        stats: dict[str, object] = {
            "checked": True,
            "allclose": True,
            "rtol": float(rtol),
            "atol": float(atol),
            "tensor_leaves": 0,
            "max_abs": 0.0,
            "max_relative_l2": 0.0,
            "mismatch": None,
        }

        def fail(where: str, reason: str) -> bool:
            stats["allclose"] = False
            if stats["mismatch"] is None:
                stats["mismatch"] = f"{where}: {reason}"
            return False

        def visit(ref: object, cand: object, where: str) -> bool:
            if torch.is_tensor(ref) or torch.is_tensor(cand):
                if not (torch.is_tensor(ref) and torch.is_tensor(cand)):
                    return fail(where, "tensor/non-tensor type mismatch")
                if tuple(ref.shape) != tuple(cand.shape):
                    return fail(where, f"shape {tuple(ref.shape)} != {tuple(cand.shape)}")

                ref_cmp = ref.detach()
                cand_cmp = cand.detach()
                if ref_cmp.device != cand_cmp.device:
                    cand_cmp = cand_cmp.to(ref_cmp.device)
                # Compare floating/complex tensors in a stable accumulator dtype.
                if ref_cmp.is_floating_point() or ref_cmp.is_complex():
                    if ref_cmp.is_complex():
                        ref_num = ref_cmp.to(torch.complex64)
                        cand_num = cand_cmp.to(torch.complex64)
                    else:
                        ref_num = ref_cmp.float()
                        cand_num = cand_cmp.float()
                    diff = (cand_num - ref_num).abs()
                    max_abs = float(diff.max().item()) if diff.numel() else 0.0
                    ref_norm = torch.linalg.vector_norm(ref_num.reshape(-1))
                    diff_norm = torch.linalg.vector_norm((cand_num - ref_num).reshape(-1))
                    rel_l2 = float((diff_norm / (ref_norm + 1.0e-12)).item()) if diff.numel() else 0.0
                    stats["max_abs"] = max(float(stats["max_abs"]), max_abs)
                    stats["max_relative_l2"] = max(float(stats["max_relative_l2"]), rel_l2)
                    close = bool(torch.allclose(cand_num, ref_num, rtol=rtol, atol=atol, equal_nan=False))
                else:
                    close = bool(torch.equal(cand_cmp, ref_cmp))
                stats["tensor_leaves"] = int(stats["tensor_leaves"]) + 1
                if not close:
                    return fail(where, "tensor values differ from PyTorch reference")
                return True

            if isinstance(ref, Mapping) or isinstance(cand, Mapping):
                if not (isinstance(ref, Mapping) and isinstance(cand, Mapping)):
                    return fail(where, "mapping/non-mapping type mismatch")
                if set(ref.keys()) != set(cand.keys()):
                    return fail(where, "mapping keys differ")
                ok = True
                for key in sorted(ref.keys(), key=lambda value: repr(value)):
                    ok = visit(ref[key], cand[key], f"{where}[{key!r}]") and ok
                return ok

            if isinstance(ref, (tuple, list)) or isinstance(cand, (tuple, list)):
                if type(ref) is not type(cand):
                    return fail(where, f"sequence types differ: {type(ref).__name__} != {type(cand).__name__}")
                if len(ref) != len(cand):
                    return fail(where, f"sequence lengths differ: {len(ref)} != {len(cand)}")
                ok = True
                for index, (ref_item, cand_item) in enumerate(zip(ref, cand)):
                    ok = visit(ref_item, cand_item, f"{where}[{index}]") and ok
                return ok

            if ref is None or cand is None:
                if ref is cand:
                    return True
                return fail(where, "None/non-None mismatch")

            try:
                equal = bool(ref == cand)
            except Exception:
                equal = repr(ref) == repr(cand)
            if not equal:
                return fail(where, f"value mismatch: {ref!r} != {cand!r}")
            return True

        ok = visit(reference, candidate, path)
        stats["allclose"] = bool(ok and stats["allclose"])
        return bool(stats["allclose"]), stats

    def qualify_operator_once(
        self,
        owner: object,
        op: str,
        tensor: torch.Tensor,
        candidates: Mapping[str, Callable[[], object]],
        *,
        requested_backend: str = "auto",
        native_available: bool,
        native_supports_training: bool = False,
        training: bool | None = None,
        extra: tuple[object, ...] = (),
        default_auto: str | None = None,
        warmup: int = 2,
        trials: int = 5,
        switch_margin: float = 0.05,
        verify_parity: bool = True,
        parity_rtol: float | None = None,
        parity_atol: float | None = None,
    ) -> str:
        """Validate and benchmark one element once, then freeze its route.

        ``backend='auto'`` is correctness-first and element-local.  When both
        candidates are available, the PyTorch path is executed once as the
        reference and the native output must match it within dtype-appropriate
        tolerances before native is allowed into the speed race.  If parity
        fails (or native raises), PyTorch is frozen immediately for that element.

        Only parity-qualified candidates are timed.  A small hysteresis band
        keeps the conservative default when candidates are within
        ``switch_margin`` of each other.  After the winner is frozen, later
        calls do not re-check load/hardware or switch routes unless the owner is
        explicitly reset/reconfigured.
        """
        policy = normalize_backend(requested_backend, warn_legacy=True)
        is_training = bool(torch.is_grad_enabled()) if training is None else bool(training)
        if policy != "auto":
            return self.select_operator(
                op, tensor, requested_backend=policy, native_available=native_available,
                native_supports_training=native_supports_training, training=is_training,
                extra=extra, default_auto=default_auto,
            )
        if is_training and not native_supports_training:
            return "pytorch"
        if not native_available:
            return self.select_operator_once(
                owner, op, tensor, requested_backend="auto", native_available=False,
                native_supports_training=native_supports_training, training=is_training,
                extra=extra, default_auto=default_auto,
            )

        cache = getattr(owner, "_mlbricks_frozen_auto_routes", None)
        if not isinstance(cache, dict):
            cache = {}
            try:
                setattr(owner, "_mlbricks_frozen_auto_routes", cache)
            except Exception:
                pass
        frozen_key = (str(op), bool(is_training))
        frozen = cache.get(frozen_key)
        if frozen in {"native", "pytorch"}:
            return frozen

        usable = {
            str(name): fn for name, fn in candidates.items()
            if str(name) in {"native", "pytorch"} and callable(fn)
        }
        if not usable:
            return self.select_operator_once(
                owner, op, tensor, requested_backend="auto", native_available=native_available,
                native_supports_training=native_supports_training, training=is_training,
                extra=extra, default_auto=default_auto,
            )

        key = self.operator_key(op, tensor, training=is_training, extra=extra)
        timings: dict[str, float] = {}
        errors: dict[str, str] = {}
        parity: dict[str, object] = {"checked": False, "allclose": None}

        def freeze(route: str, reason: str) -> str:
            self.operator_benchmarks[key] = dict(timings)
            previous = self.operator_cache.get(key)
            self.operator_cache[key] = route
            self.operator_reasons[key] = reason
            if previous != route:
                self._operator_revision += 1
            cache[frozen_key] = route
            try:
                details = getattr(owner, "_mlbricks_frozen_auto_details", None)
                if not isinstance(details, dict):
                    details = {}
                    setattr(owner, "_mlbricks_frozen_auto_details", details)
                details[frozen_key] = {
                    "route": route,
                    "timings_ms": dict(timings),
                    "errors": dict(errors),
                    "parity": dict(parity),
                    "reason": reason,
                }
            except Exception:
                pass
            return route

        # Correctness gate: PyTorch is the reference implementation. Native is
        # eligible for timing only after its first output matches that reference.
        if bool(verify_parity) and "pytorch" in usable and "native" in usable:
            rtol, atol = self._parity_tolerances(
                tensor, rtol=parity_rtol, atol=parity_atol,
            )
            try:
                reference_output = usable["pytorch"]()
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
            except Exception as exc:
                errors["pytorch_reference"] = f"{type(exc).__name__}: {exc}"
                # Native is never auto-selected without a successful PyTorch
                # reference comparison. Preserve correctness-first semantics.
                parity.update({
                    "checked": False,
                    "allclose": None,
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "reference_error": errors["pytorch_reference"],
                })
                return freeze("pytorch", "correctness-gate:reference-error;pytorch")

            try:
                native_output = usable["native"]()
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
            except Exception as exc:
                errors["native_validation"] = f"{type(exc).__name__}: {exc}"
                parity.update({
                    "checked": True,
                    "allclose": False,
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "mismatch": "native candidate raised during correctness validation",
                })
                return freeze("pytorch", "correctness-gate:native-error;pytorch")

            parity_ok, parity_details = self._compare_candidate_outputs(
                reference_output,
                native_output,
                rtol=rtol,
                atol=atol,
            )
            parity.clear()
            parity.update(parity_details)
            if not parity_ok:
                return freeze("pytorch", "correctness-gate:parity-failed;pytorch")

        warmup = max(0, int(warmup))
        trials = max(1, int(trials))
        for name, fn in usable.items():
            try:
                for _ in range(warmup):
                    fn()
                if tensor.is_cuda:
                    torch.cuda.synchronize(tensor.device)
                    samples: list[float] = []
                    for _ in range(trials):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
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
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"

        if not timings:
            route = self.select_operator_once(
                owner, op, tensor, requested_backend="auto", native_available=native_available,
                native_supports_training=native_supports_training, training=is_training,
                extra=extra, default_auto=default_auto,
            )
            return route

        # If native passed parity but later failed during timing, never select it.
        # If PyTorch alone failed during timing, native is the only qualified route.
        fastest = min(timings, key=timings.get)
        default = self._auto_default(op, fallback=default_auto)
        route = fastest
        if default in timings and fastest != default:
            base_ms = float(timings[default])
            best_ms = float(timings[fastest])
            improvement = (base_ms - best_ms) / max(base_ms, 1.0e-12)
            if improvement < max(0.0, float(switch_margin)):
                route = default

        reason = "correctness-pass;benchmark-once:" if parity.get("checked") else "benchmark-once:"
        reason += ",".join(
            f"{name}={ms:.6f}ms" for name, ms in sorted(timings.items())
        )
        if errors:
            reason += ";errors=" + ",".join(f"{k}:{v}" for k, v in sorted(errors.items()))
        return freeze(route, reason)

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

        # Auto: training safety and native availability beat every cached or
        # heuristic choice. Record these fallbacks too so diagnostics can show
        # why an element resolved to PyTorch.
        key = self.operator_key(op, tensor, training=is_training, extra=extra)
        if is_training and not native_supports_training:
            self.operator_cache[key] = "pytorch"
            self.operator_reasons[key] = "training-safe:pytorch"
            return "pytorch"
        if not native_available:
            self.operator_cache[key] = "pytorch"
            self.operator_reasons[key] = "native-unavailable:pytorch"
            return "pytorch"

        cached = self.operator_cache.get(key)
        if cached in {"native", "pytorch"}:
            return cached

        default = self._auto_default(op, fallback=default_auto)
        route = default if (default != "native" or native_available) else "pytorch"
        self.operator_cache[key] = route
        self.operator_reasons[key] = (
            f"heuristic:{route}" if route == default else f"native-unavailable:{route}"
        )
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
