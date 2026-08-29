"""Uniform MLBricks execution-backend policy.

Public components use the same three backend names:

``auto``
    Qualify each backend-aware element independently on its first eligible
    inference call, choose native or PyTorch for that element, and freeze the
    winner for steady-state execution. Composite models can therefore remain
    heterogeneous instead of forcing one whole-model backend.
``native``
    Require the native implementation.  Components raise a RuntimeError when
    their native path is unavailable for the current call.
``pytorch``
    Always use the PyTorch/reference implementation.

The default is always ``auto``.  Historical backend names are accepted as
compatibility aliases but are normalized immediately.
"""
from __future__ import annotations

import warnings
import time
from dataclasses import replace, is_dataclass
from typing import Any

import torch

VALID_BACKENDS = frozenset({"auto", "native", "pytorch"})

# Compatibility only. New docs/API never need these names.
_LEGACY = {
    "cuda": "native",
    "cpp": "native",
    "cpp/cuda": "native",
    "torch": "pytorch",
    "python": "pytorch",
    "reference": "pytorch",
    "thunder": "auto",
    "flare": "pytorch",
    "pulse": "pytorch",
    "lightning": "pytorch",
}


def normalize_backend(value: str | None = "auto", *, warn_legacy: bool = False) -> str:
    """Return canonical ``auto`` / ``native`` / ``pytorch`` backend name."""
    if value is None:
        return "auto"
    name = str(value).strip().lower().replace("-", "_")
    if name in VALID_BACKENDS:
        return name
    if name in _LEGACY:
        mapped = _LEGACY[name]
        if warn_legacy:
            warnings.warn(
                f"backend={value!r} is a legacy MLBricks backend name; "
                f"use backend={mapped!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return mapped
    raise ValueError(
        f"backend must be one of: auto, native, pytorch; got {value!r}."
    )


def set_module_backend(module: Any, backend: str, *, recursive: bool = True) -> Any:
    """Set backend on a component/model and optionally all backend-aware children."""
    value = normalize_backend(backend)
    setter = getattr(module, "set_backend", None)
    if callable(setter):
        try:
            return setter(value, recursive=recursive)
        except TypeError:
            setter(value)
            if recursive and hasattr(module, "modules"):
                for child in module.modules():
                    if child is module:
                        continue
                    child_setter = getattr(child, "set_backend", None)
                    if callable(child_setter):
                        child_setter(value)
            return module
    if hasattr(module, "backend"):
        module.backend = value
    if recursive and hasattr(module, "modules"):
        for child in module.modules():
            if child is module:
                continue
            child_setter = getattr(child, "set_backend", None)
            if callable(child_setter):
                child_setter(value)
            elif hasattr(child, "backend"):
                child.backend = value
    return module


def backend_report(module: Any) -> list[dict[str, str]]:
    """Collect requested/resolved backend information from a module tree."""
    rows: list[dict[str, str]] = []
    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        return rows
    for name, child in named_modules():
        if child is module and not name:
            continue
        requested = getattr(child, "backend", None)
        resolver = getattr(child, "resolved_backend", None)
        if requested is None and not callable(resolver):
            continue
        resolved = "dynamic"
        if callable(resolver):
            try:
                resolved = str(resolver())
            except TypeError:
                resolved = "dynamic-by-input"
            except Exception:
                resolved = "unavailable"
        rows.append(
            {
                "name": name or child.__class__.__name__,
                "component": child.__class__.__name__,
                "requested": str(requested or "auto"),
                "resolved": resolved,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Hierarchical execution route helpers
# ---------------------------------------------------------------------------

def _set_backend_direct(module: Any, backend: str) -> None:
    """Set one module's backend without recursively calling public setters.

    This is used only by explicit execution preparation.  It updates the common
    MLBricks backend/config fields so composed model plans can be benchmarked
    without rebuilding the model or touching parameters.
    """
    value = normalize_backend(backend)
    if hasattr(module, "backend"):
        try:
            module.backend = value
        except Exception:
            pass
    if hasattr(module, "runtime_backend"):
        try:
            module.runtime_backend = value
        except Exception:
            pass
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "backend"):
        try:
            if is_dataclass(config):
                module.config = replace(config, backend=value)
            else:
                config.backend = value
        except Exception:
            try:
                config.backend = value
            except Exception:
                pass
    if hasattr(module, "use_native"):
        try:
            module.use_native = value != "pytorch"
        except Exception:
            pass


def apply_execution_route(module: Any, route: str) -> Any:
    """Apply a composed inference route to a full module tree.

    ``operator`` restores normal ``backend='auto'`` per-operation planning.
    ``native`` and ``pytorch`` force the corresponding route across backend-aware
    children. Standard PyTorch Linear/Conv/Norm operations remain ATen/cuBLAS/
    cuDNN either way.
    """
    name = str(route).strip().lower()
    if name not in {"operator", "native", "pytorch"}:
        raise ValueError("execution route must be operator, native, or pytorch")
    backend = "auto" if name == "operator" else name
    modules = getattr(module, "modules", None)
    if callable(modules):
        for child in modules():
            _set_backend_direct(child, backend)
    else:
        _set_backend_direct(module, backend)
    setattr(module, "_mlbricks_model_route", name)
    return module


def reset_execution_route(module: Any) -> Any:
    """Return a prepared model to auto and allow every element to re-qualify."""
    apply_execution_route(module, "operator")
    try:
        from .planner import EXECUTION_PLANNER
        modules = getattr(module, "modules", None)
        if callable(modules):
            for child in modules():
                EXECUTION_PLANNER.clear_owner_routes(child)
        else:
            EXECUTION_PLANNER.clear_owner_routes(module)
    except Exception:
        pass
    setattr(module, "_mlbricks_requested_backend", "auto")
    setattr(module, "_mlbricks_model_route_reason", "reset:elementwise-auto")
    setattr(module, "_mlbricks_model_timings", None)
    setattr(module, "_mlbricks_model_diagnostic_winner", None)
    return module


def _first_tensor(value: Any):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _median_call_ms(fn, tensor: torch.Tensor, *, warmup: int, trials: int) -> float:
    for _ in range(max(0, int(warmup))):
        fn()
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
        samples = []
        for _ in range(max(1, int(trials))):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    else:
        samples = []
        for _ in range(max(1, int(trials))):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def prepare_module_execution(
    module: Any,
    *sample_args: Any,
    sample_kwargs: dict[str, Any] | None = None,
    warmup: int = 5,
    trials: int = 20,
    candidates: tuple[str, ...] = ("operator", "native", "pytorch"),
    force: bool = False,
) -> "ExecutionPlan":
    """Prepare inference while preserving heterogeneous element-wise ``auto``.

    ``backend='auto'`` is intentionally *not* converted into one whole-model
    native/PyTorch decision.  Every backend-aware brick resolves its own route
    (ESA, VESA scan, Bolt, vision scan/norm, FFNBrick, ResController,
    ElasticBit, ...), and that brick freezes the decision for steady-state
    execution.

    ``native`` and ``pytorch`` candidates may still be timed as diagnostics,
    but their whole-model timing can never override the element-wise operator
    route.  Explicit user requests for ``backend='native'`` or
    ``backend='pytorch'`` remain strict and global.
    """
    if bool(getattr(module, "training", False)):
        raise RuntimeError(
            "prepare_execution requires model.eval(); training keeps operator-safe auto routing"
        )
    kwargs = {} if sample_kwargs is None else dict(sample_kwargs)
    key_tensor = _first_tensor(sample_args)
    if key_tensor is None:
        key_tensor = _first_tensor(kwargs)
    if key_tensor is None:
        raise TypeError("prepare_execution needs at least one Tensor sample input")

    requested = getattr(module, "_mlbricks_requested_backend", None)
    if requested is None:
        requested = getattr(module, "backend", None)
    if requested is None:
        requested = getattr(getattr(module, "config", None), "backend", "auto")
    requested = normalize_backend(requested)
    if requested != "auto":
        apply_execution_route(module, requested)
        setattr(module, "_mlbricks_requested_backend", requested)
        setattr(module, "_mlbricks_model_route_reason", f"explicit:{requested}")
        return build_execution_plan(module)

    from .planner import EXECUTION_PLANNER
    model_key = EXECUTION_PLANNER.model_key(module, key_tensor, training=False)
    if not force and model_key in EXECUTION_PLANNER.model_benchmarks:
        cached_timings = dict(EXECUTION_PLANNER.model_benchmarks[model_key])
        apply_execution_route(module, "operator")
        setattr(module, "_mlbricks_requested_backend", "auto")
        setattr(module, "_mlbricks_model_route", "operator")
        setattr(module, "_mlbricks_model_route_reason", "elementwise:auto:cached-diagnostics")
        setattr(module, "_mlbricks_model_timings", cached_timings)
        setattr(module, "_mlbricks_model_errors", {})
        setattr(
            module, "_mlbricks_model_diagnostic_winner",
            min(cached_timings, key=cached_timings.get) if cached_timings else None,
        )
        return build_execution_plan(module)

    names: list[str] = []
    for name in candidates:
        value = str(name).strip().lower()
        if value in {"operator", "native", "pytorch"} and value not in names:
            names.append(value)
    if "operator" not in names:
        # Auto always needs the heterogeneous candidate because it is the route
        # that will actually be kept after preparation.
        names.insert(0, "operator")

    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    with torch.no_grad():
        for name in names:
            try:
                apply_execution_route(module, name)
                call = lambda: module(*sample_args, **kwargs)
                timings[name] = _median_call_ms(
                    call, key_tensor, warmup=warmup, trials=trials
                )
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"

    # Keep whole-model timings as diagnostics/cache compatibility, but never
    # allow their winner to become the active auto route.
    diagnostic_winner = None
    if timings:
        diagnostic_winner = EXECUTION_PLANNER.record_model_benchmark(
            module, key_tensor, timings, training=False
        )
        model_key = EXECUTION_PLANNER.model_key(module, key_tensor, training=False)
        EXECUTION_PLANNER.model_cache[model_key] = "operator"
        EXECUTION_PLANNER.model_reasons[model_key] = (
            f"elementwise:auto;diagnostic-winner:{diagnostic_winner}"
        )

    # Critical policy: auto always returns to heterogeneous per-element routing.
    # The operator pass above has already allowed each element to resolve/freeze
    # its route using its own workload. Whole-model native/PyTorch results are
    # retained only as diagnostics and never become the active route.
    apply_execution_route(module, "operator")
    setattr(module, "_mlbricks_requested_backend", "auto")
    setattr(module, "_mlbricks_model_route", "operator")
    setattr(module, "_mlbricks_model_route_reason", "elementwise:auto")
    setattr(module, "_mlbricks_model_timings", dict(timings) if timings else None)
    setattr(module, "_mlbricks_model_errors", errors)
    setattr(
        module,
        "_mlbricks_model_diagnostic_winner",
        diagnostic_winner,
    )
    return build_execution_plan(module)


# ---------------------------------------------------------------------------
# One-call inference convenience
# ---------------------------------------------------------------------------

def _tree_map_tensors(value: Any, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [_tree_map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: _tree_map_tensors(v, fn) for k, v in value.items()}
    return value


def _infer_predict_device(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], device: Any = "auto") -> torch.device:
    if device not in (None, "auto"):
        return torch.device(device)
    sample = _first_tensor(args)
    if sample is None:
        sample = _first_tensor(kwargs)
    # Respect an already-CUDA input first. This makes predict() natural in
    # larger GPU pipelines and avoids an unnecessary device hop.
    if sample is not None and sample.is_cuda:
        return sample.device
    # If the model was already placed explicitly, respect that placement.
    try:
        param = next(module.parameters())
        if param.is_cuda:
            return param.device
    except Exception:
        pass
    # Otherwise select the best generally available execution device.
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _infer_predict_dtype(device: torch.device, dtype: Any = "auto") -> torch.dtype:
    if dtype not in (None, "auto"):
        if isinstance(dtype, torch.dtype):
            return dtype
        name = str(dtype).strip().lower().replace("torch.", "")
        aliases = {
            "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32, "float": torch.float32,
        }
        if name not in aliases:
            raise ValueError(f"unsupported predict dtype {dtype!r}; use auto/fp16/bf16/fp32")
        return aliases[name]
    if device.type == "cuda":
        # FP16 is supported by every CUDA architecture that MLBricks currently
        # targets and matches the validated T4 path. Users can explicitly pick
        # BF16 on Ampere+ when preferred.
        return torch.float16
    return torch.float32


def predict_module(
    module: Any,
    *args: Any,
    device: Any = "auto",
    dtype: Any = "auto",
    calibrate: bool = True,
    calibration_warmup: int = 1,
    calibration_trials: int = 3,
    candidates: tuple[str, ...] = ("operator", "native", "pytorch"),
    **kwargs: Any,
):
    """Run one-call optimized inference without manual ``cuda/half/eval`` setup.

    ``predict`` is the simple MLBricks inference path:

    * choose CUDA when available (or respect an already-CUDA input/model);
    * choose FP16 on CUDA and FP32 on CPU unless explicitly overridden;
    * move floating inputs and model weights to the selected runtime;
    * switch the module to eval mode;
    * optionally prepare/freeze element-wise auto routes on the first matching
      workload without forcing one backend across the whole model;
    * execute under ``torch.inference_mode``.

    Integer/token/timestep tensors keep their dtype while moving device.
    Model parameters are never changed numerically beyond the requested dtype
    cast, and no training step/weight update is performed.
    """
    runtime_device = _infer_predict_device(module, args, kwargs, device=device)
    runtime_dtype = _infer_predict_dtype(runtime_device, dtype=dtype)

    # Place the model once. PyTorch keeps integer buffers in their integer dtype.
    module.to(device=runtime_device, dtype=runtime_dtype)
    module.eval()

    def move_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() or t.is_complex():
            return t.to(device=runtime_device, dtype=runtime_dtype, non_blocking=True)
        return t.to(device=runtime_device, non_blocking=True)

    moved_args = _tree_map_tensors(args, move_tensor)
    moved_kwargs = _tree_map_tensors(kwargs, move_tensor)

    requested = getattr(module, "_mlbricks_requested_backend", None)
    if requested is None:
        requested = getattr(module, "backend", getattr(getattr(module, "config", None), "backend", "auto"))
    requested = normalize_backend(requested)

    # Keep normal forward() free of hidden whole-model route switching.
    # predict() may prepare the element-wise route once; every brick then keeps
    # its own frozen native/PyTorch decision.
    if calibrate and requested == "auto":
        key_tensor = _first_tensor(moved_args)
        if key_tensor is None:
            key_tensor = _first_tensor(moved_kwargs)
        if key_tensor is not None:
            try:
                prepare_module_execution(
                    module, *moved_args, sample_kwargs=moved_kwargs,
                    warmup=max(0, int(calibration_warmup)),
                    trials=max(1, int(calibration_trials)),
                    candidates=tuple(candidates), force=False,
                )
            except RuntimeError as exc:
                # Unsupported candidate mixes must never make the simple predict
                # API unusable; the operator planner remains a safe fallback.
                if "no execution candidate succeeded" in str(exc):
                    apply_execution_route(module, "operator")
                else:
                    raise

    with torch.inference_mode():
        return module(*moved_args, **moved_kwargs)

# ---------------------------------------------------------------------------
# Model-level execution planning / compilation
# ---------------------------------------------------------------------------
from dataclasses import dataclass, asdict


@dataclass
class ExecutionPlan:
    """Resolved high-level execution plan for an MLBricks model.

    Native components and PyTorch components may safely share a device: CUDA
    tensors stay CUDA tensors across the boundary.  ``device_transfers`` is
    therefore expected to remain zero unless user code explicitly moves data.
    """

    device: str
    dtype: str
    training: bool
    requested_backend: str
    native_components: int
    pytorch_components: int
    dynamic_components: int
    device_transfers: int = 0
    compiled: bool = False
    compile_mode: str | None = None
    fullgraph: bool = False
    planner_routes: int = 0
    benchmarked_routes: int = 0
    model_route: str = "operator"
    model_route_reason: str | None = None
    model_benchmarked: bool = False
    model_benchmarked_routes: int = 0
    diagnostic_model_winner: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_execution_plan(module: Any) -> ExecutionPlan:
    param = None
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            param = next(parameters())
        except StopIteration:
            param = None
    device = str(param.device) if param is not None else "cpu"
    dtype = str(param.dtype).replace("torch.", "") if param is not None else "unknown"
    rows = backend_report(module)
    native = pytorch = dynamic = 0
    for row in rows:
        resolved = row["resolved"].lower()
        if resolved.startswith("native"):
            native += 1
        elif resolved.startswith("pytorch"):
            pytorch += 1
        else:
            dynamic += 1
    try:
        from .planner import EXECUTION_PLANNER
        planner_routes = len(EXECUTION_PLANNER.operator_cache)
        benchmarked_routes = len(EXECUTION_PLANNER.operator_benchmarks)
        model_benchmarked_routes = len(EXECUTION_PLANNER.model_benchmarks)
    except Exception:
        planner_routes = 0
        benchmarked_routes = 0
        model_benchmarked_routes = 0
    return ExecutionPlan(
        device=device,
        dtype=dtype,
        training=bool(getattr(module, "training", False)),
        requested_backend=str(getattr(module, "_mlbricks_requested_backend", getattr(module, "backend", getattr(getattr(module, "config", None), "backend", "auto")))),
        native_components=native,
        pytorch_components=pytorch,
        dynamic_components=dynamic,
        device_transfers=0,
        planner_routes=planner_routes,
        benchmarked_routes=benchmarked_routes,
        model_route=str(getattr(module, "_mlbricks_model_route", "operator")),
        model_route_reason=getattr(module, "_mlbricks_model_route_reason", None),
        model_benchmarked=getattr(module, "_mlbricks_model_timings", None) is not None,
        model_benchmarked_routes=model_benchmarked_routes,
        diagnostic_model_winner=getattr(module, "_mlbricks_model_diagnostic_winner", None),
    )


def compile_module(
    module: Any,
    *,
    mode: str = "default",
    dynamic: bool | None = None,
    fullgraph: bool = False,
    strict: bool = False,
) -> ExecutionPlan:
    """Compile graphable PyTorch regions while preserving native op boundaries.

    ``fullgraph=False`` is the safe default for heterogeneous MLBricks models:
    dispatcher-registered custom ops remain visible to Dynamo/AOTAutograd, and
    any non-traceable third-party/custom Python component becomes a graph break
    instead of making the model unusable.
    """
    plan = build_execution_plan(module)
    try:
        # Call the base implementation explicitly so MLBricks models may expose
        # their own chainable ``compile`` convenience method.
        import torch
        torch.nn.Module.compile(
            module,
            backend="inductor",
            mode=mode,
            fullgraph=bool(fullgraph),
            dynamic=dynamic,
        )
        plan.compiled = True
        plan.compile_mode = str(mode)
        plan.fullgraph = bool(fullgraph)
    except Exception:
        if strict:
            raise
        plan.compiled = False
        plan.compile_mode = str(mode)
        plan.fullgraph = bool(fullgraph)
    setattr(module, "_mlbricks_execution_plan", plan)
    return plan
