# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .precision import resolve_scan_dtype


@dataclass(frozen=True)
class GenerationStats:
    prompt_tokens: int
    prefill_tokens: int
    generated_tokens: int
    decode_steps: int
    prefill_seconds: float
    decode_seconds: float
    decode_tok_s: float
    total_seconds: float
    state_bytes: int
    state_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Return generation statistics as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    generated_ids: torch.Tensor
    stats: GenerationStats
    text: str | list[str] | None = None


@dataclass(frozen=True)
class EngineSpec:
    """Parsed ESA execution-engine specification."""

    backend: str
    compiled: bool = False
    compass: int | str | None = None


def parse_engine_spec(
    value: str,
    *,
    default_thunder_compass: int | str = "auto",
) -> EngineSpec:
    """Parse Thunder prefill and Lightning recurrent-engine names.

    ``thunder`` uses automatic Compass planning. Append a positive integer for
    a manual setting, for example ``thunder_16``. Legacy prefill names such as
    ``thunder_compiled_32`` remain parseable for compatibility, but model
    prefill always executes Thunder directly; the ``compiled`` flag is only
    operationally relevant to Lightning runtime selection.
    """
    name = str(value).strip().lower().replace("-", "_")
    if not name:
        raise ValueError("Engine name must not be empty.")

    if name in {"lightning", "lightning_compiled"}:
        return EngineSpec(
            backend="lightning",
            compiled=name.endswith("_compiled"),
            compass=None,
        )

    parts = [part for part in name.split("_") if part]
    if parts[0] != "thunder":
        raise ValueError(
            f"Unknown ESA engine {value!r}. Supported engines are lightning and thunder."
        )

    if "auto" in parts[1:]:
        raise ValueError(
            "Automatic Compass is already the default for engine='thunder'; "
            "'thunder_auto' is not a public engine name."
        )
    unknown = [p for p in parts[1:] if p != "compiled" and not p.isdigit()]
    if unknown:
        raise ValueError(f"Unknown Thunder engine option(s) {unknown!r} in {value!r}.")

    compiled = "compiled" in parts[1:]
    numeric = [int(part) for part in parts[1:] if part.isdigit()]
    if len(numeric) > 1:
        raise ValueError(f"Thunder engine has multiple compass values: {value!r}")

    if numeric:
        compass: int | str = numeric[0]
    elif isinstance(default_thunder_compass, str):
        compass = default_thunder_compass.strip().lower()
    else:
        compass = int(default_thunder_compass)

    if isinstance(compass, int) and compass <= 0:
        raise ValueError(f"Thunder compass must be positive, got {compass}.")
    if isinstance(compass, str) and compass != "auto":
        raise ValueError("Thunder compass must be a positive integer or 'auto'.")

    return EngineSpec(backend="thunder", compiled=compiled, compass=compass)


def _unwrap_backend(module: torch.nn.Module) -> torch.nn.Module:
    current = getattr(module, "layer", module)
    seen: set[int] = set()

    while hasattr(current, "_orig_mod") and id(current) not in seen:
        seen.add(id(current))
        current = current._orig_mod

    return current


def _backend_name(module: torch.nn.Module) -> str:
    if hasattr(module, "backend"):
        return str(module.backend).lower()
    name = _unwrap_backend(module).__class__.__name__.lower()
    return "thunder" if "thunder" in name else name


def _dimensions(
    module: torch.nn.Module,
) -> tuple[int, int, int]:
    backend = _unwrap_backend(module)

    embd = getattr(backend, "embd", None)
    head = getattr(backend, "head", None)
    head_dim = getattr(backend, "head_dim", None)

    if embd is None or head is None:
        raise AttributeError(
            "ESA backend must expose embd and head."
        )

    embd = int(embd)
    head = int(head)

    if head_dim is None:
        head_dim = embd // head

    return embd, head, int(head_dim)


def _state_dtype(
    module: torch.nn.Module,
    device: torch.device,
    input_dtype: torch.dtype,
    *,
    backend_name: str | None = None,
) -> torch.dtype:
    backend = _unwrap_backend(module)
    precision = str(
        getattr(
            backend,
            "precision",
            getattr(module, "precision", "fp16"),
        )
    )

    return resolve_scan_dtype(
        precision,
        device,
        strict_precision=bool(
            getattr(
                backend,
                "strict_precision",
                False,
            )
        ),
    )


def _project_affine_terms(
    module: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backend = _unwrap_backend(module)
    embd, head, head_dim = _dimensions(module)

    if x.ndim != 3 or x.size(-1) != embd:
        raise ValueError(
            f"Expected x [B,T,{embd}], got {tuple(x.shape)}"
        )

    B, T, C = x.shape

    q, gate_raw, value_raw = backend.qgv(x).split(
        C,
        dim=-1,
    )

    q = q.reshape(B, T, head, head_dim)
    gate_raw = gate_raw.reshape(
        B,
        T,
        head,
        head_dim,
    )
    value_raw = value_raw.reshape(
        B,
        T,
        head,
        head_dim,
    )

    gate = torch.sigmoid(gate_raw)
    A = backend.gate_min + (
        backend.gate_max - backend.gate_min
    ) * gate

    V = torch.tanh(value_raw)
    B_write = (1.0 - A) * V

    return q, A, B_write


def _backend_scan(
    module: torch.nn.Module,
    A: torch.Tensor,
    B_write: torch.Tensor,
    *,
    backend_override: str | None = None,
    compass_override: int | str | None = None,
) -> torch.Tensor:
    backend = _unwrap_backend(module)
    from ..runtime import normalize_backend
    raw_name = backend_override or _backend_name(module)
    name = normalize_backend(raw_name, warn_legacy=False)

    dtype = _state_dtype(module, A.device, A.dtype, backend_name=name)
    A = A.to(dtype).contiguous()
    B_write = B_write.to(dtype).contiguous()

    raw_compass = (
        compass_override
        if compass_override is not None
        else getattr(backend, "compass", "auto")
    )
    if raw_compass is None:
        raw_compass = "auto"
    compass = raw_compass.strip().lower() if isinstance(raw_compass, str) else int(raw_compass)

    if compass == "auto":
        from .backends.thunder import _thunder_scan_auto
        return _thunder_scan_auto(A, B_write, backend=name)

    from .backends.thunder import thunder_scan
    return thunder_scan(A, B_write, compass=int(compass), backend=name)

def _readout(
    module: torch.nn.Module,
    q: torch.Tensor,
    states: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    backend = _unwrap_backend(module)

    B, T, H, D = states.shape
    C = H * D

    E = states.reshape(B, T, C)
    q = q.reshape(B, T, C).to(E.dtype)

    E = E * torch.rsqrt(
        E.pow(2).mean(dim=-1, keepdim=True)
        + float(backend.eps)
    )

    y = torch.sigmoid(q) * E
    y = backend.out_proj(y.to(output_dtype))

    return backend.dropout(y)


def lightning_init_state(
    module: torch.nn.Module,
    batch: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    layout: str = "heads",
) -> torch.Tensor:
    backend = _unwrap_backend(module)
    embd, head, head_dim = _dimensions(module)

    qgv = backend.qgv
    reference_device = None
    reference_dtype = None
    for parameter in qgv.parameters():
        reference_device = parameter.device
        if parameter.is_floating_point():
            reference_dtype = parameter.dtype
        break
    if reference_device is None:
        for buffer in qgv.buffers():
            reference_device = buffer.device
            if buffer.is_floating_point() and reference_dtype is None:
                reference_dtype = buffer.dtype
    if reference_device is None:
        reference_device = torch.device("cpu")
    if reference_dtype is None:
        reference_dtype = torch.get_default_dtype()

    if device is None:
        device = reference_device

    device = torch.device(device)

    if dtype is None:
        dtype = _state_dtype(
            module,
            device,
            reference_dtype,
        )

    if layout == "heads":
        shape = (batch, head, head_dim)
    elif layout == "flat":
        shape = (batch, embd)
    else:
        raise ValueError(
            "layout must be 'heads' or 'flat'"
        )

    return torch.zeros(
        shape,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def esa_prefill(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    compass: int | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ESA prompt prefill with an optional execution-backend override.

    ``backend=None`` preserves the model's configured backend.
    ``backend='lightning'`` performs the exact recurrent scan directly.
    """
    if x.ndim != 3 or x.size(1) <= 0:
        raise ValueError(
            f"Prefill expects non-empty [B,T,C], got {tuple(x.shape)}"
        )

    B = x.size(0)
    _, head, head_dim = _dimensions(module)
    q, A, B_write = _project_affine_terms(module, x)
    requested = None if backend is None else str(backend).lower()

    if state is None and requested != "lightning":
        states = _backend_scan(
            module,
            A,
            B_write,
            backend_override=requested,
            compass_override=compass,
        )
    else:
        if state is None:
            state_dtype = _state_dtype(
                module,
                A.device,
                A.dtype,
                backend_name=_backend_name(module),
            )
            current = torch.zeros(
                (B, head, head_dim),
                device=A.device,
                dtype=state_dtype,
            )
        else:
            current = (
                state.reshape(B, head, head_dim)
                if state.ndim == 2
                else state
            )

        outputs = []
        for t in range(x.size(1)):
            current = (
                A[:, t].to(current.dtype) * current
                + B_write[:, t].to(current.dtype)
            )
            outputs.append(current)
        states = torch.stack(outputs, dim=1)

    y = _readout(
        module,
        q,
        states,
        output_dtype=x.dtype,
    )
    return y, states[:, -1].contiguous()


def esa_forward_with_state(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    compass: int | str | None = None,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable canonical ESA sequence pass with recurrent state.

    This is the compositional-model interface for wrappers such as VESA. It
    uses the same Q/G/V projection, bounded retention gate, Thunder scan,
    RMS-normalized query readout, backend planner, and learned projections as
    :class:`mlbricks.esa.ESA` while additionally returning the final recurrent
    state.  ``state`` may use either ``[B,H,D]`` or flattened ``[B,C]`` layout.

    ``reverse=True`` scans the sequence in the opposite direction and restores
    outputs to the original token order.  Unlike :func:`esa_prefill`, this
    function intentionally keeps autograd enabled and is suitable for training.
    """
    if x.ndim != 3 or x.size(1) <= 0:
        raise ValueError(
            f"ESA sequence expects non-empty [B,T,C], got {tuple(x.shape)}"
        )

    work = x.flip(1) if reverse else x
    B = work.size(0)
    embd, head, head_dim = _dimensions(module)
    if work.size(-1) != embd:
        raise ValueError(f"Expected final dimension {embd}, got {work.size(-1)}")

    q, A, B_write = _project_affine_terms(module, work)
    requested = None if backend is None else str(backend).lower()

    # Use the canonical Thunder/native scan for the zero-state contribution.
    states = _backend_scan(
        module,
        A,
        B_write,
        backend_override=requested,
        compass_override=compass,
    )

    # Affine recurrence with a non-zero incoming state:
    # E_t(state0) = E_t(0) + prod_{i<=t}(A_i) * state0.
    # This retains the canonical optimized scan and adds the initial-state
    # contribution without a second recurrence implementation.
    if state is not None:
        if state.ndim == 2:
            if state.shape != (B, embd):
                raise ValueError(f"flattened state must have shape {(B, embd)}")
            state_h = state.reshape(B, head, head_dim)
        elif state.ndim == 3:
            if state.shape != (B, head, head_dim):
                raise ValueError(
                    f"headed state must have shape {(B, head, head_dim)}"
                )
            state_h = state
        else:
            raise ValueError("state must have shape [B,C] or [B,H,D]")

        prefix_A = torch.cumprod(A.to(states.dtype), dim=1)
        states = states + prefix_A * state_h.to(states.dtype).unsqueeze(1)

    y = _readout(module, q, states, output_dtype=work.dtype)
    final_state = states[:, -1].contiguous()
    if reverse:
        y = y.flip(1)
    return y, final_state


# Backward-compatible low-level name. With no override it keeps the model's
# configured backend, matching ESA v2.1 behavior.
lightning_prefill = esa_prefill


def _decode_runtime_backend(module: torch.nn.Module, x: torch.Tensor) -> str:
    """Resolve explicit/Auto policy for one-token decode.

    Historically decode_step opportunistically used the native kernel even
    when the module was configured with backend="pytorch".  Keep explicit
    policies strict and let only backend="auto" use the performance planner.
    """
    from ..runtime import normalize_backend

    backend = _unwrap_backend(module)
    raw = getattr(module, "backend", getattr(backend, "runtime_backend", "auto"))
    policy = normalize_backend(raw, warn_legacy=False)
    if policy != "auto":
        return policy

    try:
        from .auto_backend import select_esa_auto_backend
        from .native import available as _native_available
        from .native import cuda_available as _native_cuda_available
        return select_esa_auto_backend(
            x,
            workload="decode",
            training=bool(getattr(backend, "training", False) or torch.is_grad_enabled()),
            compile_mode=getattr(backend, "_mlbricks_compile_mode", None),
            native_available=bool(_native_available()),
            native_cuda_available=bool(_native_cuda_available()),
        )
    except Exception:
        return "auto"


def lightning_decode_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact one-token ESA-Lightning decode step.

    The default path intentionally mirrors the proven MLBricks 0.1.0
    Lightning implementation: project Q/G/V, update only the compact ESA
    recurrent state with the native one-token kernel when available, then
    perform the readout with ordinary tensor operations.  This shape is very
    friendly to ``torch.compile`` because the fixed one-token decode graph can
    fuse the surrounding work instead of crossing an additional fused custom
    operator boundary.

    Set ``MLBRICKS_LIGHTNING_FUSED=1`` to opt into the newer experimental
    fused QGV/state/readout CUDA operator.
    """
    if x.ndim == 2:
        x3 = x.unsqueeze(1)
        squeeze = True
    elif x.ndim == 3 and x.size(1) == 1:
        x3 = x
        squeeze = False
    else:
        raise ValueError(
            "decode_step expects [B,C] or [B,1,C], "
            f"got {tuple(x.shape)}"
        )

    B = x3.size(0)
    _, head, head_dim = _dimensions(module)
    backend = _unwrap_backend(module)
    runtime_policy = _decode_runtime_backend(module, x3)
    flat_state = state.ndim == 2
    state_h = state.reshape(B, head, head_dim) if flat_state else state

    # The all-in-one fused Lightning kernel is retained as an explicit opt-in
    # experiment, but it is not the production default because the older
    # decomposed fixed-shape graph benchmarks better under torch.compile.
    import os
    use_fused = os.getenv("MLBRICKS_LIGHTNING_FUSED", "0") == "1"
    if use_fused:
        qgv = backend.qgv(x3)
        try:
            from .native import fused_enabled_for as _fused_enabled_for
            from .native import lightning_fused_step as _native_fused_step
            if (
                runtime_policy != "pytorch"
                and _fused_enabled_for(qgv)
                and state_h.is_cuda
                and state_h.dtype == qgv.dtype
                and (not backend.training or backend.dropout.p == 0.0)
            ):
                readout, new_state_h = _native_fused_step(
                    qgv[:, 0],
                    state_h,
                    backend.gate_min,
                    backend.gate_max,
                    backend.eps,
                )
                y = backend.out_proj(readout.to(x3.dtype))
                y = backend.dropout(y)
                if not squeeze:
                    y = y.unsqueeze(1)
                new_state = (
                    new_state_h.reshape(B, -1).contiguous()
                    if flat_state
                    else new_state_h.contiguous()
                )
                return y, new_state
        except (ImportError, AttributeError, RuntimeError):
            if os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
                raise

    # Proven 0.1.0 Lightning path.
    q, A, B_write = _project_affine_terms(module, x3)

    try:
        from .native import enabled_for as _native_enabled_for
        from .native import lightning_step as _native_lightning_step
        native_ok = bool(_native_enabled_for(A))
        if runtime_policy == "native" and not native_ok:
            raise RuntimeError(
                "ESA backend='native' requested but native decode is unavailable"
            )
        if runtime_policy != "pytorch" and native_ok:
            new_state_h = _native_lightning_step(
                A[:, 0],
                B_write[:, 0],
                state_h,
            )
        else:
            new_state_h = (
                A[:, 0].to(state_h.dtype) * state_h
                + B_write[:, 0].to(state_h.dtype)
            )
    except (ImportError, AttributeError, RuntimeError):
        if runtime_policy == "native" or os.getenv("MLBRICKS_NATIVE_STRICT", "0") == "1":
            raise
        new_state_h = (
            A[:, 0].to(state_h.dtype) * state_h
            + B_write[:, 0].to(state_h.dtype)
        )

    y = _readout(
        module,
        q,
        new_state_h.unsqueeze(1),
        output_dtype=x3.dtype,
    )

    if squeeze:
        y = y[:, 0]

    new_state = (
        new_state_h.reshape(B, -1).contiguous()
        if flat_state
        else new_state_h.contiguous()
    )

    return y, new_state


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits[:, -1]

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits [B,V] or [B,T,V], got {tuple(logits.shape)}"
        )

    if temperature <= 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    logits = logits / max(
        float(temperature),
        1e-5,
    )

    if top_k is not None:
        k = min(
            int(top_k),
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            k,
        )

        logits = logits.masked_fill(
            logits < values[:, [-1]],
            float("-inf"),
        )

    if (
        top_p is not None
        and 0.0 < float(top_p) < 1.0
    ):
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )

        sorted_probs = F.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        remove = cumulative > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(
            remove,
            float("-inf"),
        )

        logits = torch.full_like(
            logits,
            float("-inf"),
        )

        logits.scatter_(
            1,
            sorted_indices,
            sorted_logits,
        )

    return torch.multinomial(
        F.softmax(
            logits,
            dim=-1,
        ),
        num_samples=1,
    )
