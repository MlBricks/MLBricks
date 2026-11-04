# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F



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


def _unwrap_backend(module):
    seen = set()
    while hasattr(module, "_orig_mod") and id(module) not in seen:
        seen.add(id(module))
        module = module._orig_mod
    return module


def _fast_esa_module(module):
    module = _unwrap_backend(module)
    if not callable(getattr(module, "prefill", None)):
        raise TypeError("Expected a Fast-EsaQ ESA layer")
    if not callable(getattr(module, "decode_step", None)):
        raise TypeError("Fast-EsaQ decode_step is required")
    return module


def lightning_init_state(module, batch, *, device=None, dtype=None, layout="heads"):
    layer = _fast_esa_module(module)

    if hasattr(layer, "init_state"):
        return layer.init_state(
            batch, device=device, dtype=dtype, layout=layout
        )

    if layout not in {"heads", "flat"} or not isinstance(batch, int) or batch < 1:
        raise ValueError("Invalid batch or layout")

    ref = next(layer.parameters())
    shape = (
        (batch, layer.heads, layer.dim // layer.heads)
        if layout == "heads" else (batch, layer.dim)
    )
    return torch.zeros(
        shape,
        device=device or ref.device,
        dtype=torch.float32 if dtype is None else dtype,
    )


@torch.no_grad()
def esa_prefill(module, x, state=None, *, backend=None, compass=None):
    layer = _fast_esa_module(module)
    if x.ndim != 3 or x.size(1) == 0:
        raise ValueError("Expected non-empty [B,T,C] input")
    if hasattr(layer, "forward_with_state"):
        return layer.prefill(
            x, state, backend=backend, compass=compass
        )
    if backend is not None or compass is not None:
        raise ValueError("Overrides require an MLBricks ESA wrapper")
    return layer.prefill(x, state)


def esa_forward_with_state(
    module, x, state=None, *, backend=None,
    compass=None, reverse=False
):
    layer = _fast_esa_module(module)
    if x.ndim != 3 or x.size(1) == 0:
        raise ValueError("Expected non-empty [B,T,C] input")
    if hasattr(layer, "forward_with_state"):
        return layer.forward_with_state(
            x, state, backend=backend,
            compass=compass, reverse=reverse
        )
    if backend is not None or compass is not None:
        raise ValueError("Overrides require an MLBricks ESA wrapper")
    y, last = layer(x.flip(1) if reverse else x, state)
    return (y.flip(1) if reverse else y), last


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


def lightning_decode_step(module, x, state):
    layer = _fast_esa_module(module)
    if x.ndim not in (2, 3) or (x.ndim == 3 and x.size(1) != 1):
        raise ValueError("Expected [B,C] or [B,1,C] input")
    output, next_state = layer.decode_step(x, state)
    if x.ndim == 2 and output.ndim == 3:
        output = output[:, 0]
    return output, next_state


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
