"""Optional native C++/CUDA backend for ResidualBrick.

The original PyTorch equations remain the training and ``torch.compile``
reference path. The native backend is deliberately an eager CUDA inference
specialization.
"""
from __future__ import annotations

import torch

from ..planner import EXECUTION_PLANNER
from ..runtime import normalize_backend

try:
    from . import _C  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def is_available() -> bool:
    return _C is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def backend_name() -> str:
    if _C is None:
        return "python"
    return "cpp/cuda" if torch.version.cuda is not None else "cpp"


def is_compiling() -> bool:
    try:
        compiler = getattr(torch, "compiler", None)
        fn = getattr(compiler, "is_compiling", None)
        if fn is not None and fn():
            return True
    except Exception:
        pass
    try:
        import torch._dynamo as dynamo
        return bool(dynamo.is_compiling())
    except Exception:
        return False


def inference_native_eligible(
    module: object,
    residual: torch.Tensor,
    update: torch.Tensor,
) -> bool:
    """Whether the native residual kernel can legally execute this element."""
    return bool(
        getattr(module, "use_native", False)
        and _C is not None
        and not torch.is_grad_enabled()
        and not is_compiling()
        and residual.is_cuda
        and update.is_cuda
        and residual.dtype == update.dtype
    )


def inference_native_allowed(
    module: object,
    residual: torch.Tensor,
    update: torch.Tensor,
) -> bool:
    """Use the shared planner for eager no-grad ResidualBrick inference."""
    eligible = inference_native_eligible(module, residual, update)
    policy = normalize_backend(getattr(module, "backend", "auto"))
    if policy == "pytorch":
        return False
    if policy == "native":
        return eligible
    if not eligible:
        return False
    route = EXECUTION_PLANNER.select_operator_once(module, 
        "rescontroller", residual, requested_backend="auto",
        native_available=True, native_supports_training=False, training=False,
        extra=(int(residual.shape[-1]),),
    )
    return route == "native"


def residual_forward(
    residual: torch.Tensor,
    update: torch.Tensor,
    *,
    update_ratio: float,
    stream_ratio: float,
    update_softness: float,
    stream_softness: float,
    eps: float,
    fused_cuda: bool,
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("ResidualBrick native extension is not loaded") from _IMPORT_ERROR

    return _C.residual_forward(
        residual,
        update,
        float(update_ratio),
        float(stream_ratio),
        float(update_softness),
        float(stream_softness),
        float(eps),
        bool(fused_cuda),
    )
