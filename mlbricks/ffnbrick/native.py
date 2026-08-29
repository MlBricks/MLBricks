"""Optional native C++/CUDA backend for FFNBrick.

The original FFNBrick equations live in the Python modules and remain the
training/``torch.compile`` reference implementation.  The native backend is an
*inference specialization*: it packs projection weights once, removes repeated
physical-depth projections, and uses small fused CUDA kernels for the recurrent
pointwise stages.

Why this split matters
----------------------
The first native FFNBrick port merely moved the Python orchestration into a
pybind11 function while still launching every ``at::linear`` independently.
That made the call opaque to TorchDynamo and kept 8--11 tiny GEMM launches per
layer, which is particularly expensive for one-token decode.  The optimized
path below therefore:

* keeps the exact PyTorch graph for training and while Dynamo is compiling;
* packs the three ``x`` projections and two recurrent-state projections;
* caches learned depth projections and scalar transition transforms in eval;
* lets CUDA fuse ESA transition/delta statistics plus state/read elementwise
  work; and
* packs MicroVirtualFFN gate/up projections, consumes the packed activation
  without gate/value copy kernels, fuses residual accumulation into GEMM, and
  can run all refinement passes in one native call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from ..planner import EXECUTION_PLANNER
from ..runtime import normalize_backend

try:
    from . import _C  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - depends on local compiler/build
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

if TYPE_CHECKING:
    from .micro_virtual import MicroVirtualFFN
    from .state_aware import StateAwareFFN
    from .state_aware_virtual import VirtualStateAwareFFN


def is_available() -> bool:
    return _C is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def backend_name() -> str:
    if _C is None:
        return "python"
    return "cpp/cuda" if torch.version.cuda is not None else "cpp"


def is_compiling() -> bool:
    """True while TorchDynamo/torch.compile is tracing the module."""
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


def inference_native_allowed(module: object, x: torch.Tensor) -> bool:
    """Planner-controlled eager inference specialization.

    FFNBrick native calls are intentionally inference-only. Training and
    ``torch.compile`` retain the PyTorch graph. Explicit ``backend='native'``
    remains strict; ``auto`` lets the shared MLBricks planner choose.
    """
    eligible = bool(
        getattr(module, "use_native", False)
        and _C is not None
        and not torch.is_grad_enabled()
        and not is_compiling()
        and x.device.type in {"cpu", "cuda"}
    )
    policy = normalize_backend(getattr(module, "backend", "auto"))
    if policy == "pytorch":
        return False
    if policy == "native":
        return eligible
    if not eligible:
        return False

    name = module.__class__.__name__.lower()
    if "micro" in name:
        op = "ffnbrick_micro"
    elif "virtual" in name:
        op = "ffnbrick_virtual"
    else:
        op = "ffnbrick_state"
    route = EXECUTION_PLANNER.select_operator_once(module, 
        op, x, requested_backend="auto", native_available=True,
        native_supports_training=False, training=False,
        extra=(int(getattr(module, "d_model", x.shape[-1])),
               int(getattr(module, "state_dim", 0)),
               int(getattr(module, "refinements", 0))),
    )
    return route == "native"


def _tensor_key(t: torch.Tensor) -> tuple[int, int, str, str, tuple[int, ...]]:
    return (id(t), int(getattr(t, "_version", 0)), str(t.device), str(t.dtype), tuple(t.shape))


def _state_source_tensors(module: "StateAwareFFN") -> tuple[torch.Tensor, ...]:
    return (
        module.x_candidate.weight,
        module.x_candidate.bias,
        module.x_write.weight,
        module.x_write.bias,
        module.value.weight,
        module.value.bias,
        module.state_candidate.weight,
        module.state_write.weight,
        module.esa_candidate.weight,
        module.esa_write.weight,
        module.output.weight,
        module.output.bias,
        module.depth_embedding,
        module.depth_to_candidate.weight,
        module.depth_to_write.weight,
        module.depth_to_value.weight,
        module.retain_logit,
        module.read_logit,
        module.retain_delta_scale,
        module.read_delta_scale,
        module.candidate_transition_logit,
        module.write_transition_logit,
        module.delta_magnitude_log_scale,
    )


def _state_packed_params(module: "StateAwareFFN") -> tuple[torch.Tensor, ...]:
    """Create/cached inference-only packed tensors.

    Order must match ``state_aware_forward_packed`` in ``csrc/ffnbrick.cpp``.
    The cache is automatically rebuilt after an optimizer/in-place update,
    ``load_state_dict``, or device/dtype conversion because all source tensor
    versions/device/dtypes are included in the key.
    """
    src = _state_source_tensors(module)
    key = tuple(_tensor_key(t) for t in src)
    cache = getattr(module, "_ffnbrick_native_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]

    with torch.no_grad():
        x_weight = torch.cat(
            [module.x_candidate.weight, module.x_write.weight, module.value.weight],
            dim=0,
        ).contiguous()
        x_bias = torch.cat(
            [module.x_candidate.bias, module.x_write.bias, module.value.bias],
            dim=0,
        ).contiguous()
        state_weight = torch.cat(
            [module.state_candidate.weight, module.state_write.weight],
            dim=0,
        ).contiguous()

        # Depth is constant for a physical layer during inference.  The old
        # native port recalculated these three GEMVs for every token.
        depth_candidate = module.depth_to_candidate(module.depth_embedding).contiguous()
        depth_write = module.depth_to_write(module.depth_embedding).contiguous()
        depth_value = module.depth_to_value(module.depth_embedding).contiguous()

        # Scalar transforms are also invariant until the parameter changes.
        candidate_transition = torch.sigmoid(module.candidate_transition_logit).contiguous()
        write_transition = torch.sigmoid(module.write_transition_logit).contiguous()
        delta_scale = torch.exp(module.delta_magnitude_log_scale).contiguous()

        packed = (
            x_weight,
            x_bias,
            state_weight,
            module.esa_candidate.weight,
            module.esa_write.weight,
            module.output.weight,
            module.output.bias,
            depth_candidate,
            depth_write,
            depth_value,
            module.retain_logit,
            module.read_logit,
            module.retain_delta_scale,
            module.read_delta_scale,
            candidate_transition,
            write_transition,
            delta_scale,
        )

    module._ffnbrick_native_cache = (key, packed)
    return packed


def clear_state_cache(module: "StateAwareFFN") -> None:
    module._ffnbrick_native_cache = None


def state_aware_forward(
    module: "StateAwareFFN",
    x: torch.Tensor,
    esa_update: torch.Tensor,
    previous_esa: torch.Tensor,
    previous_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise RuntimeError("FFNBrick native extension is not loaded") from _IMPORT_ERROR
    out = _C.state_aware_forward_packed(
        x,
        esa_update,
        previous_esa,
        previous_state,
        _state_packed_params(module),
        bool(module.fused_cuda),
    )
    return out[0], out[1]


def _micro_source_tensors(module: "MicroVirtualFFN") -> tuple[torch.Tensor, ...]:
    return (module.gate, module.up, module.down)


def _micro_packed_params(module: "MicroVirtualFFN") -> tuple[torch.Tensor, torch.Tensor]:
    src = _micro_source_tensors(module)
    key = tuple(_tensor_key(t) for t in src)
    cache = getattr(module, "_ffnbrick_native_cache", None)
    if cache is not None and cache[0] == key:
        return cache[1]
    with torch.no_grad():
        # [R,H,D] + [R,H,D] -> [R,2H,D], so one GEMM produces gate+value.
        gate_up = torch.cat([module.gate, module.up], dim=1).contiguous()
        down = module.down.contiguous()
        packed = (gate_up, down)
    module._ffnbrick_native_cache = (key, packed)
    return packed


def clear_micro_cache(module: "MicroVirtualFFN") -> None:
    module._ffnbrick_native_cache = None


def micro_virtual_forward(
    module: "MicroVirtualFFN", x: torch.Tensor, refinement_index: int
) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("FFNBrick native extension is not loaded") from _IMPORT_ERROR
    gate_up, down = _micro_packed_params(module)
    return _C.micro_virtual_forward_packed(
        x,
        gate_up[refinement_index],
        down[refinement_index],
        bool(module.fused_cuda),
    )


def micro_virtual_refine(module: "MicroVirtualFFN", x: torch.Tensor) -> torch.Tensor:
    if _C is None:
        raise RuntimeError("FFNBrick native extension is not loaded") from _IMPORT_ERROR
    gate_up, down = _micro_packed_params(module)
    return _C.micro_virtual_refine_packed(
        x,
        gate_up,
        down,
        int(module.refinements),
        bool(module.fused_cuda),
    )


def virtual_state_aware_forward(
    module: "VirtualStateAwareFFN",
    x: torch.Tensor,
    esa_update: torch.Tensor,
    previous_esa: torch.Tensor,
    previous_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _C is None:
        raise RuntimeError("FFNBrick native extension is not loaded") from _IMPORT_ERROR

    vr = module.virtual_refiner
    virtual_params = (
        vr.norm.weight,
        vr.state_up.weight,
        vr.state_up.bias,
        vr.x_condition.weight,
        vr.esa_condition.weight,
        vr.down.weight,
        vr.down.bias,
        vr.pass_embedding,
        vr.gate_logit,
    )
    out = _C.virtual_state_aware_forward_packed(
        x,
        esa_update,
        previous_esa,
        previous_state,
        _state_packed_params(module),
        virtual_params,
        int(vr.refinements),
        float(vr.norm.eps),
        bool(module.fused_cuda),
    )
    return out[0], out[1]
