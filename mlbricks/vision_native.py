"""Shared C++/CUDA vision operators for VESA and VisionBolt.

MLBricks only custom-implements the vision-specific work that PyTorch does not
already provide efficiently: directional scan reorder/restore, deterministic
2-D sin/cos injection, patch-layout reconstruction, and the full-sequence Bolt
core. Dense/conv/norm/activation operations intentionally remain PyTorch ops;
on CUDA they already execute in C++/CUDA through cuBLAS/cuDNN/ATen.
"""
from __future__ import annotations

import importlib
from functools import lru_cache

import torch
from torch import Tensor

from .runtime import normalize_backend
from .planner import EXECUTION_PLANNER

_SCAN_CODES = {"cross": 0, "horizontal": 1, "vertical": 2, "raster": 3}
_REGISTERED = False


@lru_cache(maxsize=1)
def _load_extension():
    try:
        module = importlib.import_module("mlbricks._vision_native")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MLBricks shared vision native extension is not built. Install from source "
            "with `pip install -v .` (CUDA toolkit present for CUDA support), or use "
            "backend='pytorch'."
        ) from exc
    _register_helpers()
    return module


def _register_helpers() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    @torch.library.register_fake("mlbricks_vision_native::scan_reorder")
    def _fake_scan_reorder(x, height: int, width: int, scan_kind: int, phase: int, inverse: bool = False):
        del height, width, scan_kind, phase, inverse
        return torch.empty_like(x)

    @torch.library.register_fake("mlbricks_vision_native::add_sincos2d")
    def _fake_add_sincos2d(x, height: int, width: int):
        del height, width
        return torch.empty_like(x)

    @torch.library.register_fake("mlbricks_vision_native::sincos2d")
    def _fake_sincos2d(reference, height: int, width: int):
        return reference.new_empty((height * width, reference.shape[-1]))

    @torch.library.register_fake("mlbricks_vision_native::unpatchify")
    def _fake_unpatchify(patches, gh: int, gw: int, patch: int, channels: int):
        return patches.new_empty((patches.shape[0], channels, gh * patch, gw * patch))

    @torch.library.register_fake("mlbricks_vision_native::patchify_layout")
    def _fake_patchify_layout(image, patch: int):
        b, c, h, w = image.shape
        return image.new_empty((b, (h // patch) * (w // patch), patch * patch * c))

    @torch.library.register_fake("mlbricks_vision_native::bolt_full_fused")
    def _fake_bolt_full_fused(q, u, g, heads: int, latent_dim: int, head_dim: int, eps: float, causal: bool):
        del u, g, heads, latent_dim, head_dim, eps, causal
        return torch.empty_like(q)

    def scan_setup(ctx, inputs, output):
        del output
        _, height, width, scan_kind, phase, inverse = inputs
        ctx.params = (height, width, scan_kind, phase, inverse)

    def scan_backward(ctx, grad):
        height, width, scan_kind, phase, inverse = ctx.params
        if grad is None:
            return None, None, None, None, None, None
        gx = torch.ops.mlbricks_vision_native.scan_reorder(
            grad.contiguous(), height, width, scan_kind, phase, not inverse
        )
        return gx, None, None, None, None, None

    torch.library.register_autograd(
        "mlbricks_vision_native::scan_reorder", scan_backward, setup_context=scan_setup
    )

    def addpos_backward(ctx, grad):
        del ctx
        return grad, None, None

    torch.library.register_autograd(
        "mlbricks_vision_native::add_sincos2d", addpos_backward
    )

    def unpatch_setup(ctx, inputs, output):
        del output
        _, gh, gw, patch, channels = inputs
        ctx.patch = patch
        ctx.shape = (gh, gw, channels)

    def unpatch_backward(ctx, grad):
        if grad is None:
            return None, None, None, None, None
        gp = torch.ops.mlbricks_vision_native.patchify_layout(grad.contiguous(), ctx.patch)
        return gp, None, None, None, None

    torch.library.register_autograd(
        "mlbricks_vision_native::unpatchify", unpatch_backward, setup_context=unpatch_setup
    )

    def patch_setup(ctx, inputs, output):
        del output
        image, patch = inputs
        ctx.patch = patch
        ctx.gh = image.shape[2] // patch
        ctx.gw = image.shape[3] // patch
        ctx.channels = image.shape[1]

    def patch_backward(ctx, grad):
        if grad is None:
            return None, None
        gi = torch.ops.mlbricks_vision_native.unpatchify(
            grad.contiguous(), ctx.gh, ctx.gw, ctx.patch, ctx.channels
        )
        return gi, None

    torch.library.register_autograd(
        "mlbricks_vision_native::patchify_layout", patch_backward, setup_context=patch_setup
    )
    _REGISTERED = True


def available() -> bool:
    try:
        _load_extension()
    except RuntimeError:
        return False
    return True


def cuda_built() -> bool:
    return bool(_load_extension().has_cuda())


class _VisionRouteOwner:
    pass


_VISION_ROUTE_OWNER = _VisionRouteOwner()


def _use_native(
    x: Tensor,
    backend: str,
    *,
    op: str,
    owner: object | None = None,
    native_supports_training: bool = True,
    extra: tuple[object, ...] = (),
    default_auto: str | None = None,
) -> bool:
    policy = normalize_backend(backend, warn_legacy=True)
    if policy == "pytorch":
        return False
    if policy == "auto" and owner is not None:
        frozen = EXECUTION_PLANNER.owner_routes(owner).get(
            (str(op), bool(torch.is_grad_enabled()))
        )
        if frozen in {"native", "pytorch"}:
            return frozen == "native"
    ok = available()
    cuda_ok = ok and (not x.is_cuda or cuda_built())
    if policy == "native" and not ok:
        raise RuntimeError("backend='native' requested but shared vision native extension is unavailable")
    if policy == "native" and x.is_cuda and ok and not cuda_built():
        raise RuntimeError("backend='native' requested on CUDA but vision extension has no CUDA kernels")
    route = EXECUTION_PLANNER.select_operator_once(
        _VISION_ROUTE_OWNER if owner is None else owner,
        op,
        x,
        requested_backend=policy,
        native_available=bool(cuda_ok),
        native_supports_training=bool(native_supports_training),
        training=bool(torch.is_grad_enabled()),
        extra=extra,
        default_auto=default_auto,
    )
    return route == "native"


def scan_code(scan: str) -> int:
    try:
        return _SCAN_CODES[str(scan)]
    except KeyError as exc:
        raise ValueError("scan must be cross, horizontal, vertical, or raster") from exc


def reorder(x: Tensor, height: int, width: int, *, scan: str, layer_index: int, backend: str, inverse: bool = False, owner: object | None = None) -> Tensor | None:
    if not _use_native(x, backend, op="vision_scan", owner=owner, extra=(str(scan), int(layer_index), bool(inverse))):
        return None
    return torch.ops.mlbricks_vision_native.scan_reorder(
        x.contiguous(), int(height), int(width), scan_code(scan), int(layer_index), bool(inverse)
    )


def add_sincos2d(x: Tensor, height: int, width: int, *, backend: str, owner: object | None = None) -> Tensor | None:
    if not _use_native(x, backend, op="vision_position_2d", owner=owner):
        return None
    return torch.ops.mlbricks_vision_native.add_sincos2d(x.contiguous(), int(height), int(width))


def sincos2d(reference: Tensor, height: int, width: int, *, backend: str, owner: object | None = None) -> Tensor | None:
    if not _use_native(reference, backend, op="vision_position_2d", owner=owner):
        return None
    return torch.ops.mlbricks_vision_native.sincos2d(reference, int(height), int(width))


def unpatchify(patches: Tensor, gh: int, gw: int, patch: int, channels: int, *, backend: str, owner: object | None = None) -> Tensor | None:
    if not _use_native(patches, backend, op="vision_unpatchify", owner=owner, extra=(int(gh), int(gw), int(patch), int(channels))):
        return None
    return torch.ops.mlbricks_vision_native.unpatchify(
        patches.contiguous(), int(gh), int(gw), int(patch), int(channels)
    )


def bolt_full(q: Tensor, u: Tensor, g: Tensor, *, heads: int, latent_dim: int, head_dim: int, eps: float, causal: bool, backend: str, fused_inference: bool = False, owner: object | None = None) -> Tensor | None:
    if not _use_native(
        q, backend, op="bolt_full", owner=owner, native_supports_training=True,
        extra=(int(heads), int(latent_dim), int(head_dim), bool(causal), bool(fused_inference)),
    ):
        return None
    if fused_inference and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32) and latent_dim <= 64:
        return torch.ops.mlbricks_vision_native.bolt_full_fused(
            q.contiguous(), u.contiguous(), g.contiguous(), int(heads), int(latent_dim),
            int(head_dim), float(eps), bool(causal)
        )
    return torch.ops.mlbricks_vision_native.bolt_full(
        q, u, g, int(heads), int(latent_dim), int(head_dim), float(eps), bool(causal)
    )


def perspective_norm(x: Tensor, weight: Tensor, bias: Tensor, *, groups: int, eps: float = 1e-5, backend: str = "auto", owner: object | None = None) -> Tensor | None:
    if not _use_native(x, backend, op="perspective_norm", owner=owner, extra=(int(groups),)):
        return None
    return torch.ops.mlbricks_vision_native.perspective_norm(
        x, weight, bias, int(groups), float(eps)
    )


# Register operators eagerly when installed; pure-source checkouts remain valid.
try:
    _load_extension()
except RuntimeError:
    pass


__all__ = [
    "available", "cuda_built", "reorder", "add_sincos2d", "sincos2d",
    "unpatchify", "bolt_full", "perspective_norm", "scan_code",
]
