"""Residual control for MLBricks neural-network components."""

from __future__ import annotations

import torch
from torch import nn

from . import native as _native
from ..runtime import normalize_backend


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)


class ResController(nn.Module):
    """Adaptively control update energy and residual-stream growth.

    Args:
        update_ratio: Maximum update RMS relative to the residual RMS.
        stream_ratio: Maximum candidate-stream RMS relative to residual RMS.
        update_softness: Sharpness of update-pressure gating.
        stream_softness: Sharpness of stream-pressure gating.
        eps: Positive numerical-stability constant.
        use_native: Use the compiled C++ backend when available.
        fused_cuda: Use the fused CUDA inference kernel when available.
    """

    def __init__(
        self,
        update_ratio: float,
        stream_ratio: float = 1.08,
        update_softness: float = 8.0,
        stream_softness: float = 8.0,
        eps: float = 1e-12,
        *,
        use_native: bool | None = None,
        fused_cuda: bool = True,
        backend: str = "auto",
    ) -> None:
        super().__init__()

        values = {
            "update_ratio": update_ratio,
            "stream_ratio": stream_ratio,
            "update_softness": update_softness,
            "stream_softness": stream_softness,
            "eps": eps,
        }
        for name, value in values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero")

        self.update_ratio = float(update_ratio)
        self.stream_ratio = float(stream_ratio)
        self.update_softness = float(update_softness)
        self.stream_softness = float(stream_softness)
        self.eps = float(eps)
        self.backend = normalize_backend(backend, warn_legacy=True)
        if use_native is not None and self.backend == "auto":
            self.backend = "auto" if bool(use_native) else "pytorch"
        self.use_native = self.backend != "pytorch"
        self.fused_cuda = bool(fused_cuda)

    def set_backend(self, backend: str, *, recursive: bool = True):
        del recursive
        self.backend = normalize_backend(backend, warn_legacy=True)
        self.use_native = self.backend != "pytorch"
        return self

    def resolved_backend(self) -> str:
        if self.backend == "pytorch":
            return "pytorch"
        if self.backend == "native":
            return "native-required"
        return "planner(auto)" if _native.is_available() else "pytorch"

    def _python_forward(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        residual32 = residual.float()
        update32 = update.float()

        residual_rms = _rms(residual32, self.eps)
        raw_update_rms = _rms(update32, self.eps)

        allowed_update_rms = self.update_ratio * residual_rms
        hard_update_scale = torch.clamp(
            allowed_update_rms / (raw_update_rms + self.eps),
            max=1.0,
        )
        update_pressure = raw_update_rms / (allowed_update_rms + self.eps)
        update_gate = torch.sigmoid(
            self.update_softness * (update_pressure - 1.0)
        )
        update_scale = 1.0 - update_gate * (1.0 - hard_update_scale)
        bounded_update = update32 * update_scale

        candidate = residual32 + bounded_update
        candidate_rms = _rms(candidate, self.eps)
        allowed_stream_rms = self.stream_ratio * residual_rms
        hard_stream_scale = torch.clamp(
            allowed_stream_rms / (candidate_rms + self.eps),
            max=1.0,
        )
        stream_pressure = candidate_rms / (allowed_stream_rms + self.eps)
        stream_gate = torch.sigmoid(
            self.stream_softness * (stream_pressure - 1.0)
        )
        stream_scale = 1.0 - stream_gate * (1.0 - hard_stream_scale)

        final_update = bounded_update * stream_scale
        return (residual32 + final_update).to(residual.dtype)

    def forward(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """Combine residual and update under adaptive RMS control."""
        if residual.shape != update.shape:
            raise ValueError(
                "residual and update must have identical shapes; "
                f"got {tuple(residual.shape)} and {tuple(update.shape)}"
            )
        if not residual.is_floating_point() or not update.is_floating_point():
            raise TypeError("residual and update must be floating-point tensors")
        if residual.device != update.device:
            raise ValueError(
                "residual and update must be on the same device; "
                f"got {residual.device} and {update.device}"
            )

        native_allowed = _native.inference_native_allowed(self, residual, update)
        if self.backend == "native" and not native_allowed:
            raise RuntimeError(
                "ResController backend='native' requested but native CUDA eager inference "
                "is unavailable for this call"
            )
        if native_allowed:
            return _native.residual_forward(
                residual,
                update,
                update_ratio=self.update_ratio,
                stream_ratio=self.stream_ratio,
                update_softness=self.update_softness,
                stream_softness=self.stream_softness,
                eps=self.eps,
                fused_cuda=self.fused_cuda,
            )

        return self._python_forward(residual, update)

    def extra_repr(self) -> str:
        return (
            f"update_ratio={self.update_ratio}, "
            f"stream_ratio={self.stream_ratio}, "
            f"update_softness={self.update_softness}, "
            f"stream_softness={self.stream_softness}, "
            f"eps={self.eps}, "
            f"backend={self.backend!r}, "
            f"fused_cuda={self.fused_cuda}"
        )
