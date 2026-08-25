# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE.md and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

from collections.abc import Iterable
import warnings
from typing import Any

import torch


# Adam/AdamW's historical default eps=1e-8 is too small for direct FP16
# parameter/state updates.  On CUDA (and on CPU in a simple reproduction) it
# can round/flush to zero, so a parameter whose second-moment accumulator is
# still zero sees a 0/0 update and becomes NaN on the first optimizer step.
FP16_ADAM_MIN_EPS = 1e-5
DEFAULT_ADAM_EPS = 1e-8


def _group_has_fp16(group: dict[str, Any]) -> bool:
    return any(
        isinstance(parameter, torch.Tensor)
        and parameter.requires_grad
        and parameter.is_floating_point()
        and parameter.dtype == torch.float16
        for parameter in group.get("params", ())
    )


def stabilize_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    min_fp16_eps: float = FP16_ADAM_MIN_EPS,
    warn: bool = True,
) -> bool:
    """Make Adam-family optimizer groups safe for direct FP16 parameters.

    The function changes only Adam/AdamW groups that actually contain FP16
    trainable parameters and whose epsilon is below ``min_fp16_eps``.  FP32
    and BF16 groups are left untouched.

    Returns ``True`` when at least one group was adjusted.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return False
    if min_fp16_eps <= 0:
        raise ValueError("min_fp16_eps must be positive")

    changed = False
    for group in optimizer.param_groups:
        if not _group_has_fp16(group):
            continue
        current = float(group.get("eps", DEFAULT_ADAM_EPS))
        if current < min_fp16_eps:
            group["eps"] = float(min_fp16_eps)
            changed = True

    if changed and warn:
        warnings.warn(
            "Adjusted Adam/AdamW eps to 1e-5 for FP16 parameters to prevent "
            "first-step NaNs. Pass an FP16-safe epsilon explicitly or use "
            "mlbricks.AdamW/mlbricks.Adam to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
    return changed


class AdamW(torch.optim.AdamW):
    """AdamW with an FP16-safe epsilon policy.

    It is API-compatible with ``torch.optim.AdamW``.  When a parameter group
    contains direct FP16 parameters, epsilon is raised to at least ``1e-5``.
    FP32/BF16 groups retain the standard epsilon unless the user specifies a
    different value.

    This intentionally does not change FFNBrick mathematics and does not
    affect inference.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = DEFAULT_ADAM_EPS,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        stabilize_optimizer(self, warn=False)


class Adam(torch.optim.Adam):
    """Adam with the same FP16-safe epsilon policy as :class:`AdamW`."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = DEFAULT_ADAM_EPS,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        decoupled_weight_decay: bool = False,
    ) -> None:
        # decoupled_weight_decay was added to newer torch versions.  Keep the
        # package compatible with torch>=2.1 by passing it only when supported.
        kwargs: dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        import inspect
        if "decoupled_weight_decay" in inspect.signature(torch.optim.Adam).parameters:
            kwargs["decoupled_weight_decay"] = decoupled_weight_decay
        super().__init__(params, **kwargs)
        stabilize_optimizer(self, warn=False)


__all__ = [
    "Adam",
    "AdamW",
    "DEFAULT_ADAM_EPS",
    "FP16_ADAM_MIN_EPS",
    "stabilize_optimizer",
]
