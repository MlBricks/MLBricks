# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

from __future__ import annotations

import warnings
import torch

from .constants import SUPPORTED_PRECISIONS


def resolve_scan_dtype(
    precision: str,
    device: torch.device | str,
    *,
    strict_precision: bool = False,
) -> torch.dtype:
    """Resolve a user precision string into the scan accumulation dtype.

    Notes
    -----
    - fp16 is the default training mode. On CPU it falls back to fp32 unless
      strict_precision=True, because CPU fp16 training is usually not practical.
    - fp8 is accepted as an experimental mode. The current safe path uses fp16
      scan accumulation. A true FP8 kernel can be added later behind this flag.
    - fp64 is intended for correctness/reference checks, not normal training.
    """
    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision={precision!r}. "
            f"Expected one of {sorted(SUPPORTED_PRECISIONS)}."
        )

    dev = torch.device(device)

    if precision == "fp16":
        if dev.type == "cuda":
            return torch.float16
        if strict_precision:
            return torch.float16
        return torch.float32

    if precision == "bf16":
        return torch.bfloat16

    if precision == "fp32":
        return torch.float32

    if precision == "fp64":
        return torch.float64

    if precision == "fp8":
        msg = (
            "precision='fp8' is experimental in ESA v2. "
            "The current safe implementation uses fp16 scan accumulation. "
            "A true FP8 Thunder kernel can be added later."
        )
        if strict_precision:
            raise NotImplementedError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return torch.float16 if dev.type == "cuda" else torch.float32

    raise AssertionError("unreachable")
