# Copyright 2026 Zameer Hussain and Akhtar Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

"""PyTorch subsystem registrations for MLBricks native operators.

The C++ extension owns the operator schemas/implementations through
``TORCH_LIBRARY(mlbricks_native, ...)``.  This module adds the Python-side
registrations needed by FakeTensor/AOTAutograd/torch.compile and direct
training use of the recurrent scan operators.

It is internal and imported by :mod:`mlbricks.native` after ``mlbricks._C``
has loaded (which causes the C++ dispatcher schemas to be registered).
"""
from __future__ import annotations

from typing import Callable

import torch

_NAMESPACE = "mlbricks_native"
_REGISTERED = False


def _has_op(name: str) -> bool:
    try:
        getattr(getattr(torch.ops, _NAMESPACE), name)
        return True
    except (AttributeError, RuntimeError):
        return False


def _safe_fake(name: str, fn: Callable) -> None:
    """Register a FakeTensor kernel when this PyTorch exposes the API.

    PyTorch 2.10+ uses ``register_fake``.  ``impl_abstract`` is retained as a
    compatibility fallback for older supported installations.
    """
    full_name = f"{_NAMESPACE}::{name}"
    register = getattr(torch.library, "register_fake", None)
    if register is None:
        register = getattr(torch.library, "impl_abstract", None)
    if register is None:  # pragma: no cover - very old torch fallback
        return
    try:
        register(full_name)(fn)
    except RuntimeError as exc:
        # Harmless on notebook/module reload when an identical registration is
        # already present.  Anything else should remain visible to developers.
        msg = str(exc).lower()
        if "already" not in msg and "previous" not in msg:
            raise


def _safe_autograd(name: str, backward: Callable, setup_context: Callable) -> None:
    register = getattr(torch.library, "register_autograd", None)
    if register is None:  # pragma: no cover - torch versions before API
        return
    full_name = f"{_NAMESPACE}::{name}"
    try:
        register(full_name, backward, setup_context=setup_context)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "already" not in msg and "previous" not in msg:
            raise


def _empty_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _fake_fused_readout(
    qgv: torch.Tensor,
    embd: int,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    return qgv.new_empty((qgv.shape[0], qgv.shape[1], embd))


def _fake_full_forward(
    x: torch.Tensor,
    qgv_weight: torch.Tensor,
    out_weight: torch.Tensor,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1], out_weight.shape[0]))


def _fake_full_forward_hybrid(
    x: torch.Tensor,
    qgv_weight: torch.Tensor,
    out_weight: torch.Tensor,
    gate_min: float,
    gate_max: float,
    eps: float,
    compass: int,
    out_chunk: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], x.shape[1], out_weight.shape[0]))


def _fake_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _fake_linear_hybrid(
    x: torch.Tensor, weight: torch.Tensor, chunk: int
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _fake_prepare_ab(
    qgv: torch.Tensor, embd: int, gate_min: float, gate_max: float
) -> list[torch.Tensor]:
    shape = (qgv.shape[0], qgv.shape[1], embd)
    return [qgv.new_empty(shape), qgv.new_empty(shape)]


def _fake_readout(q: torch.Tensor, states: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(q)


def _fake_scan(
    A: torch.Tensor, B_write: torch.Tensor, compass: int
) -> torch.Tensor:
    return torch.empty_like(A)


def _fake_scan_local(
    A: torch.Tensor, B_write: torch.Tensor, compass: int
) -> list[torch.Tensor]:
    chunks = (A.shape[1] + compass - 1) // compass
    channels = A.shape[2] * A.shape[3]
    summary_shape = (A.shape[0], chunks, channels)
    return [
        torch.empty_like(A),
        A.new_empty(summary_shape),
        A.new_empty(summary_shape),
    ]


def _fake_summary_scan(
    A: torch.Tensor, B_write: torch.Tensor, group_size: int
) -> list[torch.Tensor]:
    groups = (A.shape[1] + group_size - 1) // group_size
    parent_shape = (A.shape[0], groups, A.shape[2])
    return [
        torch.empty_like(A),
        torch.empty_like(B_write),
        A.new_empty(parent_shape),
        B_write.new_empty(parent_shape),
    ]


def _fake_group_prefix(A: torch.Tensor, B_write: torch.Tensor) -> list[torch.Tensor]:
    return [torch.empty_like(A), torch.empty_like(B_write)]


def _fake_apply_group(
    pref_A: torch.Tensor,
    pref_B: torch.Tensor,
    parent_A: torch.Tensor,
    parent_B: torch.Tensor,
    group_size: int,
) -> list[torch.Tensor]:
    return [torch.empty_like(pref_A), torch.empty_like(pref_B)]


def _fake_apply_chunk_prefix(
    A: torch.Tensor,
    local_states: torch.Tensor,
    chunk_B_prefix: torch.Tensor,
    compass: int,
) -> torch.Tensor:
    return torch.empty_like(A)


def _fake_backward_chunked(
    A: torch.Tensor,
    states: torch.Tensor,
    grad: torch.Tensor,
    compass: int,
) -> list[torch.Tensor]:
    return [torch.empty_like(A), torch.empty_like(A)]


def _fake_reverse_prepare(A: torch.Tensor, grad: torch.Tensor) -> list[torch.Tensor]:
    return [torch.empty_like(A), torch.empty_like(A)]


def _fake_reverse_finish(
    grad_reverse: torch.Tensor, states: torch.Tensor
) -> list[torch.Tensor]:
    return [torch.empty_like(grad_reverse), torch.empty_like(grad_reverse)]


def _fake_lightning_step(
    A: torch.Tensor, B_write: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(state)


def _fake_ffn_gelu_residual(
    normalized: torch.Tensor,
    residual: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def _fake_elastic_linear_packed(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], out_features))


def _fake_residual_layer_norm(
    x: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> list[torch.Tensor]:
    return [torch.empty_like(x), torch.empty_like(x)]


def _fake_lightning_fused_step(
    qgv: torch.Tensor,
    state: torch.Tensor,
    gate_min: float,
    gate_max: float,
    eps: float,
) -> list[torch.Tensor]:
    channels = state.shape[1] * state.shape[2]
    readout = qgv.new_empty((qgv.shape[0], channels))
    return [readout, torch.empty_like(state)]


def _scan_setup_context(ctx, inputs, output) -> None:
    A, B_write, compass = inputs
    ctx.compass = int(compass)
    ctx.save_for_backward(A, output)


def _scan_backward(ctx, grad_out: torch.Tensor):
    A, states = ctx.saved_tensors
    if A.is_cuda:
        # The native chunked reverse recurrence is valid for both the direct and
        # hierarchical forward kernels. Because this calls another dispatcher
        # op, AOTAutograd/torch.compile can keep the CUDA backward boundary
        # visible instead of graph-breaking on pybind.
        grad_A, grad_B = torch.ops.mlbricks_native.thunder_scan_backward_chunked(
            A.contiguous(), states.contiguous(), grad_out.contiguous(), ctx.compass
        )
        return grad_A, grad_B, None

    # ``thunder_scan`` also has a CPU implementation. Preserve direct CPU
    # autograd with a PyTorch reverse recurrence rather than calling the
    # CUDA-only native backward kernel. This path is primarily correctness/CI;
    # performance-critical training uses the CUDA dispatcher op above.
    grad_A = torch.empty_like(A)
    grad_B = torch.empty_like(A)
    carry = torch.zeros_like(A[:, 0])
    zero = torch.zeros_like(A[:, 0])
    for t in range(A.shape[1] - 1, -1, -1):
        total = grad_out[:, t] + carry
        prev_state = states[:, t - 1] if t > 0 else zero
        grad_A[:, t] = total * prev_state
        grad_B[:, t] = total
        carry = total * A[:, t]
    return grad_A, grad_B, None


def register_native_custom_ops() -> bool:
    """Attach FakeTensor/autograd registrations to the loaded C++ operators."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not _has_op("thunder_scan"):
        return False

    for name in (
        "thunder_fused_readout",
        "thunder_fused_readout_hierarchical",
        "thunder_fused_readout_hierarchical_mixed32",
        "thunder_fused_readout_hierarchical_full32",
        "thunder_fused_readout_hierarchical_precise_gate",
        "thunder_fused_readout_hierarchical_precise_readout",
        "thunder_fused_readout_hierarchical_precise_both",
    ):
        _safe_fake(name, _fake_fused_readout)

    for name in (
        "thunder_forward_hierarchical",
        "thunder_forward_hierarchical_fp16gemm",
        "thunder_forward_hierarchical_mixedgemm",
    ):
        _safe_fake(name, _fake_full_forward)
    _safe_fake("thunder_forward_hierarchical_hybridgemm", _fake_full_forward_hybrid)

    _safe_fake("linear_fp16_accum", _fake_linear)
    _safe_fake("linear_fp32_accum", _fake_linear)
    _safe_fake("linear_hybrid_accum", _fake_linear_hybrid)

    _safe_fake("thunder_prepare_ab", _fake_prepare_ab)
    _safe_fake("thunder_readout", _fake_readout)
    _safe_fake("thunder_prepare_ab_precise", _fake_prepare_ab)
    _safe_fake("thunder_readout_precise", _fake_readout)

    _safe_fake("thunder_scan", _fake_scan)
    _safe_fake("thunder_scan_hierarchical", _fake_scan)
    _safe_fake("thunder_scan_local", _fake_scan_local)
    _safe_fake("thunder_summary_scan", _fake_summary_scan)
    _safe_fake("thunder_group_prefix", _fake_group_prefix)
    _safe_fake("thunder_apply_group", _fake_apply_group)
    _safe_fake("thunder_apply_chunk_prefix", _fake_apply_chunk_prefix)
    _safe_fake("thunder_scan_backward_chunked", _fake_backward_chunked)
    _safe_fake("thunder_reverse_prepare", _fake_reverse_prepare)
    _safe_fake("thunder_reverse_finish", _fake_reverse_finish)

    _safe_fake("lightning_step", _fake_lightning_step)
    _safe_fake("lightning_fused_step", _fake_lightning_fused_step)
    _safe_fake("residual_layer_norm", _fake_residual_layer_norm)
    _safe_fake("elastic_linear_packed", _fake_elastic_linear_packed)
    _safe_fake("ffn_gelu_residual", _fake_ffn_gelu_residual)

    # Direct use of either scan now has an explicit autograd formula.  This is
    # preferable to relying on Autograd.Function around an opaque pybind call
    # and is the registration style recommended by PyTorch for torch.compile.
    _safe_autograd("thunder_scan", _scan_backward, _scan_setup_context)
    _safe_autograd("thunder_scan_hierarchical", _scan_backward, _scan_setup_context)

    _REGISTERED = True
    return True


__all__ = ["register_native_custom_ops"]
