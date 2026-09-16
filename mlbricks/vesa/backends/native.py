# Copyright (c) 2026 Zameer Hussain
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Native C++/CUDA ESA recurrence backend.

The extension owns the recurrent scan and one-token Lightning update. Linear,
normalization, convolution, and activation layers remain ordinary PyTorch ops,
which already dispatch to optimized C++/CUDA libraries on supported devices.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _load_extension():
    try:
        module = importlib.import_module("mlbricks.vesa._C")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "MLBricks VESA native extension is not built. Install the project from source "
            "with `pip install -v .` (CUDA toolkit present for CUDA support), or select "
            "the `thunder` backend."
        ) from exc
    _register_dispatch_helpers()
    return module


_REGISTERED = False


def _register_dispatch_helpers() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    # FakeTensor/meta implementations let torch.compile reason about shapes
    # without running the native kernel.
    @torch.library.register_fake("mlbricks_vesa_native::scan_forward")
    def _fake_scan_forward(
        gates: Tensor,
        values: Tensor,
        initial_state: Tensor,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del values, reverse
        return torch.empty_like(gates), torch.empty_like(initial_state)

    @torch.library.register_fake("mlbricks_vesa_native::scan_backward")
    def _fake_scan_backward(
        gates: Tensor,
        values: Tensor,
        initial_state: Tensor,
        states: Tensor,
        grad_states: Tensor,
        grad_final: Tensor,
        reverse: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del states, grad_states, grad_final, reverse
        return torch.empty_like(gates), torch.empty_like(values), torch.empty_like(initial_state)

    @torch.library.register_fake("mlbricks_vesa_native::lightning_forward")
    def _fake_lightning_forward(gate: Tensor, value: Tensor, state: Tensor) -> Tensor:
        del value, state
        return torch.empty_like(gate)

    @torch.library.register_fake("mlbricks_vesa_native::lightning_backward")
    def _fake_lightning_backward(
        gate: Tensor,
        value: Tensor,
        state: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del grad_output
        return torch.empty_like(gate), torch.empty_like(value), torch.empty_like(state)

    def scan_setup_context(ctx, inputs, output) -> None:
        gates, values, initial_state, reverse = inputs
        states, _ = output
        ctx.save_for_backward(gates, values, initial_state, states)
        ctx.reverse = reverse

    def scan_backward(ctx, grad_states, grad_final):
        gates, values, initial_state, states = ctx.saved_tensors
        if grad_states is None:
            grad_states = torch.zeros_like(states)
        if grad_final is None:
            grad_final = torch.zeros_like(initial_state)
        grad_gates, grad_values, grad_initial = torch.ops.mlbricks_vesa_native.scan_backward(
            gates,
            values,
            initial_state,
            states,
            grad_states.contiguous(),
            grad_final.contiguous(),
            ctx.reverse,
        )
        return grad_gates, grad_values, grad_initial, None

    torch.library.register_autograd(
        "mlbricks_vesa_native::scan_forward",
        scan_backward,
        setup_context=scan_setup_context,
    )

    def step_setup_context(ctx, inputs, output) -> None:
        del output
        gate, value, state = inputs
        ctx.save_for_backward(gate, value, state)

    def step_backward(ctx, grad_output):
        gate, value, state = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(state)
        return torch.ops.mlbricks_vesa_native.lightning_backward(
            gate, value, state, grad_output.contiguous()
        )

    torch.library.register_autograd(
        "mlbricks_vesa_native::lightning_forward",
        step_backward,
        setup_context=step_setup_context,
    )
    _REGISTERED = True


def native_scan(
    gates: Tensor,
    values: Tensor,
    initial_state: Tensor | None = None,
    *,
    reverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Execute the ESA recurrence in the compiled C++ or CUDA extension."""

    if not _REGISTERED:
        _load_extension()
    if gates.ndim != 3 or values.shape != gates.shape:
        raise ValueError("gates and values must have matching [batch, tokens, dim] shapes")
    batch, _, dim = gates.shape
    if initial_state is None:
        initial_state = torch.zeros(batch, dim, device=gates.device, dtype=gates.dtype)
    if initial_state.shape != (batch, dim):
        raise ValueError(f"initial_state must have shape {(batch, dim)}")
    return torch.ops.mlbricks_vesa_native.scan_forward(
        gates.contiguous(), values.contiguous(), initial_state.contiguous(), reverse
    )


def native_lightning_step(gate: Tensor, value: Tensor, state: Tensor) -> Tensor:
    """Execute one ESA recurrent update in the compiled C++ or CUDA extension."""

    if not _REGISTERED:
        _load_extension()
    if gate.ndim != 2 or gate.shape != value.shape or gate.shape != state.shape:
        raise ValueError("gate, value, and state must have identical [batch, dim] shapes")
    return torch.ops.mlbricks_vesa_native.lightning_forward(
        gate.contiguous(), value.contiguous(), state.contiguous()
    )


def native_available() -> bool:
    """Return True when the compiled extension can be imported."""

    try:
        _load_extension()
    except RuntimeError:
        return False
    return True


def native_cuda_built() -> bool:
    """Return whether the loaded extension contains CUDA kernels."""

    return bool(_load_extension().has_cuda())


# If the package extension is already installed, register its operators during
# module import. Failure is intentionally silent so pure-Python backends remain
# usable from an unbuilt source checkout.
_EAGER_LOAD_ATTEMPTED = True
try:
    _load_extension()
except RuntimeError:
    pass
