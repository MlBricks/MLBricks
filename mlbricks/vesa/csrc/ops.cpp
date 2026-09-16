// Copyright (c) 2026 Zameer Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>
#include <vector>

namespace vision_esa {

std::tuple<torch::Tensor, torch::Tensor> scan_forward_cpu(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    bool reverse);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_backward_cpu(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    const torch::Tensor& states,
    const torch::Tensor& grad_states,
    const torch::Tensor& grad_final,
    bool reverse);

torch::Tensor lightning_forward_cpu(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightning_backward_cpu(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state,
    const torch::Tensor& grad_output);

#ifdef WITH_CUDA
std::tuple<torch::Tensor, torch::Tensor> scan_forward_cuda(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    bool reverse);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_backward_cuda(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    const torch::Tensor& states,
    const torch::Tensor& grad_states,
    const torch::Tensor& grad_final,
    bool reverse);

torch::Tensor lightning_forward_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightning_backward_cuda(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state,
    const torch::Tensor& grad_output);
#endif

static void check_scan_inputs(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state) {
    TORCH_CHECK(gates.dim() == 3, "gates must have shape [batch, tokens, dim]");
    TORCH_CHECK(values.sizes() == gates.sizes(), "values must match gates shape");
    TORCH_CHECK(initial_state.dim() == 2, "initial_state must have shape [batch, dim]");
    TORCH_CHECK(initial_state.size(0) == gates.size(0) && initial_state.size(1) == gates.size(2),
                "initial_state must have shape [batch, dim]");
    TORCH_CHECK(gates.device() == values.device() && gates.device() == initial_state.device(),
                "gates, values, and initial_state must be on the same device");
    TORCH_CHECK(gates.scalar_type() == values.scalar_type() &&
                gates.scalar_type() == initial_state.scalar_type(),
                "gates, values, and initial_state must have the same dtype");
    TORCH_CHECK(gates.is_floating_point(), "native ESA supports floating-point tensors only");
}

static void check_step_inputs(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state) {
    TORCH_CHECK(gate.dim() == 2, "gate, value, and state must have shape [batch, dim]");
    TORCH_CHECK(value.sizes() == gate.sizes() && state.sizes() == gate.sizes(),
                "gate, value, and state must have identical shapes");
    TORCH_CHECK(gate.device() == value.device() && gate.device() == state.device(),
                "gate, value, and state must be on the same device");
    TORCH_CHECK(gate.scalar_type() == value.scalar_type() && gate.scalar_type() == state.scalar_type(),
                "gate, value, and state must have the same dtype");
    TORCH_CHECK(gate.is_floating_point(), "native ESA supports floating-point tensors only");
}

std::tuple<torch::Tensor, torch::Tensor> scan_forward(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    bool reverse) {
    check_scan_inputs(gates, values, initial_state);
    if (gates.is_cuda()) {
#ifdef WITH_CUDA
        return scan_forward_cuda(gates, values, initial_state, reverse);
#else
        TORCH_CHECK(false, "VisionESA native extension was built without CUDA support");
#endif
    }
    return scan_forward_cpu(gates, values, initial_state, reverse);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_backward(
    const torch::Tensor& gates,
    const torch::Tensor& values,
    const torch::Tensor& initial_state,
    const torch::Tensor& states,
    const torch::Tensor& grad_states,
    const torch::Tensor& grad_final,
    bool reverse) {
    check_scan_inputs(gates, values, initial_state);
    TORCH_CHECK(states.sizes() == gates.sizes(), "states must match gates shape");
    TORCH_CHECK(grad_states.sizes() == gates.sizes(), "grad_states must match gates shape");
    TORCH_CHECK(grad_final.sizes() == initial_state.sizes(), "grad_final must match initial_state shape");
    TORCH_CHECK(states.device() == gates.device() && grad_states.device() == gates.device() &&
                grad_final.device() == gates.device(), "all backward tensors must be on the same device");
    TORCH_CHECK(states.scalar_type() == gates.scalar_type() &&
                grad_states.scalar_type() == gates.scalar_type() &&
                grad_final.scalar_type() == gates.scalar_type(),
                "all backward tensors must have the same dtype");
    if (gates.is_cuda()) {
#ifdef WITH_CUDA
        return scan_backward_cuda(
            gates, values, initial_state, states, grad_states, grad_final, reverse);
#else
        TORCH_CHECK(false, "VisionESA native extension was built without CUDA support");
#endif
    }
    return scan_backward_cpu(
        gates, values, initial_state, states, grad_states, grad_final, reverse);
}

torch::Tensor lightning_forward(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state) {
    check_step_inputs(gate, value, state);
    if (gate.is_cuda()) {
#ifdef WITH_CUDA
        return lightning_forward_cuda(gate, value, state);
#else
        TORCH_CHECK(false, "VisionESA native extension was built without CUDA support");
#endif
    }
    return lightning_forward_cpu(gate, value, state);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightning_backward(
    const torch::Tensor& gate,
    const torch::Tensor& value,
    const torch::Tensor& state,
    const torch::Tensor& grad_output) {
    check_step_inputs(gate, value, state);
    TORCH_CHECK(grad_output.sizes() == gate.sizes(), "grad_output must match gate shape");
    TORCH_CHECK(grad_output.device() == gate.device(), "grad_output must be on the same device");
    TORCH_CHECK(grad_output.scalar_type() == gate.scalar_type(), "grad_output dtype must match gate dtype");
    if (gate.is_cuda()) {
#ifdef WITH_CUDA
        return lightning_backward_cuda(gate, value, state, grad_output);
#else
        TORCH_CHECK(false, "VisionESA native extension was built without CUDA support");
#endif
    }
    return lightning_backward_cpu(gate, value, state, grad_output);
}

bool has_cuda() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace vision_esa

TORCH_LIBRARY(mlbricks_vesa_native, m) {
    m.def("scan_forward(Tensor gates, Tensor values, Tensor initial_state, bool reverse=False) -> (Tensor, Tensor)");
    m.def("scan_backward(Tensor gates, Tensor values, Tensor initial_state, Tensor states, Tensor grad_states, Tensor grad_final, bool reverse=False) -> (Tensor, Tensor, Tensor)");
    m.def("lightning_forward(Tensor gate, Tensor value, Tensor state) -> Tensor");
    m.def("lightning_backward(Tensor gate, Tensor value, Tensor state, Tensor grad_output) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mlbricks_vesa_native, CompositeExplicitAutograd, m) {
    m.impl("scan_forward", TORCH_FN(vision_esa::scan_forward));
    m.impl("scan_backward", TORCH_FN(vision_esa::scan_backward));
    m.impl("lightning_forward", TORCH_FN(vision_esa::lightning_forward));
    m.impl("lightning_backward", TORCH_FN(vision_esa::lightning_backward));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VisionESA native C++/CUDA recurrence operators";
    m.def("has_cuda", &vision_esa::has_cuda, "Whether the extension was built with CUDA support");
}
