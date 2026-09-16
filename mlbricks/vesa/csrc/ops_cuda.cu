// Copyright (c) 2026 Zameer Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <tuple>
#include <vector>

namespace vision_esa {

template <typename scalar_t>
__device__ __forceinline__ scalar_t qstep(scalar_t g, scalar_t v, scalar_t s) {
    using acc_t = at::acc_type<scalar_t, true>;
    const acc_t ga = static_cast<acc_t>(g);
    const acc_t va = static_cast<acc_t>(v);
    const acc_t sa = static_cast<acc_t>(s);
    return static_cast<scalar_t>(ga * sa + (acc_t(1) - ga) * va);
}

template <typename scalar_t>
__global__ void scan_forward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ initial,
    scalar_t* __restrict__ states,
    scalar_t* __restrict__ final_state,
    int64_t B,
    int64_t T,
    int64_t D,
    bool reverse) {
    const int64_t bd = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bd >= B * D) return;
    const int64_t b = bd / D;
    const int64_t d = bd - b * D;
    scalar_t state = initial[bd];

    for (int64_t k = 0; k < T; ++k) {
        const int64_t t = reverse ? (T - 1 - k) : k;
        const int64_t idx = (b * T + t) * D + d;
        state = qstep<scalar_t>(gates[idx], values[idx], state);
        states[idx] = state;
    }
    final_state[bd] = state;
}

template <typename scalar_t>
__global__ void scan_backward_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ initial,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ grad_states,
    const scalar_t* __restrict__ grad_final,
    scalar_t* __restrict__ grad_gates,
    scalar_t* __restrict__ grad_values,
    scalar_t* __restrict__ grad_initial,
    int64_t B,
    int64_t T,
    int64_t D,
    bool reverse) {
    using acc_t = at::acc_type<scalar_t, true>;
    const int64_t bd = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bd >= B * D) return;
    const int64_t b = bd / D;
    const int64_t d = bd - b * D;
    acc_t adj = static_cast<acc_t>(grad_final[bd]);

    for (int64_t k = T - 1; k >= 0; --k) {
        const int64_t t = reverse ? (T - 1 - k) : k;
        const int64_t idx = (b * T + t) * D + d;
        adj += static_cast<acc_t>(grad_states[idx]);

        scalar_t prev;
        if (k == 0) {
            prev = initial[bd];
        } else {
            const int64_t prev_t = reverse ? (T - k) : (k - 1);
            const int64_t prev_idx = (b * T + prev_t) * D + d;
            prev = states[prev_idx];
        }

        const acc_t ga = static_cast<acc_t>(gates[idx]);
        const acc_t va = static_cast<acc_t>(values[idx]);
        const acc_t pa = static_cast<acc_t>(prev);
        grad_gates[idx] = static_cast<scalar_t>(adj * (pa - va));
        grad_values[idx] = static_cast<scalar_t>(adj * (acc_t(1) - ga));
        adj *= ga;
    }
    grad_initial[bd] = static_cast<scalar_t>(adj);
}

template <typename scalar_t>
__global__ void lightning_forward_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ out,
    int64_t N) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) out[i] = qstep<scalar_t>(gate[i], value[i], state[i]);
}

template <typename scalar_t>
__global__ void lightning_backward_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_gate,
    scalar_t* __restrict__ grad_value,
    scalar_t* __restrict__ grad_state,
    int64_t N) {
    using acc_t = at::acc_type<scalar_t, true>;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const acc_t grad = static_cast<acc_t>(grad_output[i]);
    const acc_t ga = static_cast<acc_t>(gate[i]);
    const acc_t va = static_cast<acc_t>(value[i]);
    const acc_t sa = static_cast<acc_t>(state[i]);
    grad_gate[i] = static_cast<scalar_t>(grad * (sa - va));
    grad_value[i] = static_cast<scalar_t>(grad * (acc_t(1) - ga));
    grad_state[i] = static_cast<scalar_t>(grad * ga);
}

static inline dim3 blocks_for(int64_t n, int threads = 256) {
    return dim3(static_cast<unsigned int>((n + threads - 1) / threads));
}

std::tuple<torch::Tensor, torch::Tensor> scan_forward_cuda(
    const torch::Tensor& gates_in,
    const torch::Tensor& values_in,
    const torch::Tensor& initial_in,
    bool reverse) {
    const c10::cuda::CUDAGuard device_guard(gates_in.device());
    auto gates = gates_in.contiguous();
    auto values = values_in.contiguous();
    auto initial = initial_in.contiguous();
    auto states = torch::empty_like(gates);
    auto final_state = torch::empty_like(initial);
    const int64_t B = gates.size(0);
    const int64_t T = gates.size(1);
    const int64_t D = gates.size(2);

    if (T == 0) {
        final_state.copy_(initial);
        return {states, final_state};
    }

    constexpr int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gates.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gates.scalar_type(),
        "vision_esa_scan_forward_cuda",
        [&] {
            scan_forward_kernel<scalar_t><<<blocks_for(B * D, threads), threads, 0, stream>>>(
                gates.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                initial.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                final_state.data_ptr<scalar_t>(),
                B, T, D, reverse);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {states, final_state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_backward_cuda(
    const torch::Tensor& gates_in,
    const torch::Tensor& values_in,
    const torch::Tensor& initial_in,
    const torch::Tensor& states_in,
    const torch::Tensor& grad_states_in,
    const torch::Tensor& grad_final_in,
    bool reverse) {
    const c10::cuda::CUDAGuard device_guard(gates_in.device());
    auto gates = gates_in.contiguous();
    auto values = values_in.contiguous();
    auto initial = initial_in.contiguous();
    auto states = states_in.contiguous();
    auto grad_states = grad_states_in.contiguous();
    auto grad_final = grad_final_in.contiguous();
    auto grad_gates = torch::empty_like(gates);
    auto grad_values = torch::empty_like(values);
    auto grad_initial = torch::empty_like(initial);
    const int64_t B = gates.size(0);
    const int64_t T = gates.size(1);
    const int64_t D = gates.size(2);

    if (T == 0) {
        grad_initial.copy_(grad_final);
        return {grad_gates, grad_values, grad_initial};
    }

    constexpr int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gates.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gates.scalar_type(),
        "vision_esa_scan_backward_cuda",
        [&] {
            scan_backward_kernel<scalar_t><<<blocks_for(B * D, threads), threads, 0, stream>>>(
                gates.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                initial.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                grad_states.data_ptr<scalar_t>(),
                grad_final.data_ptr<scalar_t>(),
                grad_gates.data_ptr<scalar_t>(),
                grad_values.data_ptr<scalar_t>(),
                grad_initial.data_ptr<scalar_t>(),
                B, T, D, reverse);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_gates, grad_values, grad_initial};
}

torch::Tensor lightning_forward_cuda(
    const torch::Tensor& gate_in,
    const torch::Tensor& value_in,
    const torch::Tensor& state_in) {
    const c10::cuda::CUDAGuard device_guard(gate_in.device());
    auto gate = gate_in.contiguous();
    auto value = value_in.contiguous();
    auto state = state_in.contiguous();
    auto out = torch::empty_like(state);
    const int64_t N = state.numel();
    if (N == 0) return out;

    constexpr int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gate.scalar_type(),
        "vision_esa_lightning_forward_cuda",
        [&] {
            lightning_forward_kernel<scalar_t><<<blocks_for(N, threads), threads, 0, stream>>>(
                gate.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), N);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightning_backward_cuda(
    const torch::Tensor& gate_in,
    const torch::Tensor& value_in,
    const torch::Tensor& state_in,
    const torch::Tensor& grad_output_in) {
    const c10::cuda::CUDAGuard device_guard(gate_in.device());
    auto gate = gate_in.contiguous();
    auto value = value_in.contiguous();
    auto state = state_in.contiguous();
    auto grad_output = grad_output_in.contiguous();
    auto grad_gate = torch::empty_like(gate);
    auto grad_value = torch::empty_like(value);
    auto grad_state = torch::empty_like(state);
    const int64_t N = state.numel();
    if (N == 0) return {grad_gate, grad_value, grad_state};

    constexpr int threads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(gate.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gate.scalar_type(),
        "vision_esa_lightning_backward_cuda",
        [&] {
            lightning_backward_kernel<scalar_t><<<blocks_for(N, threads), threads, 0, stream>>>(
                gate.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(), grad_gate.data_ptr<scalar_t>(),
                grad_value.data_ptr<scalar_t>(), grad_state.data_ptr<scalar_t>(), N);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_gate, grad_value, grad_state};
}

}  // namespace vision_esa
