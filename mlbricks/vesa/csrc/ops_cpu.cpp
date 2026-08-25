// Copyright (c) 2026 Zameer Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>

#include <tuple>
#include <vector>

namespace vision_esa {

template <typename scalar_t>
static inline scalar_t recurrence_step(scalar_t g, scalar_t v, scalar_t s) {
    using acc_t = at::acc_type<scalar_t, false>;
    const acc_t ga = static_cast<acc_t>(g);
    const acc_t va = static_cast<acc_t>(v);
    const acc_t sa = static_cast<acc_t>(s);
    return static_cast<scalar_t>(ga * sa + (acc_t(1) - ga) * va);
}

std::tuple<torch::Tensor, torch::Tensor> scan_forward_cpu(
    const torch::Tensor& gates_in,
    const torch::Tensor& values_in,
    const torch::Tensor& initial_in,
    bool reverse) {
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gates.scalar_type(),
        "vision_esa_scan_forward_cpu",
        [&] {
            const scalar_t* g = gates.data_ptr<scalar_t>();
            const scalar_t* v = values.data_ptr<scalar_t>();
            const scalar_t* init = initial.data_ptr<scalar_t>();
            scalar_t* y = states.data_ptr<scalar_t>();
            scalar_t* f = final_state.data_ptr<scalar_t>();

            at::parallel_for(0, B * D, 0, [&](int64_t begin, int64_t end) {
                for (int64_t bd = begin; bd < end; ++bd) {
                    const int64_t b = bd / D;
                    const int64_t d = bd - b * D;
                    scalar_t state = init[b * D + d];
                    for (int64_t k = 0; k < T; ++k) {
                        const int64_t t = reverse ? (T - 1 - k) : k;
                        const int64_t idx = (b * T + t) * D + d;
                        state = recurrence_step<scalar_t>(g[idx], v[idx], state);
                        y[idx] = state;
                    }
                    f[b * D + d] = state;
                }
            });
        });

    return {states, final_state};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> scan_backward_cpu(
    const torch::Tensor& gates_in,
    const torch::Tensor& values_in,
    const torch::Tensor& initial_in,
    const torch::Tensor& states_in,
    const torch::Tensor& grad_states_in,
    const torch::Tensor& grad_final_in,
    bool reverse) {
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gates.scalar_type(),
        "vision_esa_scan_backward_cpu",
        [&] {
            using acc_t = at::acc_type<scalar_t, false>;
            const scalar_t* g = gates.data_ptr<scalar_t>();
            const scalar_t* v = values.data_ptr<scalar_t>();
            const scalar_t* init = initial.data_ptr<scalar_t>();
            const scalar_t* y = states.data_ptr<scalar_t>();
            const scalar_t* gy = grad_states.data_ptr<scalar_t>();
            const scalar_t* gf = grad_final.data_ptr<scalar_t>();
            scalar_t* gg = grad_gates.data_ptr<scalar_t>();
            scalar_t* gv = grad_values.data_ptr<scalar_t>();
            scalar_t* gi = grad_initial.data_ptr<scalar_t>();

            at::parallel_for(0, B * D, 0, [&](int64_t begin, int64_t end) {
                for (int64_t bd = begin; bd < end; ++bd) {
                    const int64_t b = bd / D;
                    const int64_t d = bd - b * D;
                    acc_t adj = static_cast<acc_t>(gf[b * D + d]);

                    for (int64_t k = T - 1; k >= 0; --k) {
                        const int64_t t = reverse ? (T - 1 - k) : k;
                        const int64_t idx = (b * T + t) * D + d;
                        adj += static_cast<acc_t>(gy[idx]);

                        scalar_t prev;
                        if (k == 0) {
                            prev = init[b * D + d];
                        } else {
                            const int64_t prev_t = reverse ? (T - k) : (k - 1);
                            const int64_t prev_idx = (b * T + prev_t) * D + d;
                            prev = y[prev_idx];
                        }

                        const acc_t ga = static_cast<acc_t>(g[idx]);
                        const acc_t va = static_cast<acc_t>(v[idx]);
                        const acc_t pa = static_cast<acc_t>(prev);
                        gg[idx] = static_cast<scalar_t>(adj * (pa - va));
                        gv[idx] = static_cast<scalar_t>(adj * (acc_t(1) - ga));
                        adj *= ga;
                    }
                    gi[b * D + d] = static_cast<scalar_t>(adj);
                }
            });
        });

    return {grad_gates, grad_values, grad_initial};
}

torch::Tensor lightning_forward_cpu(
    const torch::Tensor& gate_in,
    const torch::Tensor& value_in,
    const torch::Tensor& state_in) {
    auto gate = gate_in.contiguous();
    auto value = value_in.contiguous();
    auto state = state_in.contiguous();
    auto out = torch::empty_like(state);
    const int64_t N = state.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        state.scalar_type(),
        "vision_esa_lightning_forward_cpu",
        [&] {
            const scalar_t* g = gate.data_ptr<scalar_t>();
            const scalar_t* v = value.data_ptr<scalar_t>();
            const scalar_t* s = state.data_ptr<scalar_t>();
            scalar_t* y = out.data_ptr<scalar_t>();
            at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    y[i] = recurrence_step<scalar_t>(g[i], v[i], s[i]);
                }
            });
        });
    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lightning_backward_cpu(
    const torch::Tensor& gate_in,
    const torch::Tensor& value_in,
    const torch::Tensor& state_in,
    const torch::Tensor& grad_output_in) {
    auto gate = gate_in.contiguous();
    auto value = value_in.contiguous();
    auto state = state_in.contiguous();
    auto grad_output = grad_output_in.contiguous();
    auto grad_gate = torch::empty_like(gate);
    auto grad_value = torch::empty_like(value);
    auto grad_state = torch::empty_like(state);
    const int64_t N = state.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        state.scalar_type(),
        "vision_esa_lightning_backward_cpu",
        [&] {
            using acc_t = at::acc_type<scalar_t, false>;
            const scalar_t* g = gate.data_ptr<scalar_t>();
            const scalar_t* v = value.data_ptr<scalar_t>();
            const scalar_t* s = state.data_ptr<scalar_t>();
            const scalar_t* go = grad_output.data_ptr<scalar_t>();
            scalar_t* gg = grad_gate.data_ptr<scalar_t>();
            scalar_t* gv = grad_value.data_ptr<scalar_t>();
            scalar_t* gs = grad_state.data_ptr<scalar_t>();
            at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const acc_t grad = static_cast<acc_t>(go[i]);
                    const acc_t ga = static_cast<acc_t>(g[i]);
                    const acc_t va = static_cast<acc_t>(v[i]);
                    const acc_t sa = static_cast<acc_t>(s[i]);
                    gg[i] = static_cast<scalar_t>(grad * (sa - va));
                    gv[i] = static_cast<scalar_t>(grad * (acc_t(1) - ga));
                    gs[i] = static_cast<scalar_t>(grad * ga);
                }
            });
        });
    return {grad_gate, grad_value, grad_state};
}

}  // namespace vision_esa
