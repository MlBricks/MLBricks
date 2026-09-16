// Copyright 2026 Zameer Hussain and Akhtar Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>

#include <cstdint>

namespace mlbricks {

torch::Tensor thunder_scan_cpu(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t /*compass*/) {
    auto out = torch::empty_like(B_write);

    const auto B = A.size(0);
    const auto T = A.size(1);
    const auto H = A.size(2);
    const auto D = A.size(3);
    const auto C = H * D;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_scan_cpu",
        [&] {
            using acc_t = at::acc_type<scalar_t, false>;
            const scalar_t* a = A.data_ptr<scalar_t>();
            const scalar_t* b = B_write.data_ptr<scalar_t>();
            scalar_t* y = out.data_ptr<scalar_t>();

            at::parallel_for(0, B * C, 0, [&](int64_t begin, int64_t end) {
                for (int64_t bc = begin; bc < end; ++bc) {
                    const int64_t batch = bc / C;
                    const int64_t channel = bc % C;
                    acc_t state = acc_t(0);
                    for (int64_t t = 0; t < T; ++t) {
                        const int64_t idx = (batch * T + t) * C + channel;
                        state = static_cast<acc_t>(a[idx]) * state + static_cast<acc_t>(b[idx]);
                        y[idx] = static_cast<scalar_t>(state);
                    }
                }
            });
        });

    return out;
}

torch::Tensor lightning_step_cpu(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state) {
    auto out = torch::empty_like(state);
    const auto N = state.numel();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        state.scalar_type(),
        "mlbricks_lightning_step_cpu",
        [&] {
            using acc_t = at::acc_type<scalar_t, false>;
            const scalar_t* a = A.data_ptr<scalar_t>();
            const scalar_t* b = B_write.data_ptr<scalar_t>();
            const scalar_t* s = state.data_ptr<scalar_t>();
            scalar_t* y = out.data_ptr<scalar_t>();
            at::parallel_for(0, N, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const acc_t next = static_cast<acc_t>(a[i]) * static_cast<acc_t>(s[i])
                                     + static_cast<acc_t>(b[i]);
                    y[i] = static_cast<scalar_t>(next);
                }
            });
        });

    return out;
}


std::vector<torch::Tensor> residual_layer_norm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
    auto residual = x + update;
    auto mean = residual.to(torch::kFloat32).mean(-1, true);
    auto centered = residual.to(torch::kFloat32) - mean;
    auto variance = centered.pow(2).mean(-1, true);
    auto normalized = centered * torch::rsqrt(variance + eps);
    normalized = normalized.to(x.scalar_type()) * weight + bias;
    return {residual, normalized};
}

} // namespace mlbricks
