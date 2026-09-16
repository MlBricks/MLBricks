#pragma once

#include <torch/extension.h>
#include <vector>

namespace ffnbrick {

using torch::Tensor;

Tensor micro_virtual_forward_packed(
    const Tensor& x,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    bool fused_cuda);

Tensor micro_virtual_refine_packed(
    const Tensor& x,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    int64_t refinements,
    bool fused_cuda);

std::vector<Tensor> state_aware_forward_packed(
    const Tensor& x,
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& previous_state,
    const std::vector<Tensor>& p,
    bool fused_cuda);

std::vector<Tensor> virtual_state_aware_forward_packed(
    const Tensor& x,
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& previous_state,
    const std::vector<Tensor>& p,
    const std::vector<Tensor>& vp,
    int64_t refinements,
    double rms_eps,
    bool fused_cuda);

#ifdef WITH_CUDA
Tensor silu_mul_cuda(const Tensor& gate, const Tensor& value);
Tensor silu_mul_packed_cuda(const Tensor& gate_up, int64_t hidden_dim);
std::vector<Tensor> transition_delta_cuda(
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& candidate_transition,
    const Tensor& write_transition,
    const Tensor& delta_scale);
Tensor state_mix_cuda(
    const Tensor& candidate_pre,
    const Tensor& write_pre,
    const Tensor& previous_state,
    const Tensor& scaled_delta,
    const Tensor& retain_logit,
    const Tensor& retain_delta_scale);
Tensor read_mix_cuda(
    const Tensor& value_pre,
    const Tensor& next_state,
    const Tensor& scaled_delta,
    const Tensor& read_logit,
    const Tensor& read_delta_scale);
Tensor rms_norm_cuda(
    const Tensor& state,
    const Tensor& weight,
    double eps);
Tensor condition_silu_cuda(
    const Tensor& base,
    const Tensor& x_condition,
    const Tensor& esa_condition,
    const Tensor& pass_embedding);
Tensor residual_gate_cuda(
    const Tensor& state,
    const Tensor& update,
    const Tensor& gate_logit);
#endif

} // namespace ffnbrick
