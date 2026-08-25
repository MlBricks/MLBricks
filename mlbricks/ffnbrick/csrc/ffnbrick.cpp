#include "ffnbrick.h"

#include <ATen/Functions.h>
#include <c10/util/Optional.h>

namespace ffnbrick {
namespace {

inline Tensor linear_bias(const Tensor& x, const Tensor& w, const Tensor& b) {
    return at::linear(x, w, b);
}

inline Tensor linear_nobias(const Tensor& x, const Tensor& w) {
    return at::linear(x, w, c10::nullopt);
}

inline bool all_same_dtype(const Tensor& x, const std::vector<Tensor>& ts) {
    for (const auto& t : ts) {
        if (t.defined() && t.numel() > 0 && t.scalar_type() != x.scalar_type()) {
            return false;
        }
    }
    return true;
}

inline bool can_use_fused_cuda(
    const Tensor& x,
    bool fused_cuda,
    const std::vector<Tensor>& params = {}) {
#ifdef WITH_CUDA
    return fused_cuda && x.is_cuda() && all_same_dtype(x, params);
#else
    (void)x;
    (void)fused_cuda;
    (void)params;
    return false;
#endif
}

struct StateIntermediates {
    Tensor next_state;
    Tensor scaled_delta;
    Tensor value_pre;
};

// Packed state parameter order (created/cached by mlbricks.ffnbrick.native):
//  0 x_weight      [3*S, D] : candidate, write, value
//  1 x_bias        [3*S]
//  2 state_weight  [2*S, S] : candidate, write
//  3 esa_candidate_weight [S, D]
//  4 esa_write_weight     [S, D]
//  5 output_weight [D, S]
//  6 output_bias   [D]
//  7 depth_candidate [S]
//  8 depth_write     [S]
//  9 depth_value     [S]
// 10 retain_logit       [S]
// 11 read_logit         [S]
// 12 retain_delta_scale [S]
// 13 read_delta_scale   [S]
// 14 candidate_transition = sigmoid(raw scalar)
// 15 write_transition     = sigmoid(raw scalar)
// 16 delta_scale          = exp(raw scalar)
StateIntermediates state_update_packed_impl(
    const Tensor& x,
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& previous_state,
    const std::vector<Tensor>& p,
    bool fused_cuda) {
    TORCH_CHECK(
        p.size() == 17,
        "StateAware packed backend expected 17 tensors, got ",
        p.size());

    const auto state_dim = previous_state.size(-1);
    TORCH_CHECK(state_dim > 0, "state_dim must be positive");

    const auto& x_weight = p[0];
    const auto& x_bias = p[1];
    const auto& state_weight = p[2];
    const auto& esa_candidate_w = p[3];
    const auto& esa_write_w = p[4];
    const auto& depth_candidate = p[7];
    const auto& depth_write = p[8];
    const auto& depth_value = p[9];
    const auto& retain_logit = p[10];
    const auto& retain_delta_scale = p[12];
    const auto& candidate_transition = p[14];
    const auto& write_transition = p[15];
    const auto& delta_scale = p[16];

    TORCH_CHECK(x_weight.size(0) == 3 * state_dim, "packed x projection shape mismatch");
    TORCH_CHECK(state_weight.size(0) == 2 * state_dim, "packed state projection shape mismatch");

    // One GEMM replaces x_candidate + x_write + value.
    auto x_proj = linear_bias(x, x_weight, x_bias);
    auto x_candidate = x_proj.narrow(-1, 0, state_dim);
    auto x_write = x_proj.narrow(-1, state_dim, state_dim);
    auto value_pre = x_proj.narrow(-1, 2 * state_dim, state_dim) + depth_value;

    // One GEMM replaces state_candidate + state_write.
    auto state_proj = linear_nobias(previous_state, state_weight);
    auto state_candidate = state_proj.narrow(-1, 0, state_dim);
    auto state_write = state_proj.narrow(-1, state_dim, state_dim);

    Tensor candidate_esa;
    Tensor write_esa;
    Tensor scaled_delta;

#ifdef WITH_CUDA
    if (can_use_fused_cuda(x, fused_cuda, p)) {
        // One CUDA launch computes ESA delta, both transition-conditioned ESA
        // streams, and the row-wise FP32 delta magnitude.
        auto td = transition_delta_cuda(
            esa_update.contiguous(),
            previous_esa.contiguous(),
            candidate_transition.contiguous(),
            write_transition.contiguous(),
            delta_scale.contiguous());
        candidate_esa = td[0];
        write_esa = td[1];
        scaled_delta = td[2];
    } else
#endif
    {
        auto esa_delta = esa_update - previous_esa;
        auto delta_magnitude = esa_delta.to(at::kFloat)
                                   .square()
                                   .mean(-1, true)
                                   .add(1e-6)
                                   .sqrt()
                                   .to(esa_update.scalar_type());
        scaled_delta = delta_scale * delta_magnitude;
        candidate_esa = esa_update + candidate_transition * esa_delta;
        write_esa = esa_update + write_transition * esa_delta;
    }

    // The ESA branches retain their independent learned weights; these are the
    // only two model-width -> state-width GEMMs that cannot be safely packed
    // without changing the original equations.
    auto candidate_pre = x_candidate
        + linear_nobias(candidate_esa, esa_candidate_w)
        + state_candidate
        + depth_candidate;

    auto write_pre = x_write
        + linear_nobias(write_esa, esa_write_w)
        + state_write
        + depth_write;

    Tensor next_state;
#ifdef WITH_CUDA
    if (can_use_fused_cuda(x, fused_cuda, p)) {
        next_state = state_mix_cuda(
            candidate_pre.contiguous(),
            write_pre.contiguous(),
            previous_state.contiguous(),
            scaled_delta.contiguous(),
            retain_logit.contiguous(),
            retain_delta_scale.contiguous());
    } else
#endif
    {
        auto candidate = at::tanh(candidate_pre);
        auto write_gate = at::sigmoid(write_pre);
        auto retain_gate = at::sigmoid(
            retain_logit - scaled_delta * retain_delta_scale);
        next_state = (1.0 - write_gate) * (retain_gate * previous_state)
                   + write_gate * candidate;
    }

    return {next_state, scaled_delta, value_pre};
}

Tensor read_packed_impl(
    const Tensor& x,
    const StateIntermediates& s,
    const std::vector<Tensor>& p,
    bool fused_cuda) {
    const auto& output_w = p[5];
    const auto& output_b = p[6];
    const auto& read_logit = p[11];
    const auto& read_delta_scale = p[13];

    Tensor mixed;
#ifdef WITH_CUDA
    if (can_use_fused_cuda(x, fused_cuda, p)) {
        mixed = read_mix_cuda(
            s.value_pre.contiguous(),
            s.next_state.contiguous(),
            s.scaled_delta.contiguous(),
            read_logit.contiguous(),
            read_delta_scale.contiguous());
    } else
#endif
    {
        auto value = at::silu(s.value_pre);
        auto read_gate = at::sigmoid(
            read_logit + s.scaled_delta * read_delta_scale);
        mixed = s.next_state * value * read_gate;
    }

    return linear_bias(mixed, output_w, output_b);
}

Tensor micro_hidden_packed_2d(
    const Tensor& x2d,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    bool fused_cuda) {
    const auto hidden_dim = down_w.size(-1);
    TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
    TORCH_CHECK(gate_up_w.dim() == 2, "packed gate/up weight must be rank-2");
    TORCH_CHECK(gate_up_w.size(0) == 2 * hidden_dim, "packed gate/up shape mismatch");
    TORCH_CHECK(gate_up_w.size(1) == x2d.size(1), "packed gate/up input width mismatch");
    TORCH_CHECK(down_w.dim() == 2, "down weight must be rank-2");
    TORCH_CHECK(down_w.size(0) == x2d.size(1), "down projection output width mismatch");

    // Keep MicroVirtualFFN explicitly 2-D inside the native loop.  This avoids
    // repeated generic Linear shape handling for [...,D] tensors.
    auto gate_up = at::mm(x2d, gate_up_w.transpose(0, 1));

#ifdef WITH_CUDA
    std::vector<Tensor> params{gate_up_w, down_w};
    if (can_use_fused_cuda(x2d, fused_cuda, params)) {
        // gate_up is contiguous [rows,2H].  Consume the two packed halves
        // directly instead of narrow(...).contiguous() on gate and value.
        // The old native path launched two extra device copies per pass here.
        return silu_mul_packed_cuda(gate_up.contiguous(), hidden_dim);
    }
#endif

    auto gate = gate_up.narrow(-1, 0, hidden_dim);
    auto value = gate_up.narrow(-1, hidden_dim, hidden_dim);
    return at::silu(gate) * value;
}

Tensor micro_virtual_one_packed(
    const Tensor& x,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    bool fused_cuda) {
    TORCH_CHECK(x.dim() >= 1, "MicroVirtualFFN input must have at least one dimension");
    const auto d_model = x.size(-1);
    auto output_sizes = x.sizes().vec();
    auto x2d = x.reshape({-1, d_model});
    auto hidden2d = micro_hidden_packed_2d(x2d, gate_up_w, down_w, fused_cuda);
    auto out2d = at::mm(hidden2d, down_w.transpose(0, 1));
    return out2d.reshape(output_sizes);
}

} // namespace

Tensor micro_virtual_forward_packed(
    const Tensor& x,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    bool fused_cuda) {
    return micro_virtual_one_packed(x, gate_up_w, down_w, fused_cuda);
}

Tensor micro_virtual_refine_packed(
    const Tensor& x,
    const Tensor& gate_up_w,
    const Tensor& down_w,
    int64_t refinements,
    bool fused_cuda) {
    TORCH_CHECK(refinements > 0, "refinements must be positive");
    TORCH_CHECK(gate_up_w.dim() == 3, "gate_up_w must have [R,2H,D] shape");
    TORCH_CHECK(down_w.dim() == 3, "down_w must have [R,D,H] shape");
    TORCH_CHECK(gate_up_w.size(0) >= refinements && down_w.size(0) >= refinements,
                "packed MicroVirtualFFN refinement count mismatch");
    TORCH_CHECK(x.dim() >= 1, "MicroVirtualFFN input must have at least one dimension");

    const auto d_model = x.size(-1);
    auto output_sizes = x.sizes().vec();
    auto refined2d = x.reshape({-1, d_model});

    for (int64_t i = 0; i < refinements; ++i) {
        const auto gate_up_i = gate_up_w.select(0, i);
        const auto down_i = down_w.select(0, i);
        auto hidden2d = micro_hidden_packed_2d(
            refined2d,
            gate_up_i,
            down_i,
            fused_cuda);

        // addmm computes refined + hidden @ down^T in the GEMM itself.  The
        // previous native path materialized the down projection and then
        // launched a separate residual-add kernel for every refinement.
        refined2d = at::addmm(
            refined2d,
            hidden2d,
            down_i.transpose(0, 1));
    }

    return refined2d.reshape(output_sizes);
}

std::vector<Tensor> state_aware_forward_packed(
    const Tensor& x,
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& previous_state,
    const std::vector<Tensor>& p,
    bool fused_cuda) {
    auto s = state_update_packed_impl(
        x, esa_update, previous_esa, previous_state, p, fused_cuda);
    auto update = read_packed_impl(x, s, p, fused_cuda);
    return {update, s.next_state};
}

std::vector<Tensor> virtual_state_aware_forward_packed(
    const Tensor& x,
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& previous_state,
    const std::vector<Tensor>& p,
    const std::vector<Tensor>& vp,
    int64_t refinements,
    double rms_eps,
    bool fused_cuda) {
    TORCH_CHECK(vp.size() == 9, "Virtual packed backend expected 9 tensors, got ", vp.size());
    TORCH_CHECK(refinements > 0, "refinements must be positive");

    // vp: norm_w, state_up_w, state_up_b, x_cond_w, esa_cond_w,
    //     down_w, down_b, pass_embedding, gate_logit
    const auto& norm_w = vp[0];
    const auto& state_up_w = vp[1];
    const auto& state_up_b = vp[2];
    const auto& x_cond_w = vp[3];
    const auto& esa_cond_w = vp[4];
    const auto& down_w = vp[5];
    const auto& down_b = vp[6];
    const auto& pass_embedding = vp[7];
    const auto& gate_logit = vp[8];

    auto s = state_update_packed_impl(
        x, esa_update, previous_esa, previous_state, p, fused_cuda);

    // These conditions are shared across every virtual pass exactly as in the
    // original PyTorch implementation.
    auto x_condition = linear_nobias(x, x_cond_w);
    auto esa_condition = linear_nobias(esa_update, esa_cond_w);

    auto state = s.next_state;
    std::vector<Tensor> all_params = p;
    all_params.insert(all_params.end(), vp.begin(), vp.end());

    for (int64_t i = 0; i < refinements; ++i) {
        Tensor normed;
#ifdef WITH_CUDA
        const bool use_cuda_fusion = can_use_fused_cuda(x, fused_cuda, all_params);
        if (use_cuda_fusion) {
            normed = rms_norm_cuda(state.contiguous(), norm_w.contiguous(), rms_eps);
        } else
#endif
        {
            auto state_f = state.to(at::kFloat);
            auto scale = state_f.square().mean(-1, true).add(rms_eps).rsqrt();
            normed = (state_f * scale).to(state.scalar_type()) * norm_w;
        }

        auto state_hidden = linear_bias(normed, state_up_w, state_up_b);
        Tensor hidden;
#ifdef WITH_CUDA
        if (use_cuda_fusion) {
            hidden = condition_silu_cuda(
                state_hidden.contiguous(),
                x_condition.contiguous(),
                esa_condition.contiguous(),
                pass_embedding.select(0, i).contiguous());
        } else
#endif
        {
            hidden = at::silu(
                state_hidden
                + x_condition
                + esa_condition
                + pass_embedding.select(0, i));
        }

        auto delta = linear_bias(hidden, down_w, down_b);

#ifdef WITH_CUDA
        if (use_cuda_fusion) {
            state = residual_gate_cuda(
                state.contiguous(),
                delta.contiguous(),
                gate_logit.select(0, i).contiguous());
        } else
#endif
        {
            state = state + at::sigmoid(gate_logit.select(0, i)) * delta;
        }
    }

    s.next_state = state;
    auto update = read_packed_impl(x, s, p, fused_cuda);
    return {update, state};
}

} // namespace ffnbrick
