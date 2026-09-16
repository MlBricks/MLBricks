#include "residualbrick.h"

#include <ATen/ATen.h>

namespace residualbrick {
namespace {

void validate_inputs(const Tensor& residual, const Tensor& update) {
    TORCH_CHECK(
        residual.sizes() == update.sizes(),
        "residual and update must have identical shapes; got ",
        residual.sizes(), " and ", update.sizes());
    TORCH_CHECK(
        residual.is_floating_point() && update.is_floating_point(),
        "residual and update must be floating-point tensors");
    TORCH_CHECK(
        residual.device() == update.device(),
        "residual and update must be on the same device; got ",
        residual.device(), " and ", update.device());
}

Tensor rms_last_dim(const Tensor& x, double eps) {
    // Keep the reference implementation's FP32 accumulation semantics.
    return at::sqrt(at::mean(at::square(x), {-1}, true) + eps);
}

} // namespace

Tensor residual_controller_aten(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps) {
    validate_inputs(residual, update);

    const auto residual32 = residual.to(at::kFloat);
    const auto update32 = update.to(at::kFloat);

    const auto residual_rms = rms_last_dim(residual32, eps);
    const auto raw_update_rms = rms_last_dim(update32, eps);

    const auto allowed_update_rms = residual_rms * update_ratio;
    const auto hard_update_scale = at::clamp_max(
        allowed_update_rms / (raw_update_rms + eps), 1.0);
    const auto update_pressure = raw_update_rms / (allowed_update_rms + eps);
    const auto update_gate = at::sigmoid(
        (update_pressure - 1.0) * update_softness);
    const auto update_scale = 1.0 - update_gate * (1.0 - hard_update_scale);
    const auto bounded_update = update32 * update_scale;

    const auto candidate = residual32 + bounded_update;
    const auto candidate_rms = rms_last_dim(candidate, eps);
    const auto allowed_stream_rms = residual_rms * stream_ratio;
    const auto hard_stream_scale = at::clamp_max(
        allowed_stream_rms / (candidate_rms + eps), 1.0);
    const auto stream_pressure = candidate_rms / (allowed_stream_rms + eps);
    const auto stream_gate = at::sigmoid(
        (stream_pressure - 1.0) * stream_softness);
    const auto stream_scale = 1.0 - stream_gate * (1.0 - hard_stream_scale);

    const auto final_update = bounded_update * stream_scale;
    return (residual32 + final_update).to(residual.scalar_type());
}

Tensor residual_forward(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps,
    bool fused_cuda) {
    validate_inputs(residual, update);

#ifdef WITH_CUDA
    // The fused kernel is an inference fast path. It uses FP32 reduction/math
    // but does not register a custom autograd formula; training therefore stays
    // on the ATen implementation below.
    if (fused_cuda && residual.is_cuda() && update.is_cuda() &&
        residual.scalar_type() == update.scalar_type() &&
        residual.dim() > 0 && residual.size(-1) > 0) {
        return residual_controller_cuda(
            residual, update,
            update_ratio, stream_ratio,
            update_softness, stream_softness, eps);
    }
#else
    (void)fused_cuda;
#endif

    return residual_controller_aten(
        residual, update,
        update_ratio, stream_ratio,
        update_softness, stream_softness, eps);
}

} // namespace residualbrick
