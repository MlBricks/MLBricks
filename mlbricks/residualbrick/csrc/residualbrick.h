#pragma once

#include <torch/extension.h>

namespace residualbrick {

using at::Tensor;

Tensor residual_controller_aten(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps);

Tensor residual_forward(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps,
    bool fused_cuda);

#ifdef WITH_CUDA
Tensor residual_controller_cuda(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps);
#endif

} // namespace residualbrick
