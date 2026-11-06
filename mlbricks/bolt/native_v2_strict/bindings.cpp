#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/linear.h>
#include <c10/cuda/CUDAGuard.h>

void v2_launch(const at::Tensor& o, const at::Tensor& z, at::Tensor& state,
               const at::Tensor& norm_weight, const at::Tensor& gamma, double eps);

static int validate(const at::Tensor& o, const at::Tensor& state,
                    const at::Tensor& norm_weight, const at::Tensor& gamma, double eps) {
    TORCH_CHECK(o.is_cuda() && state.is_cuda() && norm_weight.is_cuda() && gamma.is_cuda(),
                "CUDA tensors required");
    TORCH_CHECK(o.device()==state.device() && o.device()==norm_weight.device() &&
                o.device()==gamma.device(), "all input tensors must be on the same GPU");
    TORCH_CHECK(o.is_contiguous() && state.is_contiguous() && norm_weight.is_contiguous() &&
                gamma.is_contiguous(), "CUDA tensors must be contiguous");
    TORCH_CHECK(o.scalar_type()==at::kHalf && state.scalar_type()==at::kFloat,
                "o must be FP16 and state FP32");
    TORCH_CHECK(norm_weight.scalar_type()==at::kFloat && gamma.scalar_type()==at::kFloat,
                "norm/gamma must be FP32");
    TORCH_CHECK(o.dim()==3 && o.size(0)==1 && o.size(1)==1,
                "o must have shape [1,1,W]");
    TORCH_CHECK(state.dim()==2 && state.size(0)==1 && state.size(1)==o.size(2),
                "state must have shape [1,W]");
    int width=static_cast<int>(o.size(2));
    TORCH_CHECK(width==192 || width==384,"only tested widths 192 or 384");
    TORCH_CHECK(norm_weight.numel()==width && gamma.numel()==1,
                "norm/gamma shape mismatch");
    TORCH_CHECK(eps>0.0 && eps<0.1,"invalid eps");
    return width;
}

// The reference projection route: PyTorch computes Z; this native kernel updates in place.
void inplace_strict(at::Tensor o, const at::Tensor& z, at::Tensor state,
                    const at::Tensor& norm_weight, const at::Tensor& gamma, double eps) {
    int width=validate(o,state,norm_weight,gamma,eps);
    TORCH_CHECK(z.is_cuda() && z.device()==o.device() && z.is_contiguous() &&
                z.scalar_type()==at::kHalf && z.numel()==3*width,"Z must be FP16 CUDA 3W");
    c10::cuda::CUDAGuard guard(o.device());
    v2_launch(o,z,state,norm_weight,gamma,eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inplace_strict", &inplace_strict,
          "BOLT V2 strict in-place AVQ post-projection update (FP16 input, FP32 state)");
}
