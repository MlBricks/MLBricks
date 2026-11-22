// SWQR-Short Python extension. It executes a real CUDA kernel, not a fallback.
#include <torch/extension.h>
#include <cstdint>

torch::Tensor short_scan_cuda(torch::Tensor A, torch::Tensor U,
                                  torch::Tensor state, int64_t warps);

torch::Tensor scan(torch::Tensor A, torch::Tensor U,
                   torch::Tensor state, int64_t warps = 8) {
    TORCH_CHECK(A.is_cuda() && U.is_cuda() && state.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(A.device() == U.device() && A.device() == state.device(), "Device mismatch");
    TORCH_CHECK(A.dim() == 4 && A.sizes() == U.sizes(), "A/U require matching [B,T,H,Dh]");
    TORCH_CHECK(A.scalar_type() == at::kFloat && U.scalar_type() == at::kFloat,
                "A/U must be FP32");
    TORCH_CHECK(A.is_contiguous() && U.is_contiguous(), "A/U must be contiguous");
    TORCH_CHECK(state.scalar_type() == at::kFloat && state.is_contiguous(),
                "Cached state must be contiguous FP32");
    TORCH_CHECK(state.numel() == 0 || (state.dim() == 3 && state.size(0) == A.size(0)
                && state.size(1) == A.size(2) && state.size(2) == A.size(3)),
                "State must have shape [B,H,Dh] or be empty");
    TORCH_CHECK(warps == 4 || warps == 8 || warps == 16, "warps must be 4/8/16");
    return short_scan_cuda(A, U, state, warps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scan", &scan, "SWQR-Short one-launch block-parallel FP32 scan",
          pybind11::arg("a"), pybind11::arg("u"), pybind11::arg("state"),
          pybind11::arg("warps") = 8);
}
