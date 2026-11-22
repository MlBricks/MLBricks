// SWQR FastScan — inference-only CUDA operator exposed to Python.
// A custom PyTorch reverse-scan gradient is implemented separately.
#include <torch/extension.h>
#include <cstdint>

torch::Tensor parallel_scan_cuda(torch::Tensor a, torch::Tensor u,
                                  torch::Tensor state, int64_t chunk);

torch::Tensor scan(torch::Tensor a, torch::Tensor u,
                   torch::Tensor state, int64_t chunk=128) {
  TORCH_CHECK(a.is_cuda() && u.is_cuda(), "CUDA A/U tensors required");
  TORCH_CHECK(a.device()==u.device(), "Device mismatch");
  TORCH_CHECK(a.dim()==4 && a.sizes()==u.sizes(), "A/U shape [B,T,H,Dh] required");
  TORCH_CHECK(a.is_contiguous() && u.is_contiguous(), "A/U must be contiguous");
  TORCH_CHECK(a.scalar_type()==at::kFloat && u.scalar_type()==at::kFloat, "FP32 inputs required");
  TORCH_CHECK(chunk>=16 && chunk<=512 && (chunk&(chunk-1))==0,
              "chunk must be power of two in [16,512]");
  TORCH_CHECK(state.is_cuda() && state.device()==a.device(), "State device mismatch");
  TORCH_CHECK(state.scalar_type()==at::kFloat && state.is_contiguous(), "State must be contiguous FP32");
  TORCH_CHECK(state.numel()==0 || (state.dim()==3 && state.size(0)==a.size(0)
         && state.size(1)==a.size(2) && state.size(2)==a.size(3)), "State must be [B,H,Dh] or empty");
  return parallel_scan_cuda(a,u,state,chunk);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("scan",&scan,"SWQR FP32 chunked FastScan (forward only)",
        pybind11::arg("a"),pybind11::arg("u"),pybind11::arg("state"),
        pybind11::arg("chunk")=128);
}
