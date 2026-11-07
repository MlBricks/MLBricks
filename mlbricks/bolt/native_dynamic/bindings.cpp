#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Custom native CUDA decode with a GPU-resident, mutable position tensor.
// No Python position is captured: replay loads the CURRENT value on the GPU.
torch::Tensor bolt_dynamic_cuda(torch::Tensor q, torch::Tensor current,
    torch::Tensor cache, torch::Tensor scratch, torch::Tensor position,
    double scale, int64_t chunk_tokens, int64_t max_chunks, bool use_split);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &bolt_dynamic_cuda,
          "Dynamic-position graph-safe BOLT decode (FP32 Q, FP16/FP32 C)");
}
