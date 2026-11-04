// Validated native binding for BaseESA one-token inference. Separate from original kernels.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_base_decode_cuda(
    torch::Tensor qgv, torch::Tensor x,
    torch::Tensor raw_alpha, torch::Tensor raw_beta,
    torch::Tensor old_state, int64_t heads);

std::vector<torch::Tensor> forward(
    torch::Tensor qgv, torch::Tensor x,
    torch::Tensor raw_alpha, torch::Tensor raw_beta,
    torch::Tensor old_state, int64_t heads) {
  TORCH_CHECK(qgv.is_cuda() && x.is_cuda() && raw_alpha.is_cuda() &&
      raw_beta.is_cuda() && old_state.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.device()==qgv.device() && x.device()==raw_alpha.device() &&
      x.device()==raw_beta.device() && x.device()==old_state.device(),
      "All tensors must be on the same CUDA device");
  TORCH_CHECK(x.dim()==3 && x.size(1)==1 && x.size(2)>0, "x must be [B,1,D]");
  TORCH_CHECK(qgv.dim()==3 && qgv.size(0)==x.size(0) && qgv.size(1)==1 &&
      qgv.size(2)==3*x.size(2), "qgv must be [B,1,3D]");
  TORCH_CHECK(heads>0 && x.size(2)%heads==0 && x.size(0)<=65535,
      "Invalid head count, dimension, or batch size");
  TORCH_CHECK(raw_alpha.dim()==1 && raw_beta.dim()==1 &&
      raw_alpha.numel()==heads && raw_beta.numel()==heads,
      "Learned bounds parameters must each have shape [H]");
  TORCH_CHECK(raw_alpha.scalar_type()==at::kFloat &&
      raw_beta.scalar_type()==at::kFloat &&
      old_state.scalar_type()==at::kFloat, "Learned bounds and recurrent state must be FP32");
  TORCH_CHECK(old_state.numel()==0 || (old_state.dim()==3 &&
      old_state.size(0)==x.size(0) && old_state.size(1)==heads &&
      old_state.size(2)==x.size(2)/heads), "State must be empty or [B,H,D/H]");
  TORCH_CHECK(qgv.is_contiguous() && raw_alpha.is_contiguous() &&
      raw_beta.is_contiguous() && old_state.is_contiguous(),
      "Contiguous inputs required");
  TORCH_CHECK(qgv.scalar_type()==at::kFloat || qgv.scalar_type()==at::kHalf ||
      qgv.scalar_type()==at::kBFloat16, "qgv must be FP32/FP16/BF16");
  TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf ||
      x.scalar_type()==at::kBFloat16, "x must be FP32/FP16/BF16");
  return fused_base_decode_cuda(qgv, x, raw_alpha, raw_beta, old_state, heads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Fused learned-bounds BaseESA one-token state update/readout");
}
