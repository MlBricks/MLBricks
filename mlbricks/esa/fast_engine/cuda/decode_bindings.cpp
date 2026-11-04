// PyTorch binding for the original Half-Life Quality ESA inference-only one-token fusion.
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> fused_decode_cuda(
    torch::Tensor qgv,torch::Tensor x,torch::Tensor log_half,
    torch::Tensor decay_bias,torch::Tensor norm_weight,
    torch::Tensor old_state,int64_t heads);
std::vector<torch::Tensor> forward(
    torch::Tensor qgv,torch::Tensor x,torch::Tensor log_half,
    torch::Tensor decay_bias,torch::Tensor norm_weight,
    torch::Tensor old_state,int64_t heads) {
  TORCH_CHECK(x.is_cuda() && qgv.is_cuda() && log_half.is_cuda() &&
              decay_bias.is_cuda() && norm_weight.is_cuda() && old_state.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(x.device()==qgv.device() && x.device()==log_half.device() &&
              x.device()==decay_bias.device() && x.device()==norm_weight.device() &&
              x.device()==old_state.device(),"All input tensors must share device");
  TORCH_CHECK(x.dim()==3 && x.size(1)==1,"x must be [B,1,D]");
  TORCH_CHECK(qgv.dim()==3 && qgv.size(0)==x.size(0) && qgv.size(1)==1 &&
              qgv.size(2)==3*x.size(2),"qgv must be [B,1,3D]");
  TORCH_CHECK(heads>0 && x.size(2)%heads==0 && x.size(0)<=65535,
              "Invalid heads, D, or batch size");
  TORCH_CHECK(log_half.dim()==1 && decay_bias.dim()==1 &&
              log_half.numel()==heads && decay_bias.numel()==heads,
              "Half lives/decay bias shape must be [H]");
  TORCH_CHECK(norm_weight.dim()==1 && norm_weight.numel()==x.size(2),
              "state norm weight must be [D]");
  TORCH_CHECK(log_half.scalar_type()==at::kFloat &&
              decay_bias.scalar_type()==at::kFloat &&
              norm_weight.scalar_type()==at::kFloat &&
              old_state.scalar_type()==at::kFloat,"quality state/parameters must be FP32");
  TORCH_CHECK(old_state.numel()==0 || (old_state.dim()==3 &&
              old_state.size(0)==x.size(0) && old_state.size(1)==heads &&
              old_state.size(2)==x.size(2)/heads),"State must be empty or [B,H,D/H]");
  TORCH_CHECK(qgv.is_contiguous() && log_half.is_contiguous() &&
              decay_bias.is_contiguous() && norm_weight.is_contiguous() &&
              old_state.is_contiguous(),"Contiguous inputs required");
  TORCH_CHECK(x.scalar_type()==at::kFloat || x.scalar_type()==at::kHalf ||
              x.scalar_type()==at::kBFloat16,"x dtype must be fp32/fp16/bf16");
  TORCH_CHECK(qgv.scalar_type()==at::kFloat || qgv.scalar_type()==at::kHalf ||
              qgv.scalar_type()==at::kBFloat16,"qgv dtype must be fp32/fp16/bf16");
  return fused_decode_cuda(qgv,x,log_half,decay_bias,norm_weight,old_state,heads);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("forward",&forward,"Fused half-life Quality ESA one-token readout and state update");
}
