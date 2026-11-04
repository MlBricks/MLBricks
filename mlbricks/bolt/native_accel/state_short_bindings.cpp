#include <torch/extension.h>
#include <vector>
// Input za,zv: strided slices of [B,T,3D] (FP16 or FP32), FP32 initial state.
// Return FP32 E, and gradients da,dv in input dtype plus FP32 initial-state grad.
torch::Tensor avq_v7_forward(torch::Tensor za, torch::Tensor zv, torch::Tensor s0, int64_t warps, bool short_path);
std::vector<torch::Tensor> avq_v7_backward(torch::Tensor za, torch::Tensor zv, torch::Tensor states, torch::Tensor s0, torch::Tensor de, int64_t warps, bool short_path);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("forward", &avq_v7_forward, "BOLT AVQ v7 mixed-precision direct forward");
 m.def("backward", &avq_v7_backward, "BOLT AVQ v7 mixed-precision direct backward");
}
