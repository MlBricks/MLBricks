#include <torch/extension.h>
#include <vector>
torch::Tensor avq_segmented_forward(torch::Tensor a,torch::Tensor v,torch::Tensor s0);
std::vector<torch::Tensor> avq_segmented_backward(torch::Tensor a,torch::Tensor states,torch::Tensor s0,torch::Tensor ds);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &avq_segmented_forward, "BOLT AVQ segmented affine scan forward");
  m.def("backward", &avq_segmented_backward, "BOLT AVQ segmented affine scan backward");
}
