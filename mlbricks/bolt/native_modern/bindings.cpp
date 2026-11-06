// BOLT native CUDA extension. The legacy rho-normalized BOLT extension is distinct.
// Copyright 2026 MLBricks contributors. See repository licensing notices.
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> bolt_features_forward_cuda(torch::Tensor q, torch::Tensor u, torch::Tensor g);
torch::Tensor avq_scan_forward_cuda(torch::Tensor a, torch::Tensor v);
std::vector<torch::Tensor> avq_scan_backward_cuda(torch::Tensor grad_s, torch::Tensor a, torch::Tensor s);
torch::Tensor avq_decode_state_cuda(torch::Tensor a, torch::Tensor v, torch::Tensor previous);
torch::Tensor bolt_decode_append_cuda(torch::Tensor q, torch::Tensor c_new, torch::Tensor cache,
                                       torch::Tensor scores, int64_t position, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("features_forward", &bolt_features_forward_cuda, "Direct-Q gated BOLT features CUDA (no learned P)");
    m.def("avq_scan_forward", &avq_scan_forward_cuda, "AVQ recurrent scan forward CUDA");
    m.def("avq_scan_backward", &avq_scan_backward_cuda, "AVQ recurrent scan backward CUDA");
    m.def("avq_decode_state", &avq_decode_state_cuda, "AVQ recurrent decode step CUDA");
    m.def("decode_append", &bolt_decode_append_cuda, "BOLT fused cache append and attention decode CUDA");
}
