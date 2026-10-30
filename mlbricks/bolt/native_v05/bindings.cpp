#include <torch/extension.h>

void gate_out(torch::Tensor packed, torch::Tensor out);
void avq_out(torch::Tensor raw, torch::Tensor state,
             torch::Tensor features, double eps);

void avq_state_only(torch::Tensor raw, torch::Tensor state);

void avq_readout_pt(torch::Tensor state, torch::Tensor r,
                    torch::Tensor inv, torch::Tensor features);
void avq_readout_raw(torch::Tensor state, torch::Tensor raw,
                     torch::Tensor inv, torch::Tensor features);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_out", &gate_out, "BOLT v0.5 fused U/G decode gate (CUDA)");
    m.def("avq_out", &avq_out, "BOLT v0.5 fused AVQ state and read gate (CUDA)");
    m.def("avq_state_only", &avq_state_only, "BOLT v0.5g recurrence-only AVQ CUDA");
    m.def("avq_readout_pt", &avq_readout_pt, "v0.5h reference-ordered FP32 readout (precomputed torch.tanh)");
    m.def("avq_readout_raw", &avq_readout_raw, "v0.5h FP32 readout with native FP16 tanh gate");
}
