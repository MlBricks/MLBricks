
#include <torch/extension.h>

torch::Tensor baseline_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale,
    int64_t mode,
    int64_t splits);

torch::Tensor gauss_decode_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    double scale,
    int64_t mode,
    int64_t splits);

std::vector<torch::Tensor> gauss_gate_rho_cuda(
    torch::Tensor u,
    torch::Tensor g,
    double eps);

std::vector<torch::Tensor> gauss_stage1_forward_cuda(
    torch::Tensor qcg,
    int64_t heads,
    int64_t latent,
    double eps);

torch::Tensor gauss_stage1_backward_cuda(
    torch::Tensor dq,
    torch::Tensor dc,
    torch::Tensor drho,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor gate,
    int64_t heads,
    int64_t latent);

void baseline_decode_out_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits);

void gauss_decode_out_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits);


void baseline_decode_out_used_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t used);

void gauss_decode_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t used);

// Inference-only direct projection APIs. They merge the exact split-softmax
// summaries into on-chip O and immediately apply W_O, so O is never written
// to global memory. PyTorch Linear bias is intentionally unsupported here;
// bias=True falls back to the existing O + out_proj path in Python.
torch::Tensor gauss_decode_project_out_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits);

torch::Tensor gauss_decode_project_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits, int64_t used);

torch::Tensor gauss_decode_append_project_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits, int64_t position);


void baseline_append_cache_cuda(
    torch::Tensor k_now, torch::Tensor v_now,
    torch::Tensor k_cache, torch::Tensor v_cache, int64_t position);

void gauss_append_cache_cuda(
    torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache, int64_t position);

void baseline_decode_append_out_cuda(
    torch::Tensor q, torch::Tensor k_now, torch::Tensor v_now,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t position);

void gauss_decode_append_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t position);


void gauss_rope_decode_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t used,
    double rope_base, int64_t rope_dim);

void gauss_rope_decode_append_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po, torch::Tensor out,
    double scale, int64_t mode, int64_t splits, int64_t position,
    double rope_base, int64_t rope_dim);

torch::Tensor gauss_rope_decode_project_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits, int64_t used,
    double rope_base, int64_t rope_dim);

torch::Tensor gauss_rope_decode_append_project_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits, int64_t position,
    double rope_base, int64_t rope_dim);

void gauss_unpack_gate_rho_out_cuda(
    torch::Tensor qcg,
    torch::Tensor q_out,
    torch::Tensor c_out,
    torch::Tensor rho_out,
    double eps);

void baseline_unpack_qkv_out_cuda(
    torch::Tensor qkv,
    torch::Tensor q_out,
    torch::Tensor k_now,
    torch::Tensor v_now);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gauss_r16_scan_supported", []() { return true; });
    m.def("baseline_decode", &baseline_decode_cuda);
    m.def("gauss_decode", &gauss_decode_cuda);
    m.def("gauss_gate_rho", &gauss_gate_rho_cuda);
    m.def("gauss_stage1_forward", &gauss_stage1_forward_cuda);
    m.def("gauss_stage1_backward", &gauss_stage1_backward_cuda);

    m.def("baseline_decode_out", &baseline_decode_out_cuda);
    m.def("gauss_decode_out", &gauss_decode_out_cuda);
    m.def("baseline_decode_out_used", &baseline_decode_out_used_cuda);
    m.def("gauss_decode_out_used", &gauss_decode_out_used_cuda);
    m.def("gauss_decode_project_out", &gauss_decode_project_out_cuda);
    m.def("gauss_decode_project_out_used", &gauss_decode_project_out_used_cuda);
    m.def("gauss_decode_append_project_out", &gauss_decode_append_project_out_cuda);
    m.def("baseline_append_cache", &baseline_append_cache_cuda);
    m.def("gauss_append_cache", &gauss_append_cache_cuda);
    m.def("baseline_decode_append_out", &baseline_decode_append_out_cuda);
    m.def("gauss_decode_append_out", &gauss_decode_append_out_cuda);
    m.def("gauss_rope_decode_out_used", &gauss_rope_decode_out_used_cuda);
    m.def("gauss_rope_decode_append_out", &gauss_rope_decode_append_out_cuda);
    m.def("gauss_rope_decode_project_out_used", &gauss_rope_decode_project_out_used_cuda);
    m.def("gauss_rope_decode_append_project_out", &gauss_rope_decode_append_project_out_cuda);
    m.def("gauss_unpack_gate_rho_out", &gauss_unpack_gate_rho_out_cuda);
    m.def("baseline_unpack_qkv_out", &baseline_unpack_qkv_out_cuda);
}
