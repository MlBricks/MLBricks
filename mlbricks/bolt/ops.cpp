// Copyright 2026 Zameer Hussain and Akhtar Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/matmul.h>

#include <cstdint>
#include <limits>
#include <vector>

#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#endif

namespace mlbricks {

torch::Tensor thunder_scan_cpu(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass);

torch::Tensor lightning_step_cpu(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state);

std::vector<torch::Tensor> residual_layer_norm_cpu(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps);


#ifdef WITH_CUDA
torch::Tensor thunder_fused_readout_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);
#endif

#ifdef WITH_CUDA
torch::Tensor thunder_fused_readout_hierarchical_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);
#endif

#ifdef WITH_CUDA
torch::Tensor thunder_fused_readout_hierarchical_mixed32_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);

torch::Tensor thunder_fused_readout_hierarchical_full32_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);
#endif

#ifdef WITH_CUDA
std::vector<torch::Tensor> thunder_prepare_ab_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max);

torch::Tensor thunder_readout_cuda(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps);
#endif


#ifdef WITH_CUDA
std::vector<torch::Tensor> thunder_prepare_ab_precise_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max);

torch::Tensor thunder_readout_precise_cuda(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps);

torch::Tensor thunder_fused_readout_hierarchical_precise_gate_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);

torch::Tensor thunder_fused_readout_hierarchical_precise_readout_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);

torch::Tensor thunder_fused_readout_hierarchical_precise_both_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass);
#endif

#ifdef WITH_CUDA
torch::Tensor thunder_scan_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass);

torch::Tensor thunder_scan_hierarchical_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass);

std::vector<torch::Tensor> thunder_scan_local_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass);

std::vector<torch::Tensor> thunder_summary_scan_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t group_size);

std::vector<torch::Tensor> thunder_group_prefix_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write);

std::vector<torch::Tensor> thunder_apply_group_cuda(
    const torch::Tensor& pref_A,
    const torch::Tensor& pref_B,
    const torch::Tensor& parent_A,
    const torch::Tensor& parent_B,
    int64_t group_size);

torch::Tensor thunder_apply_chunk_prefix_cuda(
    const torch::Tensor& A,
    const torch::Tensor& local_states,
    const torch::Tensor& chunk_B_prefix,
    int64_t compass);

std::vector<torch::Tensor> thunder_scan_backward_chunked_cuda(
    const torch::Tensor& A,
    const torch::Tensor& states,
    const torch::Tensor& grad,
    int64_t compass);

std::vector<torch::Tensor> thunder_reverse_prepare_cuda(
    const torch::Tensor& A,
    const torch::Tensor& grad);

std::vector<torch::Tensor> thunder_reverse_finish_cuda(
    const torch::Tensor& grad_reverse,
    const torch::Tensor& states);

torch::Tensor lightning_step_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state);

std::vector<torch::Tensor> lightning_fused_step_cuda(
    const torch::Tensor& qgv,
    const torch::Tensor& state,
    double gate_min,
    double gate_max,
    double eps);

std::vector<torch::Tensor> residual_layer_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps);

torch::Tensor elastic_linear_packed_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& bias,
    int64_t bits,
    int64_t group_size,
    int64_t out_features,
    int64_t in_features);

torch::Tensor linear_hybrid_accum_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    int64_t chunk);

#endif

static void check_scan_inputs(
    const torch::Tensor& A,
    const torch::Tensor& B_write) {
    TORCH_CHECK(A.defined() && B_write.defined(), "A and B_write must be defined");
    TORCH_CHECK(A.sizes() == B_write.sizes(), "A and B_write must have the same shape");
    TORCH_CHECK(A.dim() == 4, "expected A/B_write shape [B,T,H,D]");
    TORCH_CHECK(A.scalar_type() == B_write.scalar_type(), "A and B_write must have the same dtype");
    TORCH_CHECK(A.device() == B_write.device(), "A and B_write must be on the same device");
    TORCH_CHECK(A.is_contiguous() && B_write.is_contiguous(), "A and B_write must be contiguous");
    TORCH_CHECK(A.is_floating_point(), "A and B_write must be floating point");
}

static void check_step_inputs(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state) {
    TORCH_CHECK(A.defined() && B_write.defined() && state.defined(), "A, B_write, and state must be defined");
    TORCH_CHECK(A.sizes() == B_write.sizes(), "A and B_write must have the same shape");
    TORCH_CHECK(A.sizes() == state.sizes(), "A, B_write, and state must have the same shape");
    TORCH_CHECK(A.dim() == 3, "expected A/B_write/state shape [B,H,D]");
    TORCH_CHECK(A.scalar_type() == B_write.scalar_type() && A.scalar_type() == state.scalar_type(),
                "A, B_write, and state must have the same dtype");
    TORCH_CHECK(A.device() == B_write.device() && A.device() == state.device(),
                "A, B_write, and state must be on the same device");
    TORCH_CHECK(A.is_contiguous() && B_write.is_contiguous() && state.is_contiguous(),
                "A, B_write, and state must be contiguous");
    TORCH_CHECK(A.is_floating_point(), "A, B_write, and state must be floating point");
}


torch::Tensor thunder_fused_readout(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(qgv.defined(), "qgv must be defined");
    TORCH_CHECK(qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0, "embd must be positive");
    TORCH_CHECK(qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    TORCH_CHECK(compass > 0, "compass must be positive");
    TORCH_CHECK(qgv.is_contiguous(), "qgv must be contiguous");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "fused Thunder currently supports FP16/BF16 qgv tensors");
    TORCH_CHECK(qgv.is_cuda(), "fused Thunder readout is CUDA-only");
#ifdef WITH_CUDA
    return thunder_fused_readout_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_fused_readout_hierarchical(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(qgv.defined(), "qgv must be defined");
    TORCH_CHECK(qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0, "embd must be positive");
    TORCH_CHECK(qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    TORCH_CHECK(compass > 0, "compass must be positive");
    TORCH_CHECK(qgv.is_contiguous(), "qgv must be contiguous");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "hierarchical fused Thunder currently supports FP16/BF16 qgv tensors");
    TORCH_CHECK(qgv.is_cuda(), "hierarchical fused Thunder readout is CUDA-only");
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}


torch::Tensor thunder_fused_readout_hierarchical_mixed32(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(qgv.defined(), "qgv must be defined");
    TORCH_CHECK(qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0 && qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0 && compass > 0, "eps and compass must be positive");
    TORCH_CHECK(qgv.is_contiguous(), "qgv must be contiguous");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "v8 mixed32 supports FP16/BF16 qgv tensors");
    TORCH_CHECK(qgv.is_cuda(), "v8 mixed32 readout is CUDA-only");
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_mixed32_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}


torch::Tensor thunder_fused_readout_hierarchical_full32(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(qgv.defined(), "qgv must be defined");
    TORCH_CHECK(qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0 && qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0 && compass > 0, "eps and compass must be positive");
    TORCH_CHECK(qgv.is_contiguous(), "qgv must be contiguous");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "v8 full32 supports FP16/BF16 qgv tensors");
    TORCH_CHECK(qgv.is_cuda(), "v8 full32 readout is CUDA-only");
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_full32_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}



static void check_v10_fused_inputs(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(qgv.defined() && qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0 && qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0 && compass > 0, "eps and compass must be positive");
    TORCH_CHECK(qgv.is_cuda() && qgv.is_contiguous(),
                "v10 precise fused operators require contiguous CUDA qgv");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "v10 precise fused operators support FP16/BF16 tensors");
}

torch::Tensor thunder_fused_readout_hierarchical_precise_gate(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    check_v10_fused_inputs(qgv, embd, gate_min, gate_max, eps, compass);
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_precise_gate_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_fused_readout_hierarchical_precise_readout(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    check_v10_fused_inputs(qgv, embd, gate_min, gate_max, eps, compass);
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_precise_readout_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_fused_readout_hierarchical_precise_both(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    check_v10_fused_inputs(qgv, embd, gate_min, gate_max, eps, compass);
#ifdef WITH_CUDA
    return thunder_fused_readout_hierarchical_precise_both_cuda(
        qgv, embd, gate_min, gate_max, eps, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}



#ifdef WITH_CUDA
static torch::Tensor linear_fp16_accum_cuda_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "fast16 GEMM requires CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half &&
                weight.scalar_type() == at::ScalarType::Half,
                "fast16 GEMM currently requires FP16 tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "fast16 GEMM requires contiguous tensors");
    TORCH_CHECK(x.dim() == 3, "fast16 GEMM expects x shape [B,T,K]");
    TORCH_CHECK(weight.dim() == 2, "fast16 GEMM expects weight shape [N,K]");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t K64 = x.size(2);
    const int64_t N64 = weight.size(0);
    TORCH_CHECK(weight.size(1) == K64, "weight inner dimension must match x");
    const int64_t M64 = B * T;
    TORCH_CHECK(M64 <= std::numeric_limits<int>::max() &&
                N64 <= std::numeric_limits<int>::max() &&
                K64 <= std::numeric_limits<int>::max(),
                "fast16 GEMM dimensions exceed cuBLAS int32 limits");

    auto out = torch::empty({B, T, N64}, x.options());
    if (M64 == 0 || N64 == 0 || K64 == 0) {
        return out;
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    // Row-major: out[M,N] = x[M,K] @ weight[N,K]^T.
    // cuBLAS is column-major, so reinterpret buffers and compute
    // out^T[N,M] = weight[N,K] @ x^T[K,M].
    const at::Half alpha = at::Half(1.0f);
    const at::Half beta = at::Half(0.0f);

    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        weight.data_ptr<at::Half>(),
        CUDA_R_16F,
        K,
        x.data_ptr<at::Half>(),
        CUDA_R_16F,
        K,
        &beta,
        out.data_ptr<at::Half>(),
        CUDA_R_16F,
        N,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT));

    return out;
}
#endif


torch::Tensor linear_fp16_accum(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
#ifdef WITH_CUDA
    return linear_fp16_accum_cuda_impl(x.contiguous(), weight.contiguous());
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

#ifdef WITH_CUDA
static torch::Tensor linear_fp32_accum_cuda_impl(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "fp32-accum GEMM requires CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half &&
                weight.scalar_type() == at::ScalarType::Half,
                "fp32-accum GEMM currently requires FP16 tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "fp32-accum GEMM requires contiguous tensors");
    TORCH_CHECK(x.dim() == 3, "fp32-accum GEMM expects x shape [B,T,K]");
    TORCH_CHECK(weight.dim() == 2, "fp32-accum GEMM expects weight shape [N,K]");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t K64 = x.size(2);
    const int64_t N64 = weight.size(0);
    TORCH_CHECK(weight.size(1) == K64, "weight inner dimension must match x");
    const int64_t M64 = B * T;
    TORCH_CHECK(M64 <= std::numeric_limits<int>::max() &&
                N64 <= std::numeric_limits<int>::max() &&
                K64 <= std::numeric_limits<int>::max(),
                "fp32-accum GEMM dimensions exceed cuBLAS int32 limits");

    auto out = torch::empty({B, T, N64}, x.options());
    if (M64 == 0 || N64 == 0 || K64 == 0) {
        return out;
    }

    c10::cuda::CUDAGuard device_guard(x.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointer_mode(handle, CUBLAS_POINTER_MODE_HOST);

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);
    const int K = static_cast<int>(K64);

    // Row-major: out[M,N] = x[M,K] @ weight[N,K]^T.
    // cuBLAS is column-major, so compute out^T[N,M] = weight[N,K] @ x^T[K,M].
    // Inputs and output remain FP16, but products/sums are accumulated in FP32.
    const float alpha = 1.0f;
    const float beta = 0.0f;

    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        weight.data_ptr<at::Half>(),
        CUDA_R_16F,
        K,
        x.data_ptr<at::Half>(),
        CUDA_R_16F,
        K,
        &beta,
        out.data_ptr<at::Half>(),
        CUDA_R_16F,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));

    return out;
}
#endif

torch::Tensor linear_fp32_accum(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
#ifdef WITH_CUDA
    return linear_fp32_accum_cuda_impl(x.contiguous(), weight.contiguous());
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor linear_hybrid_accum(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    int64_t chunk) {
#ifdef WITH_CUDA
    return linear_hybrid_accum_cuda(
        x.contiguous(), weight.contiguous(), chunk);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_forward_hierarchical(
    const torch::Tensor& x,
    const torch::Tensor& qgv_weight,
    const torch::Tensor& out_weight,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(x.defined() && qgv_weight.defined() && out_weight.defined(),
                "x and projection weights must be defined");
    TORCH_CHECK(x.dim() == 3, "expected x shape [B,T,C]");
    TORCH_CHECK(qgv_weight.dim() == 2 && out_weight.dim() == 2,
                "projection weights must be rank-2");
    const int64_t embd = x.size(2);
    TORCH_CHECK(qgv_weight.size(0) == 3 * embd && qgv_weight.size(1) == embd,
                "qgv_weight must have shape [3C,C]");
    TORCH_CHECK(out_weight.size(0) == embd && out_weight.size(1) == embd,
                "out_weight must have shape [C,C]");
    TORCH_CHECK(x.device() == qgv_weight.device() && x.device() == out_weight.device(),
                "x and weights must be on the same device");
    TORCH_CHECK(x.scalar_type() == qgv_weight.scalar_type() &&
                x.scalar_type() == out_weight.scalar_type(),
                "x and weights must have the same dtype");
    TORCH_CHECK(x.is_cuda(), "native hierarchical projection path is CUDA-only");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half ||
                x.scalar_type() == at::ScalarType::BFloat16,
                "native hierarchical projection path supports FP16/BF16");

    // Keep the exact trained nn.Linear semantics (bias=False):
    // qgv = x @ qgv_weight.T, out = y @ out_weight.T.
    // ATen dispatches these GEMMs to the same optimized CUDA BLAS backend as
    // nn.Linear, but keeping all three stages inside one C++ entrypoint removes
    // Python-side orchestration between the two GEMMs and the ESA CUDA core.
    auto qgv = at::linear(x, qgv_weight, c10::nullopt);
    auto y = thunder_fused_readout_hierarchical(
        qgv, embd, gate_min, gate_max, eps, compass);
    return at::linear(y, out_weight, c10::nullopt);
}


torch::Tensor thunder_forward_hierarchical_fp16gemm(
    const torch::Tensor& x,
    const torch::Tensor& qgv_weight,
    const torch::Tensor& out_weight,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half,
                "FP16-accum projection path requires FP16 input");
    TORCH_CHECK(qgv_weight.scalar_type() == at::ScalarType::Half &&
                out_weight.scalar_type() == at::ScalarType::Half,
                "FP16-accum projection weights must be FP16");
    const int64_t embd = x.size(2);
    auto qgv = linear_fp16_accum(x, qgv_weight);
    auto y = thunder_fused_readout_hierarchical(
        qgv, embd, gate_min, gate_max, eps, compass);
    return linear_fp16_accum(y, out_weight);
}


torch::Tensor thunder_forward_hierarchical_mixedgemm(
    const torch::Tensor& x,
    const torch::Tensor& qgv_weight,
    const torch::Tensor& out_weight,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half,
                "mixed projection path requires FP16 input");
    TORCH_CHECK(qgv_weight.scalar_type() == at::ScalarType::Half &&
                out_weight.scalar_type() == at::ScalarType::Half,
                "mixed projection weights must be FP16");
    const int64_t embd = x.size(2);

    // v6 mixed mode:
    //   QGV      -> native FP16 accumulation for speed
    //   ESA core -> hierarchical v3
    //   out_proj -> native FP32 accumulation for accuracy, FP16 output
    auto qgv = linear_fp16_accum(x, qgv_weight);
    auto y = thunder_fused_readout_hierarchical(
        qgv, embd, gate_min, gate_max, eps, compass);
    return linear_fp32_accum(y, out_weight);
}

torch::Tensor thunder_forward_hierarchical_hybridgemm(
    const torch::Tensor& x,
    const torch::Tensor& qgv_weight,
    const torch::Tensor& out_weight,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass,
    int64_t out_chunk) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half,
                "hybrid projection path requires FP16 input");
    TORCH_CHECK(qgv_weight.scalar_type() == at::ScalarType::Half &&
                out_weight.scalar_type() == at::ScalarType::Half,
                "hybrid projection weights must be FP16");
    const int64_t embd = x.size(2);

    // v7 hybrid mode:
    //   QGV      -> native FP16 accumulation for speed
    //   ESA core -> hierarchical v3
    //   out_proj -> K-chunk FP16 partial GEMMs + FP32 partial reduction
    auto qgv = linear_fp16_accum(x, qgv_weight);
    auto y = thunder_fused_readout_hierarchical(
        qgv, embd, gate_min, gate_max, eps, compass);
    return linear_hybrid_accum(y, out_weight, out_chunk);
}


std::vector<torch::Tensor> thunder_prepare_ab(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max) {
    TORCH_CHECK(qgv.defined(), "qgv must be defined");
    TORCH_CHECK(qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0 && qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(qgv.is_cuda(), "v9 prepare_ab debug operator is CUDA-only");
    TORCH_CHECK(qgv.is_contiguous(), "qgv must be contiguous");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "v9 prepare_ab supports FP16/BF16 tensors");
#ifdef WITH_CUDA
    return thunder_prepare_ab_cuda(qgv, embd, gate_min, gate_max);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_readout(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps) {
    TORCH_CHECK(q.defined() && states.defined(), "q and states must be defined");
    TORCH_CHECK(q.dim() == 3 && states.dim() == 3,
                "expected q/states shape [B,T,C]");
    TORCH_CHECK(q.sizes() == states.sizes(), "q and states must have same shape");
    TORCH_CHECK(q.device() == states.device(), "q and states must share device");
    TORCH_CHECK(q.scalar_type() == states.scalar_type(), "q and states must share dtype");
    TORCH_CHECK(q.is_cuda(), "v9 readout debug operator is CUDA-only");
    TORCH_CHECK(q.is_contiguous() && states.is_contiguous(),
                "q and states must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                q.scalar_type() == at::ScalarType::BFloat16,
                "v9 readout supports FP16/BF16 tensors");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
#ifdef WITH_CUDA
    return thunder_readout_cuda(q, states, eps);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}


std::vector<torch::Tensor> thunder_prepare_ab_precise(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max) {
    TORCH_CHECK(qgv.defined() && qgv.dim() == 3, "expected qgv shape [B,T,3C]");
    TORCH_CHECK(embd > 0 && qgv.size(2) == 3 * embd,
                "qgv last dimension must equal 3*embd");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(qgv.is_cuda() && qgv.is_contiguous(),
                "v10 precise prepare_ab requires contiguous CUDA qgv");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "v10 precise prepare_ab supports FP16/BF16 tensors");
#ifdef WITH_CUDA
    return thunder_prepare_ab_precise_cuda(qgv, embd, gate_min, gate_max);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_readout_precise(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps) {
    TORCH_CHECK(q.defined() && states.defined(), "q and states must be defined");
    TORCH_CHECK(q.dim() == 3 && states.dim() == 3 && q.sizes() == states.sizes(),
                "expected equal q/states shape [B,T,C]");
    TORCH_CHECK(q.device() == states.device() && q.scalar_type() == states.scalar_type(),
                "q and states must share device and dtype");
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && states.is_contiguous(),
                "v10 precise readout requires contiguous CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half ||
                q.scalar_type() == at::ScalarType::BFloat16,
                "v10 precise readout supports FP16/BF16 tensors");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
#ifdef WITH_CUDA
    return thunder_readout_precise_cuda(q, states, eps);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_scan(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass) {
    check_scan_inputs(A, B_write);
    TORCH_CHECK(compass > 0, "compass must be positive");
    if (A.is_cuda()) {
#ifdef WITH_CUDA
        return thunder_scan_cuda(A, B_write, compass);
#else
        TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
    }
    return thunder_scan_cpu(A, B_write, compass);
}

torch::Tensor thunder_scan_hierarchical(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass) {
    check_scan_inputs(A, B_write);
    TORCH_CHECK(compass > 0, "compass must be positive");
    TORCH_CHECK(A.is_cuda(), "hierarchical Thunder scan is CUDA-only");
#ifdef WITH_CUDA
    return thunder_scan_hierarchical_cuda(A, B_write, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}


static void check_summary_inputs(
    const torch::Tensor& A,
    const torch::Tensor& B_write) {
    TORCH_CHECK(A.defined() && B_write.defined(), "summary tensors must be defined");
    TORCH_CHECK(A.sizes() == B_write.sizes(), "summary A/B tensors must match");
    TORCH_CHECK(A.dim() == 3, "expected summary tensors [B,G,C]");
    TORCH_CHECK(A.is_cuda() && B_write.is_cuda(), "summary planner ops are CUDA-only");
    TORCH_CHECK(A.is_contiguous() && B_write.is_contiguous(), "summary tensors must be contiguous");
    TORCH_CHECK(A.scalar_type() == B_write.scalar_type(), "summary tensors must share dtype");
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Half ||
                A.scalar_type() == at::ScalarType::BFloat16,
                "summary planner ops support FP16/BF16");
}

std::vector<torch::Tensor> thunder_scan_local(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass) {
    check_scan_inputs(A, B_write);
    TORCH_CHECK(A.is_cuda(), "planner local scan is CUDA-only");
    TORCH_CHECK(compass > 0, "compass must be positive");
#ifdef WITH_CUDA
    return thunder_scan_local_cuda(A, B_write, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_summary_scan(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t group_size) {
    check_summary_inputs(A, B_write);
    TORCH_CHECK(group_size > 0, "group_size must be positive");
#ifdef WITH_CUDA
    return thunder_summary_scan_cuda(A, B_write, group_size);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_group_prefix(
    const torch::Tensor& A,
    const torch::Tensor& B_write) {
    check_summary_inputs(A, B_write);
#ifdef WITH_CUDA
    return thunder_group_prefix_cuda(A, B_write);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_apply_group(
    const torch::Tensor& pref_A,
    const torch::Tensor& pref_B,
    const torch::Tensor& parent_A,
    const torch::Tensor& parent_B,
    int64_t group_size) {
    check_summary_inputs(pref_A, pref_B);
    check_summary_inputs(parent_A, parent_B);
    TORCH_CHECK(pref_A.size(0) == parent_A.size(0) && pref_A.size(2) == parent_A.size(2),
                "parent summary shape is incompatible");
    TORCH_CHECK(pref_A.scalar_type() == parent_A.scalar_type(), "parent summaries must share dtype");
    TORCH_CHECK(group_size > 0, "group_size must be positive");
#ifdef WITH_CUDA
    return thunder_apply_group_cuda(pref_A, pref_B, parent_A, parent_B, group_size);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor thunder_apply_chunk_prefix(
    const torch::Tensor& A,
    const torch::Tensor& local_states,
    const torch::Tensor& chunk_B_prefix,
    int64_t compass) {
    check_scan_inputs(A, local_states);
    TORCH_CHECK(chunk_B_prefix.dim() == 3 && chunk_B_prefix.is_cuda() && chunk_B_prefix.is_contiguous(),
                "chunk prefix must be contiguous CUDA [B,G,C]");
    TORCH_CHECK(chunk_B_prefix.scalar_type() == A.scalar_type(), "chunk prefix dtype must match A");
    TORCH_CHECK(compass > 0, "compass must be positive");
#ifdef WITH_CUDA
    return thunder_apply_chunk_prefix_cuda(A, local_states, chunk_B_prefix, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_scan_backward_chunked(
    const torch::Tensor& A,
    const torch::Tensor& states,
    const torch::Tensor& grad,
    int64_t compass) {
    check_scan_inputs(A, states);
    check_scan_inputs(A, grad);
    TORCH_CHECK(A.is_cuda(), "planner backward is CUDA-only");
    TORCH_CHECK(compass > 0, "compass must be positive");
#ifdef WITH_CUDA
    return thunder_scan_backward_chunked_cuda(A, states, grad, compass);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_reverse_prepare(
    const torch::Tensor& A,
    const torch::Tensor& grad) {
    check_scan_inputs(A, grad);
    TORCH_CHECK(A.is_cuda(), "planner reverse prepare is CUDA-only");
#ifdef WITH_CUDA
    return thunder_reverse_prepare_cuda(A, grad);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> thunder_reverse_finish(
    const torch::Tensor& grad_reverse,
    const torch::Tensor& states) {
    check_scan_inputs(grad_reverse, states);
    TORCH_CHECK(grad_reverse.is_cuda(), "planner reverse finish is CUDA-only");
#ifdef WITH_CUDA
    return thunder_reverse_finish_cuda(grad_reverse, states);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor lightning_step(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state) {
    check_step_inputs(A, B_write, state);
    if (A.is_cuda()) {
#ifdef WITH_CUDA
        return lightning_step_cuda(A, B_write, state);
#else
        TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
    }
    return lightning_step_cpu(A, B_write, state);
}


std::vector<torch::Tensor> lightning_fused_step(
    const torch::Tensor& qgv,
    const torch::Tensor& state,
    double gate_min,
    double gate_max,
    double eps) {
    TORCH_CHECK(qgv.defined() && state.defined(), "qgv and state must be defined");
    TORCH_CHECK(qgv.dim() == 2, "expected qgv shape [B,3C]");
    TORCH_CHECK(state.dim() == 3, "expected state shape [B,H,D]");
    TORCH_CHECK(qgv.size(0) == state.size(0), "qgv/state batch mismatch");
    TORCH_CHECK(qgv.size(1) == 3 * state.size(1) * state.size(2),
                "qgv width must equal 3*state channels");
    TORCH_CHECK(qgv.scalar_type() == state.scalar_type(), "qgv/state dtype mismatch");
    TORCH_CHECK(qgv.device() == state.device(), "qgv/state device mismatch");
    TORCH_CHECK(qgv.is_contiguous() && state.is_contiguous(), "qgv/state must be contiguous");
    TORCH_CHECK(gate_max > gate_min, "gate_max must be greater than gate_min");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    TORCH_CHECK(qgv.scalar_type() == at::ScalarType::Half ||
                qgv.scalar_type() == at::ScalarType::BFloat16,
                "fused Lightning currently supports FP16/BF16");
    TORCH_CHECK(qgv.is_cuda(), "fused Lightning is CUDA-only");
#ifdef WITH_CUDA
    return lightning_fused_step_cuda(qgv, state, gate_min, gate_max, eps);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

torch::Tensor ffn_gelu_residual(
    const torch::Tensor& normalized,
    const torch::Tensor& residual,
    const torch::Tensor& w1,
    const torch::Tensor& b1,
    const torch::Tensor& w2,
    const torch::Tensor& b2) {
    TORCH_CHECK(normalized.defined() && residual.defined(), "FFN inputs must be defined");
    TORCH_CHECK(normalized.sizes() == residual.sizes(), "FFN residual/input shape mismatch");
    TORCH_CHECK(normalized.size(-1) == w1.size(1), "FFN first projection input mismatch");
    TORCH_CHECK(w2.size(1) == w1.size(0), "FFN intermediate dimension mismatch");
    TORCH_CHECK(w2.size(0) == residual.size(-1), "FFN output dimension mismatch");
    TORCH_CHECK(b1.numel() == w1.size(0) && b2.numel() == w2.size(0), "FFN bias mismatch");
    auto hidden = at::linear(normalized, w1, b1);
    hidden = at::gelu(hidden, "none");
    auto update = at::linear(hidden, w2, b2);
    return residual + update;
}

torch::Tensor elastic_linear_packed(
    const torch::Tensor& x,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& bias,
    int64_t bits,
    int64_t group_size,
    int64_t out_features,
    int64_t in_features) {
    TORCH_CHECK(x.is_cuda(), "elastic_linear_packed is CUDA-only");
    TORCH_CHECK(x.is_contiguous() && packed.is_contiguous() && scales.is_contiguous(),
                "x/packed/scales must be contiguous");
    TORCH_CHECK(x.is_floating_point() && scales.is_floating_point(),
                "x/scales must be floating point");
    TORCH_CHECK(packed.scalar_type() == at::ScalarType::Byte,
                "packed weight must be uint8");
    TORCH_CHECK(x.size(-1) == in_features, "input feature mismatch");
    TORCH_CHECK(bits >= 2 && bits <= 8, "bits must be in [2,8]");
    TORCH_CHECK(group_size > 0 && out_features > 0 && in_features > 0,
                "invalid ElasticBit dimensions");
    TORCH_CHECK(packed.device() == x.device() && scales.device() == x.device(),
                "ElasticBit buffers must be on the same CUDA device as x");
    TORCH_CHECK(bias.numel() == 0 || (bias.device() == x.device() &&
                bias.scalar_type() == x.scalar_type() && bias.numel() == out_features),
                "bias must be empty or match output features/device/dtype");
#ifdef WITH_CUDA
    return elastic_linear_packed_cuda(
        x, packed, scales, bias, bits, group_size, out_features, in_features);
#else
    TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
}

std::vector<torch::Tensor> residual_layer_norm(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
    TORCH_CHECK(x.defined() && update.defined() && weight.defined() && bias.defined(),
                "residual_layer_norm inputs must be defined");
    TORCH_CHECK(x.sizes() == update.sizes(), "x/update shape mismatch");
    TORCH_CHECK(x.dim() >= 2, "residual_layer_norm expects rank >= 2");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1, "weight/bias must be vectors");
    TORCH_CHECK(weight.numel() == x.size(-1) && bias.numel() == x.size(-1),
                "weight/bias size must match the final dimension");
    TORCH_CHECK(x.scalar_type() == update.scalar_type() &&
                x.scalar_type() == weight.scalar_type() &&
                x.scalar_type() == bias.scalar_type(), "dtype mismatch");
    TORCH_CHECK(x.device() == update.device() && x.device() == weight.device() &&
                x.device() == bias.device(), "device mismatch");
    TORCH_CHECK(x.is_contiguous() && update.is_contiguous() &&
                weight.is_contiguous() && bias.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(eps > 0.0, "eps must be positive");
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        return residual_layer_norm_cuda(x, update, weight, bias, eps);
#else
        TORCH_CHECK(false, "MLBricks was built without CUDA support");
#endif
    }
    return residual_layer_norm_cpu(x, update, weight, bias, eps);
}

bool has_cuda() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

} // namespace mlbricks


// -----------------------------------------------------------------------------
// PyTorch dispatcher registration
// -----------------------------------------------------------------------------
// These schemas make MLBricks native kernels first-class torch operators.  The
// extension keeps the legacy pybind surface for backwards compatibility, but
// MLBricks Python code calls torch.ops.mlbricks_native.* so torch.compile,
// FakeTensor/AOTAutograd, export, and profiler tooling can reason about kernel
// boundaries instead of seeing opaque pybind functions.
TORCH_LIBRARY(mlbricks_native, m) {
    m.def("thunder_fused_readout(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical_mixed32(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical_full32(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical_precise_gate(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical_precise_readout(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_fused_readout_hierarchical_precise_both(Tensor qgv, int embd, float gate_min, float gate_max, float eps, int compass) -> Tensor");

    m.def("thunder_forward_hierarchical(Tensor x, Tensor qgv_weight, Tensor out_weight, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_forward_hierarchical_fp16gemm(Tensor x, Tensor qgv_weight, Tensor out_weight, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_forward_hierarchical_mixedgemm(Tensor x, Tensor qgv_weight, Tensor out_weight, float gate_min, float gate_max, float eps, int compass) -> Tensor");
    m.def("thunder_forward_hierarchical_hybridgemm(Tensor x, Tensor qgv_weight, Tensor out_weight, float gate_min, float gate_max, float eps, int compass, int out_chunk) -> Tensor");

    m.def("linear_fp16_accum(Tensor x, Tensor weight) -> Tensor");
    m.def("linear_fp32_accum(Tensor x, Tensor weight) -> Tensor");
    m.def("linear_hybrid_accum(Tensor x, Tensor weight, int chunk) -> Tensor");

    m.def("thunder_prepare_ab(Tensor qgv, int embd, float gate_min, float gate_max) -> Tensor[]");
    m.def("thunder_readout(Tensor q, Tensor states, float eps) -> Tensor");
    m.def("thunder_prepare_ab_precise(Tensor qgv, int embd, float gate_min, float gate_max) -> Tensor[]");
    m.def("thunder_readout_precise(Tensor q, Tensor states, float eps) -> Tensor");

    m.def("thunder_scan(Tensor A, Tensor B_write, int compass) -> Tensor");
    m.def("thunder_scan_hierarchical(Tensor A, Tensor B_write, int compass) -> Tensor");
    m.def("thunder_scan_local(Tensor A, Tensor B_write, int compass) -> Tensor[]");
    m.def("thunder_summary_scan(Tensor A, Tensor B_write, int group_size) -> Tensor[]");
    m.def("thunder_group_prefix(Tensor A, Tensor B_write) -> Tensor[]");
    m.def("thunder_apply_group(Tensor pref_A, Tensor pref_B, Tensor parent_A, Tensor parent_B, int group_size) -> Tensor[]");
    m.def("thunder_apply_chunk_prefix(Tensor A, Tensor local_states, Tensor chunk_B_prefix, int compass) -> Tensor");
    m.def("thunder_scan_backward_chunked(Tensor A, Tensor states, Tensor grad, int compass) -> Tensor[]");
    m.def("thunder_reverse_prepare(Tensor A, Tensor grad) -> Tensor[]");
    m.def("thunder_reverse_finish(Tensor grad_reverse, Tensor states) -> Tensor[]");

    m.def("lightning_step(Tensor A, Tensor B_write, Tensor state) -> Tensor");
    m.def("lightning_fused_step(Tensor qgv, Tensor state, float gate_min, float gate_max, float eps) -> Tensor[]");
    m.def("residual_layer_norm(Tensor x, Tensor update, Tensor weight, Tensor bias, float eps) -> Tensor[]");
    m.def("elastic_linear_packed(Tensor x, Tensor packed, Tensor scales, Tensor bias, int bits, int group_size, int out_features, int in_features) -> Tensor");
    m.def("ffn_gelu_residual(Tensor normalized, Tensor residual, Tensor w1, Tensor b1, Tensor w2, Tensor b2) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlbricks_native, CompositeExplicitAutograd, m) {
    m.impl("thunder_fused_readout", TORCH_FN(mlbricks::thunder_fused_readout));
    m.impl("thunder_fused_readout_hierarchical", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical));
    m.impl("thunder_fused_readout_hierarchical_mixed32", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical_mixed32));
    m.impl("thunder_fused_readout_hierarchical_full32", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical_full32));
    m.impl("thunder_fused_readout_hierarchical_precise_gate", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical_precise_gate));
    m.impl("thunder_fused_readout_hierarchical_precise_readout", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical_precise_readout));
    m.impl("thunder_fused_readout_hierarchical_precise_both", TORCH_FN(mlbricks::thunder_fused_readout_hierarchical_precise_both));

    m.impl("thunder_forward_hierarchical", TORCH_FN(mlbricks::thunder_forward_hierarchical));
    m.impl("thunder_forward_hierarchical_fp16gemm", TORCH_FN(mlbricks::thunder_forward_hierarchical_fp16gemm));
    m.impl("thunder_forward_hierarchical_mixedgemm", TORCH_FN(mlbricks::thunder_forward_hierarchical_mixedgemm));
    m.impl("thunder_forward_hierarchical_hybridgemm", TORCH_FN(mlbricks::thunder_forward_hierarchical_hybridgemm));

    m.impl("linear_fp16_accum", TORCH_FN(mlbricks::linear_fp16_accum));
    m.impl("linear_fp32_accum", TORCH_FN(mlbricks::linear_fp32_accum));
    m.impl("linear_hybrid_accum", TORCH_FN(mlbricks::linear_hybrid_accum));

    m.impl("thunder_prepare_ab", TORCH_FN(mlbricks::thunder_prepare_ab));
    m.impl("thunder_readout", TORCH_FN(mlbricks::thunder_readout));
    m.impl("thunder_prepare_ab_precise", TORCH_FN(mlbricks::thunder_prepare_ab_precise));
    m.impl("thunder_readout_precise", TORCH_FN(mlbricks::thunder_readout_precise));

    m.impl("thunder_scan", TORCH_FN(mlbricks::thunder_scan));
    m.impl("thunder_scan_hierarchical", TORCH_FN(mlbricks::thunder_scan_hierarchical));
    m.impl("thunder_scan_local", TORCH_FN(mlbricks::thunder_scan_local));
    m.impl("thunder_summary_scan", TORCH_FN(mlbricks::thunder_summary_scan));
    m.impl("thunder_group_prefix", TORCH_FN(mlbricks::thunder_group_prefix));
    m.impl("thunder_apply_group", TORCH_FN(mlbricks::thunder_apply_group));
    m.impl("thunder_apply_chunk_prefix", TORCH_FN(mlbricks::thunder_apply_chunk_prefix));
    m.impl("thunder_scan_backward_chunked", TORCH_FN(mlbricks::thunder_scan_backward_chunked));
    m.impl("thunder_reverse_prepare", TORCH_FN(mlbricks::thunder_reverse_prepare));
    m.impl("thunder_reverse_finish", TORCH_FN(mlbricks::thunder_reverse_finish));

    m.impl("lightning_step", TORCH_FN(mlbricks::lightning_step));
    m.impl("lightning_fused_step", TORCH_FN(mlbricks::lightning_fused_step));
    m.impl("residual_layer_norm", TORCH_FN(mlbricks::residual_layer_norm));
    m.impl("elastic_linear_packed", TORCH_FN(mlbricks::elastic_linear_packed));
    m.impl("ffn_gelu_residual", TORCH_FN(mlbricks::ffn_gelu_residual));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MLBricks native ESA operators";
    m.def("thunder_fused_readout", &mlbricks::thunder_fused_readout,
          "Fused Thunder transforms + scan + normalized gated readout");
    m.def("thunder_fused_readout_hierarchical", &mlbricks::thunder_fused_readout_hierarchical,
          "Hierarchical fused Thunder readout for long sequences");
    m.def("thunder_fused_readout_hierarchical_mixed32",
          &mlbricks::thunder_fused_readout_hierarchical_mixed32,
          "v8 hierarchical Thunder with FP32 chunk summary/prefix accumulation");
    m.def("thunder_fused_readout_hierarchical_full32",
          &mlbricks::thunder_fused_readout_hierarchical_full32,
          "v8 hierarchical Thunder with full FP32 recurrence accumulation");
    m.def("thunder_fused_readout_hierarchical_precise_gate",
          &mlbricks::thunder_fused_readout_hierarchical_precise_gate,
          "v10 hierarchical Thunder with PyTorch-faithful gate/value math");
    m.def("thunder_fused_readout_hierarchical_precise_readout",
          &mlbricks::thunder_fused_readout_hierarchical_precise_readout,
          "v10 hierarchical Thunder with PyTorch-faithful RMS/readout math");
    m.def("thunder_fused_readout_hierarchical_precise_both",
          &mlbricks::thunder_fused_readout_hierarchical_precise_both,
          "v10 hierarchical Thunder with precise gate/value and readout math");
    m.def("thunder_forward_hierarchical", &mlbricks::thunder_forward_hierarchical,
          "C++-orchestrated QGV + hierarchical Thunder + output projection");
    m.def("linear_fp16_accum", &mlbricks::linear_fp16_accum,
          "FP16-accumulation cuBLAS GEMM for [B,T,K] x [N,K]^T");
    m.def("linear_fp32_accum", &mlbricks::linear_fp32_accum,
          "FP32-accumulation cuBLAS GEMM with FP16 input/output");
    m.def("linear_hybrid_accum", &mlbricks::linear_hybrid_accum,
          "Hybrid GEMM: FP16 K-chunk partials with FP32 reduction");
    m.def("thunder_forward_hierarchical_fp16gemm",
          &mlbricks::thunder_forward_hierarchical_fp16gemm,
          "Hierarchical Thunder with native FP16-accumulation cuBLAS projections");
    m.def("thunder_forward_hierarchical_mixedgemm",
          &mlbricks::thunder_forward_hierarchical_mixedgemm,
          "Hierarchical Thunder with FP16-accum QGV and FP32-accum output projection");
    m.def("thunder_forward_hierarchical_hybridgemm",
          &mlbricks::thunder_forward_hierarchical_hybridgemm,
          "Hierarchical Thunder with FP16 QGV and hybrid output accumulation");
    m.def("thunder_prepare_ab", &mlbricks::thunder_prepare_ab,
          "v9 debug: native gate/value preparation returning A and B_write");
    m.def("thunder_readout", &mlbricks::thunder_readout,
          "v9 debug: native RMS plus sigmoid(q) readout");
    m.def("thunder_prepare_ab_precise", &mlbricks::thunder_prepare_ab_precise,
          "v10 debug: PyTorch-faithful gate/value preparation");
    m.def("thunder_readout_precise", &mlbricks::thunder_readout_precise,
          "v10 debug: PyTorch-faithful RMS plus sigmoid(q) readout");
    m.def("thunder_scan", &mlbricks::thunder_scan, "Native direct ESA scan");
    m.def("thunder_scan_hierarchical", &mlbricks::thunder_scan_hierarchical,
          "Native hierarchical ESA scan");
    m.def("thunder_scan_local", &mlbricks::thunder_scan_local,
          "Auto-planner local chunk scan and summaries");
    m.def("thunder_summary_scan", &mlbricks::thunder_summary_scan,
          "Auto-planner grouped summary scan");
    m.def("thunder_group_prefix", &mlbricks::thunder_group_prefix,
          "Auto-planner base summary prefix");
    m.def("thunder_apply_group", &mlbricks::thunder_apply_group,
          "Auto-planner parent carry propagation");
    m.def("thunder_apply_chunk_prefix", &mlbricks::thunder_apply_chunk_prefix,
          "Auto-planner token carry propagation");
    m.def("thunder_scan_backward_chunked", &mlbricks::thunder_scan_backward_chunked,
          "Chunk-parallel ESA backward for direct hierarchy depth");
    m.def("thunder_reverse_prepare", &mlbricks::thunder_reverse_prepare,
          "Transform ESA backward into a reverse affine scan");
    m.def("thunder_reverse_finish", &mlbricks::thunder_reverse_finish,
          "Finish ESA gradients after reverse affine scan");
    m.def("lightning_step", &mlbricks::lightning_step, "Native ESA recurrent step");
    m.def("residual_layer_norm", &mlbricks::residual_layer_norm,
          "Fused residual add + LayerNorm (CUDA fast path)");
    m.def("elastic_linear_packed", &mlbricks::elastic_linear_packed,
          "Direct packed ElasticBit CUDA linear without full weight materialization");
    m.def("ffn_gelu_residual", &mlbricks::ffn_gelu_residual,
          "Native Linear-GELU-Linear-residual inference orchestration");
    m.def("lightning_fused_step", &mlbricks::lightning_fused_step,
          "Fused one-token ESA Lightning transform + recurrence + readout");
    m.def("has_cuda", &mlbricks::has_cuda, "Whether this extension was built with CUDA");
}
