// Copyright 2026 Zameer Hussain and Akhtar Hussain
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// See LICENSE and LICENSING_NOTICE.md; commercial use requires a separate written license.

#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace mlbricks {

template <typename scalar_t>
__global__ void direct_scan_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_write,
    scalar_t* __restrict__ out,
    int64_t batch,
    int64_t time,
    int64_t channels) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t bc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels;
    if (bc >= total) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    acc_t state = acc_t(0);

    #pragma unroll 1
    for (int64_t t = 0; t < time; ++t) {
        const int64_t idx = (b * time + t) * channels + c;
        state = static_cast<acc_t>(A[idx]) * state + static_cast<acc_t>(B_write[idx]);
        out[idx] = static_cast<scalar_t>(state);
    }
}



template <typename scalar_t>
__device__ __forceinline__ float quantize_scan_value(float x) {
    const scalar_t q = static_cast<scalar_t>(x);
    return static_cast<float>(q);
}

template <typename scalar_t>
__global__ void hierarchical_local_scan_ab_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_write,
    scalar_t* __restrict__ local_states,
    scalar_t* __restrict__ chunk_A,
    scalar_t* __restrict__ chunk_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;

    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float state = 0.0f;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t idx = (b * time + t) * channels + c;
        const float a = static_cast<float>(A[idx]);
        const float bw = static_cast<float>(B_write[idx]);

        // Match the FP16/BF16 Thunder path: each recurrent local step and
        // transition is materialized in the scan dtype before the next step.
        state = quantize_scan_value<scalar_t>(a * state + bw);
        transition = quantize_scan_value<scalar_t>(a * transition);
        local_states[idx] = static_cast<scalar_t>(state);
    }

    const int64_t chunk_idx = (b * chunks + g) * channels + c;
    chunk_A[chunk_idx] = static_cast<scalar_t>(transition);
    chunk_B[chunk_idx] = static_cast<scalar_t>(state);
}

template <typename scalar_t>
__global__ void hierarchical_chunk_prefix_scan_ab_kernel(
    scalar_t* __restrict__ chunk_A,
    scalar_t* __restrict__ chunk_B,
    int64_t batch,
    int64_t chunks,
    int64_t channels) {

    const int64_t bc = static_cast<int64_t>(blockIdx.x);
    const int64_t total_bc = batch * channels;
    if (bc >= total_bc) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    const int g = threadIdx.x;

    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = shared + blockDim.x;

    float a = 1.0f;
    float bw = 0.0f;
    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        a = static_cast<float>(chunk_A[idx]);
        bw = static_cast<float>(chunk_B[idx]);
    }
    sA[g] = a;
    sB[g] = bw;
    __syncthreads();

    // Inclusive associative scan over affine chunk summaries.
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const float curA = sA[g];
        const float curB = sB[g];
        float prevA = 1.0f;
        float prevB = 0.0f;
        if (g >= offset) {
            prevA = sA[g - offset];
            prevB = sB[g - offset];
        }
        __syncthreads();
        if (g >= offset) {
            sA[g] = quantize_scan_value<scalar_t>(curA * prevA);
            sB[g] = quantize_scan_value<scalar_t>(curA * prevB + curB);
        }
        __syncthreads();
    }

    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        chunk_A[idx] = static_cast<scalar_t>(sA[g]);
        chunk_B[idx] = static_cast<scalar_t>(sB[g]);
    }
}

template <typename scalar_t>
__global__ void hierarchical_apply_chunk_prefix_ab_kernel(
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ states,
    const scalar_t* __restrict__ chunk_B_prefix,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;

    if (g == 0) return;

    const int64_t prev_chunk_idx = (b * chunks + (g - 1)) * channels + c;
    const float chunk_init = static_cast<float>(chunk_B_prefix[prev_chunk_idx]);

    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t idx = (b * time + t) * channels + c;
        const float a = static_cast<float>(A[idx]);
        transition = quantize_scan_value<scalar_t>(a * transition);

        const float local_state = static_cast<float>(states[idx]);
        states[idx] = static_cast<scalar_t>(
            quantize_scan_value<scalar_t>(transition * chunk_init + local_state));
    }
}

static int hierarchical_next_pow2_threads(int64_t n) {
    int threads = 1;
    while (threads < n && threads < 1024) threads <<= 1;
    return threads;
}


// ============================================================
// Auto-planner multi-level summary primitives.
// These kernels are GPU-model agnostic: the Python planner chooses compass,
// hierarchy depth, and group sizes from workload + generic CUDA resources.
// ============================================================

template <typename scalar_t>
__global__ void planner_summary_scan_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_write,
    scalar_t* __restrict__ pref_A,
    scalar_t* __restrict__ pref_B,
    scalar_t* __restrict__ group_A,
    scalar_t* __restrict__ group_B,
    int64_t batch,
    int64_t summaries,
    int64_t channels,
    int64_t groups,
    int64_t group_size) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * groups * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t group = tmp % groups;
    const int64_t b = tmp / groups;

    const int64_t g0 = group * group_size;
    const int64_t g1 = (g0 + group_size < summaries) ? (g0 + group_size) : summaries;

    float pa = 1.0f;
    float pb = 0.0f;

    for (int64_t g = g0; g < g1; ++g) {
        const int64_t idx = (b * summaries + g) * channels + c;
        const float a = static_cast<float>(A[idx]);
        const float bw = static_cast<float>(B_write[idx]);
        pa = quantize_scan_value<scalar_t>(a * pa);
        pb = quantize_scan_value<scalar_t>(a * pb + bw);
        pref_A[idx] = static_cast<scalar_t>(pa);
        pref_B[idx] = static_cast<scalar_t>(pb);
    }

    const int64_t out_idx = (b * groups + group) * channels + c;
    group_A[out_idx] = static_cast<scalar_t>(pa);
    group_B[out_idx] = static_cast<scalar_t>(pb);
}

template <typename scalar_t>
__global__ void planner_group_prefix_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_write,
    scalar_t* __restrict__ out_A,
    scalar_t* __restrict__ out_B,
    int64_t batch,
    int64_t summaries,
    int64_t channels) {

    const int64_t bc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels;
    if (bc >= total) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    float pa = 1.0f;
    float pb = 0.0f;

    for (int64_t g = 0; g < summaries; ++g) {
        const int64_t idx = (b * summaries + g) * channels + c;
        const float a = static_cast<float>(A[idx]);
        const float bw = static_cast<float>(B_write[idx]);
        pa = quantize_scan_value<scalar_t>(a * pa);
        pb = quantize_scan_value<scalar_t>(a * pb + bw);
        out_A[idx] = static_cast<scalar_t>(pa);
        out_B[idx] = static_cast<scalar_t>(pb);
    }
}

template <typename scalar_t>
__global__ void planner_apply_group_kernel(
    const scalar_t* __restrict__ pref_A,
    const scalar_t* __restrict__ pref_B,
    const scalar_t* __restrict__ parent_A,
    const scalar_t* __restrict__ parent_B,
    scalar_t* __restrict__ out_A,
    scalar_t* __restrict__ out_B,
    int64_t batch,
    int64_t summaries,
    int64_t channels,
    int64_t parent_summaries,
    int64_t group_size) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * summaries * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % summaries;
    const int64_t b = tmp / summaries;
    const int64_t group = g / group_size;

    const float la = static_cast<float>(pref_A[job]);
    const float lb = static_cast<float>(pref_B[job]);

    if (group == 0) {
        out_A[job] = static_cast<scalar_t>(la);
        out_B[job] = static_cast<scalar_t>(lb);
        return;
    }

    const int64_t carry_idx = (b * parent_summaries + (group - 1)) * channels + c;
    const float ca = static_cast<float>(parent_A[carry_idx]);
    const float cb = static_cast<float>(parent_B[carry_idx]);
    out_A[job] = static_cast<scalar_t>(quantize_scan_value<scalar_t>(la * ca));
    out_B[job] = static_cast<scalar_t>(quantize_scan_value<scalar_t>(la * cb + lb));
}

template <typename scalar_t>
__global__ void planner_backward_summary_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ grad,
    float* __restrict__ chunk_P,
    float* __restrict__ chunk_Q,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float P = 1.0f;
    float Q = 0.0f;
    for (int64_t t = t1 - 1; t >= t0; --t) {
        float coeff = 0.0f;
        if (t + 1 < time) {
            const int64_t ai = (b * time + (t + 1)) * channels + c;
            coeff = static_cast<float>(A[ai]);
        }
        const int64_t gi = (b * time + t) * channels + c;
        const float u = static_cast<float>(grad[gi]);
        P = coeff * P;
        Q = coeff * Q + u;
    }

    const int64_t out_idx = (b * chunks + g) * channels + c;
    chunk_P[out_idx] = P;
    chunk_Q[out_idx] = Q;
}

__global__ void planner_backward_boundary_kernel(
    const float* __restrict__ chunk_P,
    const float* __restrict__ chunk_Q,
    float* __restrict__ boundary,
    int64_t batch,
    int64_t chunks,
    int64_t channels) {

    const int64_t bc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels;
    if (bc >= total) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    float future = 0.0f;

    for (int64_t g = chunks - 1; g >= 0; --g) {
        const int64_t idx = (b * chunks + g) * channels + c;
        boundary[idx] = future;
        future = chunk_P[idx] * future + chunk_Q[idx];
    }
}

template <typename scalar_t>
__global__ void planner_backward_apply_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ states,
    const scalar_t* __restrict__ grad,
    const float* __restrict__ boundary,
    scalar_t* __restrict__ grad_A,
    scalar_t* __restrict__ grad_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    const int64_t boundary_idx = (b * chunks + g) * channels + c;
    float future = boundary[boundary_idx];

    for (int64_t t = t1 - 1; t >= t0; --t) {
        float coeff = 0.0f;
        if (t + 1 < time) {
            const int64_t ai = (b * time + (t + 1)) * channels + c;
            coeff = static_cast<float>(A[ai]);
        }
        const int64_t idx = (b * time + t) * channels + c;
        const float u = static_cast<float>(grad[idx]);
        const float current = u + coeff * future;

        float prev_state = 0.0f;
        if (t > 0) {
            const int64_t pi = (b * time + (t - 1)) * channels + c;
            prev_state = static_cast<float>(states[pi]);
        }

        grad_B[idx] = static_cast<scalar_t>(current);
        grad_A[idx] = static_cast<scalar_t>(current * prev_state);
        future = current;
    }
}

template <typename scalar_t>
__global__ void planner_reverse_prepare_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ reverse_A,
    scalar_t* __restrict__ reverse_B,
    int64_t batch,
    int64_t time,
    int64_t channels) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * time * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t r = tmp % time;
    const int64_t b = tmp / time;
    const int64_t t = time - 1 - r;
    const int64_t src = (b * time + t) * channels + c;
    reverse_B[job] = grad[src];

    if (t + 1 < time) {
        const int64_t ai = (b * time + (t + 1)) * channels + c;
        reverse_A[job] = A[ai];
    } else {
        reverse_A[job] = static_cast<scalar_t>(0.0f);
    }
}

template <typename scalar_t>
__global__ void planner_reverse_finish_kernel(
    const scalar_t* __restrict__ grad_reverse,
    const scalar_t* __restrict__ states,
    scalar_t* __restrict__ grad_A,
    scalar_t* __restrict__ grad_B,
    int64_t batch,
    int64_t time,
    int64_t channels) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * time * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t t = tmp % time;
    const int64_t b = tmp / time;
    const int64_t r = time - 1 - t;
    const int64_t ri = (b * time + r) * channels + c;
    const float g = static_cast<float>(grad_reverse[ri]);

    float prev = 0.0f;
    if (t > 0) {
        const int64_t pi = (b * time + (t - 1)) * channels + c;
        prev = static_cast<float>(states[pi]);
    }

    grad_B[job] = static_cast<scalar_t>(g);
    grad_A[job] = static_cast<scalar_t>(quantize_scan_value<scalar_t>(g * prev));
}

template <typename scalar_t>
__global__ void lightning_step_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_write,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ out,
    int64_t n) {
    using acc_t = at::acc_type<scalar_t, true>;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const acc_t next = static_cast<acc_t>(A[i]) * static_cast<acc_t>(state[i])
                     + static_cast<acc_t>(B_write[i]);
    out[i] = static_cast<scalar_t>(next);
}


template <typename scalar_t>
__global__ void lightning_fused_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ readout,
    scalar_t* __restrict__ new_state,
    int64_t batch,
    int64_t channels,
    float gate_min,
    float gate_span,
    float eps) {
    const int64_t b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t qbase = b * (3 * channels);
    const int64_t sbase = b * channels;

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float gate_raw = static_cast<float>(qgv[qbase + channels + c]);
        const float value_raw = static_cast<float>(qgv[qbase + 2 * channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const scalar_t a_q = static_cast<scalar_t>(gate_min + gate_span * gate);
        const scalar_t v_q = static_cast<scalar_t>(__tanhf(value_raw));
        const float a = static_cast<float>(a_q);
        const scalar_t b_q = static_cast<scalar_t>((1.0f - a) * static_cast<float>(v_q));
        using acc_t = at::acc_type<scalar_t, true>;
        const acc_t next = static_cast<acc_t>(a_q) * static_cast<acc_t>(state[sbase + c])
                         + static_cast<acc_t>(b_q);
        const scalar_t next_q = static_cast<scalar_t>(next);
        new_state[sbase + c] = next_q;
        const float e = static_cast<float>(next_q);
        local_sum += e * e;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(channels) + eps);
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float q = static_cast<float>(qgv[qbase + c]);
        const float sig_q = 1.0f / (1.0f + __expf(-q));
        const float e = static_cast<float>(new_state[sbase + c]);
        readout[sbase + c] = static_cast<scalar_t>(sig_q * e * inv_rms);
    }
}

torch::Tensor thunder_scan_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t /*compass*/) {
    c10::cuda::CUDAGuard device_guard(A.device());
    auto out = torch::empty_like(B_write);

    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    const int64_t total = B * C;

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_scan_cuda",
        [&] {
            direct_scan_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B,
                T,
                C);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



torch::Tensor thunder_scan_hierarchical_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(A.device());

    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    const int64_t G = (T + compass - 1) / compass;

    TORCH_CHECK(G <= 1024,
                "hierarchical Thunder scan supports at most 1024 chunks per level; "
                "increase compass or add a multi-level prefix scan for this shape");

    auto states = torch::empty_like(B_write);
    auto chunk_A = torch::empty({B, G, C}, A.options());
    auto chunk_B = torch::empty({B, G, C}, B_write.options());

    constexpr int work_threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + work_threads - 1) / work_threads);
    const int prefix_threads = hierarchical_next_pow2_threads(G);
    const size_t prefix_shared = static_cast<size_t>(2 * prefix_threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_scan_hierarchical_cuda",
        [&] {
            hierarchical_local_scan_ab_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_A.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, T, C, G, compass);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_chunk_prefix_scan_ab_kernel<scalar_t><<<
                static_cast<int>(B * C), prefix_threads, prefix_shared, stream>>>(
                chunk_A.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_apply_chunk_prefix_ab_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, T, C, G, compass);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return states;
}


std::vector<torch::Tensor> thunder_scan_local_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    const int64_t G = (T + compass - 1) / compass;

    auto local_states = torch::empty_like(B_write);
    auto chunk_A = torch::empty({B, G, C}, A.options());
    auto chunk_B = torch::empty({B, G, C}, B_write.options());

    constexpr int threads = 256;
    const int64_t jobs = B * G * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_scan_local_cuda",
        [&] {
            hierarchical_local_scan_ab_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(),
                local_states.data_ptr<scalar_t>(),
                chunk_A.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, T, C, G, compass);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {local_states, chunk_A, chunk_B};
}

std::vector<torch::Tensor> thunder_summary_scan_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    int64_t group_size) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t G = A.size(1);
    const int64_t C = A.size(2);
    const int64_t NG = (G + group_size - 1) / group_size;

    auto pref_A = torch::empty_like(A);
    auto pref_B = torch::empty_like(B_write);
    auto group_A = torch::empty({B, NG, C}, A.options());
    auto group_B = torch::empty({B, NG, C}, B_write.options());

    constexpr int threads = 256;
    const int64_t jobs = B * NG * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_summary_scan_cuda",
        [&] {
            planner_summary_scan_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), B_write.data_ptr<scalar_t>(),
                pref_A.data_ptr<scalar_t>(), pref_B.data_ptr<scalar_t>(),
                group_A.data_ptr<scalar_t>(), group_B.data_ptr<scalar_t>(),
                B, G, C, NG, group_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pref_A, pref_B, group_A, group_B};
}

std::vector<torch::Tensor> thunder_group_prefix_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t G = A.size(1);
    const int64_t C = A.size(2);

    auto out_A = torch::empty_like(A);
    auto out_B = torch::empty_like(B_write);

    constexpr int threads = 256;
    const int64_t jobs = B * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_group_prefix_cuda",
        [&] {
            planner_group_prefix_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), B_write.data_ptr<scalar_t>(),
                out_A.data_ptr<scalar_t>(), out_B.data_ptr<scalar_t>(),
                B, G, C);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_A, out_B};
}

std::vector<torch::Tensor> thunder_apply_group_cuda(
    const torch::Tensor& pref_A,
    const torch::Tensor& pref_B,
    const torch::Tensor& parent_A,
    const torch::Tensor& parent_B,
    int64_t group_size) {

    c10::cuda::CUDAGuard device_guard(pref_A.device());
    const int64_t B = pref_A.size(0);
    const int64_t G = pref_A.size(1);
    const int64_t C = pref_A.size(2);
    const int64_t NG = parent_A.size(1);

    auto out_A = torch::empty_like(pref_A);
    auto out_B = torch::empty_like(pref_B);

    constexpr int threads = 256;
    const int64_t jobs = B * G * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(pref_A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        pref_A.scalar_type(),
        "mlbricks_thunder_apply_group_cuda",
        [&] {
            planner_apply_group_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                pref_A.data_ptr<scalar_t>(), pref_B.data_ptr<scalar_t>(),
                parent_A.data_ptr<scalar_t>(), parent_B.data_ptr<scalar_t>(),
                out_A.data_ptr<scalar_t>(), out_B.data_ptr<scalar_t>(),
                B, G, C, NG, group_size);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out_A, out_B};
}

torch::Tensor thunder_apply_chunk_prefix_cuda(
    const torch::Tensor& A,
    const torch::Tensor& local_states,
    const torch::Tensor& chunk_B_prefix,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    const int64_t G = chunk_B_prefix.size(1);
    auto states = local_states.clone();

    constexpr int threads = 256;
    const int64_t jobs = B * G * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_apply_chunk_prefix_cuda",
        [&] {
            hierarchical_apply_chunk_prefix_ab_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                chunk_B_prefix.data_ptr<scalar_t>(), B, T, C, G, compass);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return states;
}

std::vector<torch::Tensor> thunder_scan_backward_chunked_cuda(
    const torch::Tensor& A,
    const torch::Tensor& states,
    const torch::Tensor& grad,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    const int64_t G = (T + compass - 1) / compass;

    auto fp32_options = A.options().dtype(torch::kFloat32);
    auto chunk_P = torch::empty({B, G, C}, fp32_options);
    auto chunk_Q = torch::empty({B, G, C}, fp32_options);
    auto boundary = torch::empty({B, G, C}, fp32_options);
    auto grad_A = torch::empty_like(A);
    auto grad_B = torch::empty_like(A);

    constexpr int threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + threads - 1) / threads);
    const int64_t bc_jobs = B * C;
    const int bc_blocks = static_cast<int>((bc_jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_scan_backward_chunked_cuda",
        [&] {
            planner_backward_summary_kernel<scalar_t><<<chunk_blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                chunk_P.data_ptr<float>(), chunk_Q.data_ptr<float>(),
                B, T, C, G, compass);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            planner_backward_boundary_kernel<<<bc_blocks, threads, 0, stream>>>(
                chunk_P.data_ptr<float>(), chunk_Q.data_ptr<float>(),
                boundary.data_ptr<float>(), B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            planner_backward_apply_kernel<scalar_t><<<chunk_blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                boundary.data_ptr<float>(), grad_A.data_ptr<scalar_t>(), grad_B.data_ptr<scalar_t>(),
                B, T, C, G, compass);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_A, grad_B};
}

std::vector<torch::Tensor> thunder_reverse_prepare_cuda(
    const torch::Tensor& A,
    const torch::Tensor& grad) {

    c10::cuda::CUDAGuard device_guard(A.device());
    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t C = A.size(2) * A.size(3);
    auto reverse_A = torch::empty_like(A);
    auto reverse_B = torch::empty_like(grad);

    constexpr int threads = 256;
    const int64_t jobs = B * T * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "mlbricks_thunder_reverse_prepare_cuda",
        [&] {
            planner_reverse_prepare_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
                reverse_A.data_ptr<scalar_t>(), reverse_B.data_ptr<scalar_t>(),
                B, T, C);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {reverse_A, reverse_B};
}

std::vector<torch::Tensor> thunder_reverse_finish_cuda(
    const torch::Tensor& grad_reverse,
    const torch::Tensor& states) {

    c10::cuda::CUDAGuard device_guard(grad_reverse.device());
    const int64_t B = grad_reverse.size(0);
    const int64_t T = grad_reverse.size(1);
    const int64_t C = grad_reverse.size(2) * grad_reverse.size(3);
    auto grad_A = torch::empty_like(grad_reverse);
    auto grad_B = torch::empty_like(grad_reverse);

    constexpr int threads = 256;
    const int64_t jobs = B * T * C;
    const int blocks = static_cast<int>((jobs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(grad_reverse.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        grad_reverse.scalar_type(),
        "mlbricks_thunder_reverse_finish_cuda",
        [&] {
            planner_reverse_finish_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                grad_reverse.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                grad_A.data_ptr<scalar_t>(), grad_B.data_ptr<scalar_t>(),
                B, T, C);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_A, grad_B};
}

torch::Tensor lightning_step_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B_write,
    const torch::Tensor& state) {
    c10::cuda::CUDAGuard device_guard(A.device());
    auto out = torch::empty_like(state);
    const int64_t N = state.numel();

    constexpr int threads = 256;
    const int blocks = static_cast<int>((N + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        state.scalar_type(),
        "mlbricks_lightning_step_cuda",
        [&] {
            lightning_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(),
                state.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                N);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


std::vector<torch::Tensor> lightning_fused_step_cuda(
    const torch::Tensor& qgv,
    const torch::Tensor& state,
    double gate_min,
    double gate_max,
    double eps) {
    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t C = qgv.size(1) / 3;
    auto readout = torch::empty({B, C}, qgv.options());
    auto new_state = torch::empty_like(state);

    constexpr int threads = 256;
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());
    const size_t shared_bytes = threads * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_lightning_fused_step_cuda",
        [&] {
            lightning_fused_qgv_kernel<scalar_t><<<B, threads, shared_bytes, stream>>>(
                qgv.data_ptr<scalar_t>(),
                state.data_ptr<scalar_t>(),
                readout.data_ptr<scalar_t>(),
                new_state.data_ptr<scalar_t>(),
                B, C, static_cast<float>(gate_min),
                static_cast<float>(gate_max - gate_min), static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {readout, new_state};
}


template <typename scalar_t>
__global__ void fused_affine_scan_from_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ states,
    int64_t batch,
    int64_t time,
    int64_t channels,
    float gate_min,
    float gate_span) {
    using acc_t = at::acc_type<scalar_t, true>;

    const int64_t bc = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * channels;
    if (bc >= total) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    acc_t state = acc_t(0);

    #pragma unroll 1
    for (int64_t t = 0; t < time; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;

        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float value_raw = static_cast<float>(qgv[qgv_row + 2 * channels + c]);

        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a_float = gate_min + gate_span * gate;
        const float v_float = __tanhf(value_raw);

        // Quantize A and V to the source dtype before the recurrent update. This
        // mirrors the FP16/BF16 Thunder path more closely than keeping all
        // elementwise intermediates in FP32.
        const scalar_t a_q = static_cast<scalar_t>(a_float);
        const scalar_t v_q = static_cast<scalar_t>(v_float);
        const float a = static_cast<float>(a_q);
        const float v = static_cast<float>(v_q);
        const scalar_t b_q = static_cast<scalar_t>((1.0f - a) * v);

        state = static_cast<acc_t>(a_q) * state + static_cast<acc_t>(b_q);
        states[state_idx] = static_cast<scalar_t>(state);
    }
}

template <typename scalar_t>
__global__ void fused_rms_sigmoid_readout_kernel(
    const scalar_t* __restrict__ qgv,
    const scalar_t* __restrict__ states,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t channels,
    float eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t base = row * channels;

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        local_sum += e * e;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(channels) + eps);
    const int64_t qgv_base = row * (3 * channels);

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float q = static_cast<float>(qgv[qgv_base + c]);
        const float sig_q = 1.0f / (1.0f + __expf(-q));
        const float e = static_cast<float>(states[base + c]);
        out[base + c] = static_cast<scalar_t>(sig_q * e * inv_rms);
    }
}




// -----------------------------------------------------------------------------
// v10 PyTorch-faithful elementwise math
//
// v9 showed that the hierarchical scan itself remains stable through depth 64,
// while gate/value preparation and RMS/readout are the dominant drift sources.
// The important detail is that the PyTorch FP16 path quantizes at tensor-op
// boundaries. v10 mirrors those boundaries while using the more accurate CUDA
// expf/tanhf implementations instead of __expf/__tanhf fast intrinsics.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__device__ __forceinline__ float v10_quantize(float x) {
    return static_cast<float>(static_cast<scalar_t>(x));
}

template <typename scalar_t>
__device__ __forceinline__ void v10_prepare_coefficients(
    float gate_raw,
    float value_raw,
    float gate_min,
    float gate_span,
    float* a_out,
    float* b_out) {

    // torch.sigmoid(gate_raw) -> source dtype
    const float gate = 1.0f / (1.0f + expf(-gate_raw));
    const float gate_q = v10_quantize<scalar_t>(gate);

    // gate_min + gate_span * gate consists of two tensor operations in the
    // original PyTorch expression, each producing an FP16/BF16 tensor.
    const float scaled_q = v10_quantize<scalar_t>(gate_span * gate_q);
    const float a = v10_quantize<scalar_t>(gate_min + scaled_q);

    // torch.tanh(value_raw) -> source dtype
    const float v = v10_quantize<scalar_t>(tanhf(value_raw));

    // (1.0 - A) * V is likewise two source-dtype tensor operations.
    const float one_minus_a = v10_quantize<scalar_t>(1.0f - a);
    const float bw = v10_quantize<scalar_t>(one_minus_a * v);

    *a_out = a;
    *b_out = bw;
}

template <typename scalar_t>
__device__ __forceinline__ float v10_prepare_a_only(
    float gate_raw,
    float gate_min,
    float gate_span) {
    const float gate = 1.0f / (1.0f + expf(-gate_raw));
    const float gate_q = v10_quantize<scalar_t>(gate);
    const float scaled_q = v10_quantize<scalar_t>(gate_span * gate_q);
    return v10_quantize<scalar_t>(gate_min + scaled_q);
}

template <typename scalar_t>
__global__ void hierarchical_local_scan_precise_gate_from_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ local_states,
    scalar_t* __restrict__ chunk_A,
    scalar_t* __restrict__ chunk_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float state = 0.0f;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;
        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float value_raw = static_cast<float>(qgv[qgv_row + 2 * channels + c]);

        float a = 0.0f;
        float bw = 0.0f;
        v10_prepare_coefficients<scalar_t>(
            gate_raw, value_raw, gate_min, gate_span, &a, &bw);

        // Keep the proven v3 recurrence semantics unchanged.
        state = v10_quantize<scalar_t>(a * state + bw);
        transition = v10_quantize<scalar_t>(a * transition);
        local_states[state_idx] = static_cast<scalar_t>(state);
    }

    const int64_t chunk_idx = (b * chunks + g) * channels + c;
    chunk_A[chunk_idx] = static_cast<scalar_t>(transition);
    chunk_B[chunk_idx] = static_cast<scalar_t>(state);
}

template <typename scalar_t>
__global__ void hierarchical_apply_chunk_prefix_precise_gate_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ states,
    const scalar_t* __restrict__ chunk_B_prefix,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    if (g == 0) return;

    const int64_t prev_chunk_idx = (b * chunks + (g - 1)) * channels + c;
    const float chunk_init = static_cast<float>(chunk_B_prefix[prev_chunk_idx]);
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;
        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float a = v10_prepare_a_only<scalar_t>(
            gate_raw, gate_min, gate_span);
        transition = v10_quantize<scalar_t>(a * transition);

        const float local_state = static_cast<float>(states[state_idx]);
        states[state_idx] = static_cast<scalar_t>(
            v10_quantize<scalar_t>(transition * chunk_init + local_state));
    }
}

template <typename scalar_t>
__global__ void fused_rms_sigmoid_readout_precise_kernel(
    const scalar_t* __restrict__ qgv,
    const scalar_t* __restrict__ states,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t channels,
    float eps) {

    const int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t base = row * channels;

    // E.pow(2) first materializes a source-dtype tensor in PyTorch. Quantize
    // each square before the reduction, then accumulate the reduction in FP32.
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        local_sum += v10_quantize<scalar_t>(e * e);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // mean -> +eps -> rsqrt are each source-dtype tensor results in the
    // original code, so mirror those boundaries explicitly.
    const float mean_q = v10_quantize<scalar_t>(
        shared[0] / static_cast<float>(channels));
    const float mean_eps_q = v10_quantize<scalar_t>(mean_q + eps);
    const float inv_rms_q = v10_quantize<scalar_t>(rsqrtf(mean_eps_q));
    const int64_t qgv_base = row * (3 * channels);

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        const float norm_q = v10_quantize<scalar_t>(e * inv_rms_q);

        const float q = static_cast<float>(qgv[qgv_base + c]);
        const float sig_q = v10_quantize<scalar_t>(
            1.0f / (1.0f + expf(-q)));

        out[base + c] = static_cast<scalar_t>(
            v10_quantize<scalar_t>(sig_q * norm_q));
    }
}

template <typename scalar_t>
__global__ void thunder_prepare_ab_precise_debug_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ A,
    scalar_t* __restrict__ B_write,
    int64_t rows,
    int64_t channels,
    float gate_min,
    float gate_span) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * channels;
    if (i >= total) return;
    const int64_t row = i / channels;
    const int64_t c = i - row * channels;
    const int64_t qgv_base = row * (3 * channels);
    const float gate_raw = static_cast<float>(qgv[qgv_base + channels + c]);
    const float value_raw = static_cast<float>(qgv[qgv_base + 2 * channels + c]);
    float a = 0.0f;
    float bw = 0.0f;
    v10_prepare_coefficients<scalar_t>(
        gate_raw, value_raw, gate_min, gate_span, &a, &bw);
    A[i] = static_cast<scalar_t>(a);
    B_write[i] = static_cast<scalar_t>(bw);
}

template <typename scalar_t>
__global__ void thunder_readout_precise_debug_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ states,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t channels,
    float eps) {

    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t base = row * channels;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        local_sum += v10_quantize<scalar_t>(e * e);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean_q = v10_quantize<scalar_t>(
        shared[0] / static_cast<float>(channels));
    const float mean_eps_q = v10_quantize<scalar_t>(mean_q + eps);
    const float inv_rms_q = v10_quantize<scalar_t>(rsqrtf(mean_eps_q));
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        const float norm_q = v10_quantize<scalar_t>(e * inv_rms_q);
        const float qv = static_cast<float>(q[base + c]);
        const float sig_q = v10_quantize<scalar_t>(
            1.0f / (1.0f + expf(-qv)));
        out[base + c] = static_cast<scalar_t>(
            v10_quantize<scalar_t>(sig_q * norm_q));
    }
}

// -----------------------------------------------------------------------------
// v9 component-isolation debug operators
//
// These operators intentionally expose the exact CUDA elementwise math used by
// the fused Thunder path without changing production dispatch. They let the
// validation suite test gate/value preparation, the hierarchical recurrence,
// and RMS/sigmoid readout independently.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void thunder_prepare_ab_debug_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ A,
    scalar_t* __restrict__ B_write,
    int64_t rows,
    int64_t channels,
    float gate_min,
    float gate_span) {

    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * channels;
    if (i >= total) return;

    const int64_t row = i / channels;
    const int64_t c = i - row * channels;
    const int64_t qgv_base = row * (3 * channels);

    const float gate_raw = static_cast<float>(qgv[qgv_base + channels + c]);
    const float value_raw = static_cast<float>(qgv[qgv_base + 2 * channels + c]);

    // Keep this byte-for-byte equivalent in intent to the v3 hierarchical
    // fused preparation math: fast CUDA sigmoid/tanh, then quantize A, V and B
    // to the source scan dtype.
    const float gate = 1.0f / (1.0f + __expf(-gate_raw));
    const scalar_t a_q = static_cast<scalar_t>(gate_min + gate_span * gate);
    const scalar_t v_q = static_cast<scalar_t>(__tanhf(value_raw));
    const float a = static_cast<float>(a_q);
    const float v = static_cast<float>(v_q);
    const scalar_t b_q = static_cast<scalar_t>((1.0f - a) * v);

    A[i] = a_q;
    B_write[i] = b_q;
}

template <typename scalar_t>
__global__ void thunder_readout_debug_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ states,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t channels,
    float eps) {

    const int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t base = row * channels;

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = static_cast<float>(states[base + c]);
        local_sum += e * e;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(channels) + eps);

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float qv = static_cast<float>(q[base + c]);
        const float sig_q = 1.0f / (1.0f + __expf(-qv));
        const float e = static_cast<float>(states[base + c]);
        out[base + c] = static_cast<scalar_t>(sig_q * e * inv_rms);
    }
}

std::vector<torch::Tensor> thunder_prepare_ab_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max) {

    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t rows = B * T;

    auto A = torch::empty({B, T, C}, qgv.options());
    auto B_write = torch::empty({B, T, C}, qgv.options());

    constexpr int threads = 256;
    const int64_t total = rows * C;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());
    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_prepare_ab_debug_cuda",
        [&] {
            thunder_prepare_ab_debug_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(),
                rows, C, gmin, gspan);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {A, B_write};
}

torch::Tensor thunder_readout_cuda(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps) {

    c10::cuda::CUDAGuard device_guard(q.device());
    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t C = q.size(2);
    const int64_t rows = B * T;
    auto out = torch::empty_like(states);

    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "mlbricks_thunder_readout_debug_cuda",
        [&] {
            thunder_readout_debug_kernel<scalar_t><<<
                static_cast<int>(rows), threads, shared_bytes, stream>>>(
                q.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows, C, feps);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor thunder_fused_readout_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t /*compass*/) {
    c10::cuda::CUDAGuard device_guard(qgv.device());

    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    auto states = torch::empty({B, T, C}, qgv.options());
    auto out = torch::empty({B, T, C}, qgv.options());

    constexpr int scan_threads = 256;
    const int64_t scan_total = B * C;
    const int scan_blocks = static_cast<int>((scan_total + scan_threads - 1) / scan_threads);
    const int64_t rows = B * T;
    constexpr int readout_threads = 256;
    const size_t shared_bytes = readout_threads * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());

    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_fused_readout_cuda",
        [&] {
            fused_affine_scan_from_qgv_kernel<scalar_t><<<
                scan_blocks, scan_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                B,
                T,
                C,
                gmin,
                gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            fused_rms_sigmoid_readout_kernel<scalar_t><<<
                static_cast<int>(rows), readout_threads, shared_bytes, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows,
                C,
                feps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return out;
}


template <typename scalar_t>
__device__ __forceinline__ float quantize_to_float(float x) {
    const scalar_t q = static_cast<scalar_t>(x);
    return static_cast<float>(q);
}

template <typename scalar_t>
__global__ void hierarchical_local_scan_from_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ local_states,
    scalar_t* __restrict__ chunk_A,
    scalar_t* __restrict__ chunk_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;

    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float state = 0.0f;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;

        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float value_raw = static_cast<float>(qgv[qgv_row + 2 * channels + c]);

        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        const float v = quantize_to_float<scalar_t>(__tanhf(value_raw));
        const float bw = quantize_to_float<scalar_t>((1.0f - a) * v);

        // Mirror the FP16/BF16 Thunder recurrence by quantizing each local step.
        state = quantize_to_float<scalar_t>(a * state + bw);
        transition = quantize_to_float<scalar_t>(a * transition);
        local_states[state_idx] = static_cast<scalar_t>(state);
    }

    const int64_t chunk_idx = (b * chunks + g) * channels + c;
    chunk_A[chunk_idx] = static_cast<scalar_t>(transition);
    chunk_B[chunk_idx] = static_cast<scalar_t>(state);
}

template <typename scalar_t>
__global__ void hierarchical_chunk_prefix_scan_kernel(
    scalar_t* __restrict__ chunk_A,
    scalar_t* __restrict__ chunk_B,
    int64_t batch,
    int64_t chunks,
    int64_t channels) {

    const int64_t bc = static_cast<int64_t>(blockIdx.x);
    const int64_t total_bc = batch * channels;
    if (bc >= total_bc) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    const int g = threadIdx.x;

    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = shared + blockDim.x;

    float a = 1.0f;
    float bw = 0.0f;
    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        a = static_cast<float>(chunk_A[idx]);
        bw = static_cast<float>(chunk_B[idx]);
    }
    sA[g] = a;
    sB[g] = bw;
    __syncthreads();

    // Inclusive Hillis-Steele scan over affine summaries.
    // Pair composition for left prefix P and current C is:
    //   C o P = (A_C*A_P, A_C*B_P + B_C)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const float curA = sA[g];
        const float curB = sB[g];
        float prevA = 1.0f;
        float prevB = 0.0f;
        if (g >= offset) {
            prevA = sA[g - offset];
            prevB = sB[g - offset];
        }
        __syncthreads();
        if (g >= offset) {
            const float nextA = quantize_to_float<scalar_t>(curA * prevA);
            const float nextB = quantize_to_float<scalar_t>(curA * prevB + curB);
            sA[g] = nextA;
            sB[g] = nextB;
        }
        __syncthreads();
    }

    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        chunk_A[idx] = static_cast<scalar_t>(sA[g]);
        chunk_B[idx] = static_cast<scalar_t>(sB[g]);
    }
}

template <typename scalar_t>
__global__ void hierarchical_apply_chunk_prefix_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ states,
    const scalar_t* __restrict__ chunk_B_prefix,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;

    if (g == 0) return;

    const int64_t prev_chunk_idx = (b * chunks + (g - 1)) * channels + c;
    const float chunk_init = static_cast<float>(chunk_B_prefix[prev_chunk_idx]);

    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;

        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        transition = quantize_to_float<scalar_t>(a * transition);

        const float local_state = static_cast<float>(states[state_idx]);
        states[state_idx] = static_cast<scalar_t>(
            quantize_to_float<scalar_t>(transition * chunk_init + local_state));
    }
}

static int next_power_of_two_threads(int64_t n) {
    int threads = 1;
    while (threads < n && threads < 1024) threads <<= 1;
    return threads;
}


// -----------------------------------------------------------------------------
// v8 precision-stability kernels
//
// mixed32:
//   * local recurrence remains quantized exactly like v3
//   * chunk summaries and the global associative prefix stay FP32
//   * prefix correction uses the FP32 chunk state, then stores FP16/BF16 states
//
// full32:
//   * A/V/B coefficients are still quantized to the source scan dtype
//   * all recurrent state/transition accumulation stays FP32
//   * chunk summaries/prefixes stay FP32
//   * RMS/readout consumes FP32 states and emits the original scan dtype
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void hierarchical_local_scan_mixed32_from_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ local_states,
    float* __restrict__ chunk_A,
    float* __restrict__ chunk_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float state = 0.0f;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;

        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float value_raw = static_cast<float>(qgv[qgv_row + 2 * channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        const float v = quantize_to_float<scalar_t>(__tanhf(value_raw));
        const float bw = quantize_to_float<scalar_t>((1.0f - a) * v);

        // Keep v3 local semantics, but do not quantize the stored chunk summary.
        state = quantize_to_float<scalar_t>(a * state + bw);
        transition = quantize_to_float<scalar_t>(a * transition);
        local_states[state_idx] = static_cast<scalar_t>(state);
    }

    const int64_t chunk_idx = (b * chunks + g) * channels + c;
    chunk_A[chunk_idx] = transition;
    chunk_B[chunk_idx] = state;
}

__global__ void hierarchical_chunk_prefix_scan_fp32_kernel(
    float* __restrict__ chunk_A,
    float* __restrict__ chunk_B,
    int64_t batch,
    int64_t chunks,
    int64_t channels) {

    const int64_t bc = static_cast<int64_t>(blockIdx.x);
    const int64_t total_bc = batch * channels;
    if (bc >= total_bc) return;

    const int64_t b = bc / channels;
    const int64_t c = bc - b * channels;
    const int g = threadIdx.x;

    extern __shared__ float shared[];
    float* sA = shared;
    float* sB = shared + blockDim.x;

    float a = 1.0f;
    float bw = 0.0f;
    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        a = chunk_A[idx];
        bw = chunk_B[idx];
    }
    sA[g] = a;
    sB[g] = bw;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        const float curA = sA[g];
        const float curB = sB[g];
        float prevA = 1.0f;
        float prevB = 0.0f;
        if (g >= offset) {
            prevA = sA[g - offset];
            prevB = sB[g - offset];
        }
        __syncthreads();
        if (g >= offset) {
            sA[g] = curA * prevA;
            sB[g] = curA * prevB + curB;
        }
        __syncthreads();
    }

    if (g < chunks) {
        const int64_t idx = (b * chunks + g) * channels + c;
        chunk_A[idx] = sA[g];
        chunk_B[idx] = sB[g];
    }
}

template <typename scalar_t>
__global__ void hierarchical_apply_chunk_prefix_mixed32_kernel(
    const scalar_t* __restrict__ qgv,
    scalar_t* __restrict__ states,
    const float* __restrict__ chunk_B_prefix,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    if (g == 0) return;

    const int64_t prev_chunk_idx = (b * chunks + (g - 1)) * channels + c;
    const float chunk_init = chunk_B_prefix[prev_chunk_idx];
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;
        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        transition = quantize_to_float<scalar_t>(a * transition);

        const float local_state = static_cast<float>(states[state_idx]);
        states[state_idx] = static_cast<scalar_t>(transition * chunk_init + local_state);
    }
}

template <typename scalar_t>
__global__ void hierarchical_local_scan_full32_from_qgv_kernel(
    const scalar_t* __restrict__ qgv,
    float* __restrict__ local_states,
    float* __restrict__ chunk_A,
    float* __restrict__ chunk_B,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;

    float state = 0.0f;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;

        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float value_raw = static_cast<float>(qgv[qgv_row + 2 * channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));

        // Preserve the source FP16/BF16 coefficients, but accumulate state in FP32.
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        const float v = quantize_to_float<scalar_t>(__tanhf(value_raw));
        const float bw = quantize_to_float<scalar_t>((1.0f - a) * v);

        state = a * state + bw;
        transition = a * transition;
        local_states[state_idx] = state;
    }

    const int64_t chunk_idx = (b * chunks + g) * channels + c;
    chunk_A[chunk_idx] = transition;
    chunk_B[chunk_idx] = state;
}

template <typename scalar_t>
__global__ void hierarchical_apply_chunk_prefix_full32_kernel(
    const scalar_t* __restrict__ qgv,
    float* __restrict__ states,
    const float* __restrict__ chunk_B_prefix,
    int64_t batch,
    int64_t time,
    int64_t channels,
    int64_t chunks,
    int64_t compass,
    float gate_min,
    float gate_span) {

    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * chunks * channels;
    if (job >= total) return;

    const int64_t c = job % channels;
    const int64_t tmp = job / channels;
    const int64_t g = tmp % chunks;
    const int64_t b = tmp / chunks;
    if (g == 0) return;

    const int64_t prev_chunk_idx = (b * chunks + (g - 1)) * channels + c;
    const float chunk_init = chunk_B_prefix[prev_chunk_idx];
    const int64_t t0 = g * compass;
    const int64_t t1 = (t0 + compass < time) ? (t0 + compass) : time;
    float transition = 1.0f;

    for (int64_t t = t0; t < t1; ++t) {
        const int64_t qgv_row = (b * time + t) * (3 * channels);
        const int64_t state_idx = (b * time + t) * channels + c;
        const float gate_raw = static_cast<float>(qgv[qgv_row + channels + c]);
        const float gate = 1.0f / (1.0f + __expf(-gate_raw));
        const float a = quantize_to_float<scalar_t>(gate_min + gate_span * gate);
        transition = a * transition;
        states[state_idx] = transition * chunk_init + states[state_idx];
    }
}

template <typename scalar_t>
__global__ void fused_rms_sigmoid_readout_fp32_state_kernel(
    const scalar_t* __restrict__ qgv,
    const float* __restrict__ states,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t channels,
    float eps) {

    const int64_t row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    const int64_t base = row * channels;
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float e = states[base + c];
        local_sum += e * e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(channels) + eps);
    const int64_t qgv_base = row * (3 * channels);
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float q = static_cast<float>(qgv[qgv_base + c]);
        const float sig_q = 1.0f / (1.0f + __expf(-q));
        out[base + c] = static_cast<scalar_t>(sig_q * states[base + c] * inv_rms);
    }
}


std::vector<torch::Tensor> thunder_prepare_ab_precise_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max) {

    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t rows = B * T;
    auto A = torch::empty({B, T, C}, qgv.options());
    auto B_write = torch::empty({B, T, C}, qgv.options());
    constexpr int threads = 256;
    const int64_t total = rows * C;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());
    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_prepare_ab_precise_cuda",
        [&] {
            thunder_prepare_ab_precise_debug_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                B_write.data_ptr<scalar_t>(), rows, C, gmin, gspan);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {A, B_write};
}

torch::Tensor thunder_readout_precise_cuda(
    const torch::Tensor& q,
    const torch::Tensor& states,
    double eps) {

    c10::cuda::CUDAGuard device_guard(q.device());
    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t C = q.size(2);
    const int64_t rows = B * T;
    auto out = torch::empty_like(states);
    constexpr int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "mlbricks_thunder_readout_precise_cuda",
        [&] {
            thunder_readout_precise_debug_kernel<scalar_t><<<
                static_cast<int>(rows), threads, shared_bytes, stream>>>(
                q.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), rows, C, feps);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static torch::Tensor thunder_fused_readout_hierarchical_v10_impl_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass,
    bool precise_gate,
    bool precise_readout) {

    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t G = (T + compass - 1) / compass;
    TORCH_CHECK(G <= 1024,
                "v10 hierarchical Thunder supports at most 1024 chunks per level");

    auto states = torch::empty({B, T, C}, qgv.options());
    auto chunk_A = torch::empty({B, G, C}, qgv.options());
    auto chunk_B = torch::empty({B, G, C}, qgv.options());
    auto out = torch::empty({B, T, C}, qgv.options());

    constexpr int work_threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + work_threads - 1) / work_threads);
    const int prefix_threads = next_power_of_two_threads(G);
    const int64_t rows = B * T;
    constexpr int readout_threads = 256;
    const size_t prefix_shared = static_cast<size_t>(2 * prefix_threads) * sizeof(float);
    const size_t readout_shared = static_cast<size_t>(readout_threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());
    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_fused_readout_hierarchical_v10_cuda",
        [&] {
            if (precise_gate) {
                hierarchical_local_scan_precise_gate_from_qgv_kernel<scalar_t><<<
                    chunk_blocks, work_threads, 0, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    chunk_A.data_ptr<scalar_t>(), chunk_B.data_ptr<scalar_t>(),
                    B, T, C, G, compass, gmin, gspan);
            } else {
                hierarchical_local_scan_from_qgv_kernel<scalar_t><<<
                    chunk_blocks, work_threads, 0, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    chunk_A.data_ptr<scalar_t>(), chunk_B.data_ptr<scalar_t>(),
                    B, T, C, G, compass, gmin, gspan);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_chunk_prefix_scan_kernel<scalar_t><<<
                static_cast<int>(B * C), prefix_threads, prefix_shared, stream>>>(
                chunk_A.data_ptr<scalar_t>(), chunk_B.data_ptr<scalar_t>(), B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            if (precise_gate) {
                hierarchical_apply_chunk_prefix_precise_gate_kernel<scalar_t><<<
                    chunk_blocks, work_threads, 0, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    chunk_B.data_ptr<scalar_t>(), B, T, C, G, compass, gmin, gspan);
            } else {
                hierarchical_apply_chunk_prefix_kernel<scalar_t><<<
                    chunk_blocks, work_threads, 0, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    chunk_B.data_ptr<scalar_t>(), B, T, C, G, compass, gmin, gspan);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            if (precise_readout) {
                fused_rms_sigmoid_readout_precise_kernel<scalar_t><<<
                    static_cast<int>(rows), readout_threads, readout_shared, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(), rows, C, feps);
            } else {
                fused_rms_sigmoid_readout_kernel<scalar_t><<<
                    static_cast<int>(rows), readout_threads, readout_shared, stream>>>(
                    qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(), rows, C, feps);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    return out;
}

torch::Tensor thunder_fused_readout_hierarchical_precise_gate_cuda(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    return thunder_fused_readout_hierarchical_v10_impl_cuda(
        qgv, embd, gate_min, gate_max, eps, compass, true, false);
}

torch::Tensor thunder_fused_readout_hierarchical_precise_readout_cuda(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    return thunder_fused_readout_hierarchical_v10_impl_cuda(
        qgv, embd, gate_min, gate_max, eps, compass, false, true);
}

torch::Tensor thunder_fused_readout_hierarchical_precise_both_cuda(
    const torch::Tensor& qgv, int64_t embd, double gate_min, double gate_max,
    double eps, int64_t compass) {
    return thunder_fused_readout_hierarchical_v10_impl_cuda(
        qgv, embd, gate_min, gate_max, eps, compass, true, true);
}

torch::Tensor thunder_fused_readout_hierarchical_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(qgv.device());

    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t G = (T + compass - 1) / compass;

    // One CUDA block scans all chunk summaries for a single (batch, channel).
    // This implementation supports up to 1024 chunks per level, i.e. 16K
    // tokens with compass=16. Larger shapes remain on the native
    // direct fused path until the recursive multi-level variant is added.
    TORCH_CHECK(G <= 1024,
                "hierarchical Thunder supports at most 1024 chunks per level; "
                "increase compass or use direct native mode for this shape");

    auto states = torch::empty({B, T, C}, qgv.options());
    auto chunk_A = torch::empty({B, G, C}, qgv.options());
    auto chunk_B = torch::empty({B, G, C}, qgv.options());
    auto out = torch::empty({B, T, C}, qgv.options());

    constexpr int work_threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + work_threads - 1) / work_threads);
    const int prefix_threads = next_power_of_two_threads(G);
    const int64_t rows = B * T;
    constexpr int readout_threads = 256;

    const size_t prefix_shared = static_cast<size_t>(2 * prefix_threads) * sizeof(float);
    const size_t readout_shared = static_cast<size_t>(readout_threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());

    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_fused_readout_hierarchical_cuda",
        [&] {
            hierarchical_local_scan_from_qgv_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_A.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_chunk_prefix_scan_kernel<scalar_t><<<
                static_cast<int>(B * C), prefix_threads, prefix_shared, stream>>>(
                chunk_A.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_apply_chunk_prefix_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_B.data_ptr<scalar_t>(),
                B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            fused_rms_sigmoid_readout_kernel<scalar_t><<<
                static_cast<int>(rows), readout_threads, readout_shared, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows, C, feps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return out;
}


torch::Tensor thunder_fused_readout_hierarchical_mixed32_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t G = (T + compass - 1) / compass;
    TORCH_CHECK(G <= 1024,
                "v8 mixed32 supports at most 1024 chunks per level");

    auto states = torch::empty({B, T, C}, qgv.options());
    auto fp32_options = qgv.options().dtype(torch::kFloat32);
    auto chunk_A = torch::empty({B, G, C}, fp32_options);
    auto chunk_B = torch::empty({B, G, C}, fp32_options);
    auto out = torch::empty({B, T, C}, qgv.options());

    constexpr int work_threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + work_threads - 1) / work_threads);
    const int prefix_threads = next_power_of_two_threads(G);
    const int64_t rows = B * T;
    constexpr int readout_threads = 256;
    const size_t prefix_shared = static_cast<size_t>(2 * prefix_threads) * sizeof(float);
    const size_t readout_shared = static_cast<size_t>(readout_threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());

    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_fused_readout_hierarchical_mixed32_cuda",
        [&] {
            hierarchical_local_scan_mixed32_from_qgv_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_A.data_ptr<float>(),
                chunk_B.data_ptr<float>(),
                B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_chunk_prefix_scan_fp32_kernel<<<
                static_cast<int>(B * C), prefix_threads, prefix_shared, stream>>>(
                chunk_A.data_ptr<float>(), chunk_B.data_ptr<float>(), B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_apply_chunk_prefix_mixed32_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(),
                states.data_ptr<scalar_t>(),
                chunk_B.data_ptr<float>(),
                B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            fused_rms_sigmoid_readout_kernel<scalar_t><<<
                static_cast<int>(rows), readout_threads, readout_shared, stream>>>(
                qgv.data_ptr<scalar_t>(), states.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), rows, C, feps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return out;
}


torch::Tensor thunder_fused_readout_hierarchical_full32_cuda(
    const torch::Tensor& qgv,
    int64_t embd,
    double gate_min,
    double gate_max,
    double eps,
    int64_t compass) {

    c10::cuda::CUDAGuard device_guard(qgv.device());
    const int64_t B = qgv.size(0);
    const int64_t T = qgv.size(1);
    const int64_t C = embd;
    const int64_t G = (T + compass - 1) / compass;
    TORCH_CHECK(G <= 1024,
                "v8 full32 supports at most 1024 chunks per level");

    auto fp32_options = qgv.options().dtype(torch::kFloat32);
    auto states = torch::empty({B, T, C}, fp32_options);
    auto chunk_A = torch::empty({B, G, C}, fp32_options);
    auto chunk_B = torch::empty({B, G, C}, fp32_options);
    auto out = torch::empty({B, T, C}, qgv.options());

    constexpr int work_threads = 256;
    const int64_t chunk_jobs = B * G * C;
    const int chunk_blocks = static_cast<int>((chunk_jobs + work_threads - 1) / work_threads);
    const int prefix_threads = next_power_of_two_threads(G);
    const int64_t rows = B * T;
    constexpr int readout_threads = 256;
    const size_t prefix_shared = static_cast<size_t>(2 * prefix_threads) * sizeof(float);
    const size_t readout_shared = static_cast<size_t>(readout_threads) * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(qgv.get_device());

    const float gmin = static_cast<float>(gate_min);
    const float gspan = static_cast<float>(gate_max - gate_min);
    const float feps = static_cast<float>(eps);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        qgv.scalar_type(),
        "mlbricks_thunder_fused_readout_hierarchical_full32_cuda",
        [&] {
            hierarchical_local_scan_full32_from_qgv_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(), states.data_ptr<float>(),
                chunk_A.data_ptr<float>(), chunk_B.data_ptr<float>(),
                B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_chunk_prefix_scan_fp32_kernel<<<
                static_cast<int>(B * C), prefix_threads, prefix_shared, stream>>>(
                chunk_A.data_ptr<float>(), chunk_B.data_ptr<float>(), B, G, C);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            hierarchical_apply_chunk_prefix_full32_kernel<scalar_t><<<
                chunk_blocks, work_threads, 0, stream>>>(
                qgv.data_ptr<scalar_t>(), states.data_ptr<float>(),
                chunk_B.data_ptr<float>(), B, T, C, G, compass, gmin, gspan);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            fused_rms_sigmoid_readout_fp32_state_kernel<scalar_t><<<
                static_cast<int>(rows), readout_threads, readout_shared, stream>>>(
                qgv.data_ptr<scalar_t>(), states.data_ptr<float>(),
                out.data_ptr<scalar_t>(), rows, C, feps);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });

    return out;
}


__global__ void hybrid_reduce_fp16_partials_fp32_kernel(
    const at::Half* __restrict__ partials,
    at::Half* __restrict__ out,
    int64_t outputs,
    int64_t groups) {

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= outputs) return;

    float sum = 0.0f;
    #pragma unroll 1
    for (int64_t g = 0; g < groups; ++g) {
        sum += static_cast<float>(partials[g * outputs + idx]);
    }
    out[idx] = static_cast<at::Half>(sum);
}


torch::Tensor linear_hybrid_accum_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    int64_t chunk) {

    TORCH_CHECK(x.is_cuda() && weight.is_cuda(),
                "hybrid GEMM requires CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Half &&
                weight.scalar_type() == at::ScalarType::Half,
                "hybrid GEMM currently requires FP16 tensors");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "hybrid GEMM requires contiguous tensors");
    TORCH_CHECK(x.dim() == 3, "hybrid GEMM expects x shape [B,T,K]");
    TORCH_CHECK(weight.dim() == 2, "hybrid GEMM expects weight shape [N,K]");
    TORCH_CHECK(chunk > 0, "hybrid GEMM chunk must be positive");

    const int64_t B = x.size(0);
    const int64_t T = x.size(1);
    const int64_t K64 = x.size(2);
    const int64_t N64 = weight.size(0);
    const int64_t M64 = B * T;

    TORCH_CHECK(weight.size(1) == K64,
                "weight inner dimension must match x");

    // v7.1 tail-aware partitioning. Most chunks have exactly `chunk` K
    // elements and are evaluated together with one strided-batched GEMM.
    // If K is not divisible by chunk, one final GEMM evaluates the smaller
    // tail. All partial matrices are still reduced in FP32 below.
    const int64_t full_groups64 = K64 / chunk;
    const int64_t tail64 = K64 % chunk;
    const int64_t groups64 = full_groups64 + (tail64 != 0 ? 1 : 0);
    TORCH_CHECK(groups64 > 0,
                "hybrid GEMM requires at least one K chunk");
    TORCH_CHECK(M64 <= std::numeric_limits<int>::max() &&
                N64 <= std::numeric_limits<int>::max() &&
                K64 <= std::numeric_limits<int>::max() &&
                chunk <= std::numeric_limits<int>::max() &&
                full_groups64 <= std::numeric_limits<int>::max() &&
                groups64 <= std::numeric_limits<int>::max(),
                "hybrid GEMM dimensions exceed cuBLAS int32 limits");

    auto out = torch::empty({B, T, N64}, x.options());
    if (M64 == 0 || N64 == 0 || K64 == 0) {
        return out;
    }

    // Each batch item is one K-slice of the same logical GEMM. cuBLAS computes
    // every slice with FP16 accumulation and stores one FP16 partial matrix.
    // A single CUDA kernel then sums those partial matrices in FP32.
    auto partials = torch::empty({groups64, B, T, N64}, x.options());

    c10::cuda::CUDAGuard device_guard(x.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::PointerModeGuard pointer_mode(
        handle, CUBLAS_POINTER_MODE_HOST);

    const int M = static_cast<int>(M64);
    const int N = static_cast<int>(N64);

    // Row-major target: out[M,N] = x[M,K] @ weight[N,K]^T.
    // As in the direct v5 GEMM, column-major cuBLAS sees
    // partial^T[N,M] = weight_slice[N,KC] @ x_slice^T[KC,M].
    // Successive batches advance by `chunk` elements into K while retaining
    // lda=K, so each batch views a different contiguous K-slice from every row.
    const at::Half alpha = at::Half(1.0f);
    const at::Half beta = at::Half(0.0f);
    const long long strideA = static_cast<long long>(chunk);
    const long long strideB = static_cast<long long>(chunk);
    const long long strideC = static_cast<long long>(M64 * N64);

    if (full_groups64 > 0) {
        const int KC = static_cast<int>(chunk);
        const int full_groups = static_cast<int>(full_groups64);
        TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            M,
            KC,
            &alpha,
            weight.data_ptr<at::Half>(),
            CUDA_R_16F,
            static_cast<int>(K64),
            strideA,
            x.data_ptr<at::Half>(),
            CUDA_R_16F,
            static_cast<int>(K64),
            strideB,
            &beta,
            partials.data_ptr<at::Half>(),
            CUDA_R_16F,
            N,
            strideC,
            full_groups,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT));
    }

    if (tail64 != 0) {
        const int tail = static_cast<int>(tail64);
        const int64_t k_offset = full_groups64 * chunk;
        const int64_t c_offset = full_groups64 * strideC;

        TORCH_CUDABLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            N,
            M,
            tail,
            &alpha,
            weight.data_ptr<at::Half>() + k_offset,
            CUDA_R_16F,
            static_cast<int>(K64),
            x.data_ptr<at::Half>() + k_offset,
            CUDA_R_16F,
            static_cast<int>(K64),
            &beta,
            partials.data_ptr<at::Half>() + c_offset,
            CUDA_R_16F,
            N,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT));
    }

    constexpr int threads = 256;
    const int64_t outputs = M64 * N64;
    const int blocks = static_cast<int>((outputs + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    hybrid_reduce_fp16_partials_fp32_kernel<<<blocks, threads, 0, stream>>>(
        partials.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        outputs,
        groups64);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


template <typename scalar_t>
__global__ void residual_layer_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ residual,
    scalar_t* __restrict__ normalized,
    int64_t rows,
    int64_t channels,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sq = shared + blockDim.x;
    const int64_t base = row * channels;
    float local_sum = 0.0f;
    float local_sq = 0.0f;

    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float r = static_cast<float>(x[base + c])
                      + static_cast<float>(update[base + c]);
        const scalar_t r_q = static_cast<scalar_t>(r);
        residual[base + c] = r_q;
        const float rq = static_cast<float>(r_q);
        local_sum += rq;
        local_sq += rq * rq;
    }
    s_sum[threadIdx.x] = local_sum;
    s_sq[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean = s_sum[0] / static_cast<float>(channels);
    const float mean_sq = s_sq[0] / static_cast<float>(channels);
    const float inv_std = rsqrtf(fmaxf(mean_sq - mean * mean, 0.0f) + eps);
    for (int64_t c = threadIdx.x; c < channels; c += blockDim.x) {
        const float r = static_cast<float>(residual[base + c]);
        const float n = (r - mean) * inv_std;
        const float out = n * static_cast<float>(weight[c])
                        + static_cast<float>(bias[c]);
        normalized[base + c] = static_cast<scalar_t>(out);
    }
}

std::vector<torch::Tensor> residual_layer_norm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& update,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t channels = x.size(-1);
    const int64_t rows = x.numel() / channels;
    auto residual = torch::empty_like(x);
    auto normalized = torch::empty_like(x);

    constexpr int threads = 256;
    const size_t shared_bytes = 2 * threads * sizeof(float);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "mlbricks_residual_layer_norm_cuda",
        [&] {
            residual_layer_norm_kernel<scalar_t><<<rows, threads, shared_bytes, stream>>>(
                x.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(), normalized.data_ptr<scalar_t>(),
                rows, channels, static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual, normalized};
}


template <typename scalar_t, typename scale_t>
__global__ void elastic_linear_packed_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ packed,
    const scale_t* __restrict__ scales,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int bits,
    int group_size,
    bool has_bias) {
    const int64_t job = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * out_features;
    if (job >= total) return;
    const int64_t row = job / out_features;
    const int64_t o = job - row * out_features;
    const int mask = (1 << bits) - 1;
    const int qmin = -(1 << (bits - 1));
    float acc = 0.0f;

    const int64_t xbase = row * in_features;
    const int64_t wbase = o * in_features;
    #pragma unroll 1
    for (int64_t k = 0; k < in_features; ++k) {
        const int64_t wi = wbase + k;
        const int64_t bit_pos = wi * static_cast<int64_t>(bits);
        const int64_t byte_index = bit_pos >> 3;
        const int offset = static_cast<int>(bit_pos & 7);
        uint16_t word = static_cast<uint16_t>(packed[byte_index]);
        if (offset + bits > 8) {
            word |= static_cast<uint16_t>(packed[byte_index + 1]) << 8;
        }
        const int uq = (word >> offset) & mask;
        const int q = uq + qmin;
        const int64_t group = wi / group_size;
        const float w = static_cast<float>(q) * static_cast<float>(scales[group]);
        acc += static_cast<float>(x[xbase + k]) * w;
    }
    if (has_bias) acc += static_cast<float>(bias[o]);
    out[job] = static_cast<scalar_t>(acc);
}

torch::Tensor elastic_linear_packed_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    const torch::Tensor& bias,
    int64_t bits,
    int64_t group_size,
    int64_t out_features,
    int64_t in_features) {
    c10::cuda::CUDAGuard device_guard(x.device());
    const int64_t rows = x.numel() / in_features;
    auto out_shape = x.sizes().vec();
    out_shape.back() = out_features;
    auto out = torch::empty(out_shape, x.options());
    const bool has_bias = bias.numel() != 0;
    constexpr int threads = 256;
    const int64_t total = rows * out_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "mlbricks_elastic_linear_packed_x",
        [&] {
            using x_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half,
                at::ScalarType::BFloat16,
                scales.scalar_type(),
                "mlbricks_elastic_linear_packed_scales",
                [&] {
                    elastic_linear_packed_kernel<x_t, scalar_t><<<blocks, threads, 0, stream>>>(
                        x.data_ptr<x_t>(), packed.data_ptr<uint8_t>(),
                        scales.data_ptr<scalar_t>(),
                        has_bias ? bias.data_ptr<x_t>() : nullptr,
                        out.data_ptr<x_t>(), rows, in_features, out_features,
                        static_cast<int>(bits), static_cast<int>(group_size), has_bias);
                });
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace mlbricks
