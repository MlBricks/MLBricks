#include "../residualbrick.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace residualbrick {
namespace {

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float controller_scale_from_moments(
    float residual_sq,
    float update_sq,
    float residual_update,
    float inv_width,
    float update_ratio,
    float stream_ratio,
    float update_softness,
    float stream_softness,
    float eps) {
    const float residual_rms = sqrtf(residual_sq * inv_width + eps);
    const float raw_update_rms = sqrtf(update_sq * inv_width + eps);

    const float allowed_update_rms = update_ratio * residual_rms;
    const float hard_update_scale = fminf(
        allowed_update_rms / (raw_update_rms + eps), 1.0f);
    const float update_pressure = raw_update_rms / (allowed_update_rms + eps);
    const float update_gate = sigmoidf_fast(
        update_softness * (update_pressure - 1.0f));
    const float update_scale =
        1.0f - update_gate * (1.0f - hard_update_scale);

    float candidate_sq = residual_sq
        + 2.0f * update_scale * residual_update
        + update_scale * update_scale * update_sq;
    candidate_sq = fmaxf(candidate_sq, 0.0f);
    const float candidate_rms = sqrtf(candidate_sq * inv_width + eps);

    const float allowed_stream_rms = stream_ratio * residual_rms;
    const float hard_stream_scale = fminf(
        allowed_stream_rms / (candidate_rms + eps), 1.0f);
    const float stream_pressure = candidate_rms / (allowed_stream_rms + eps);
    const float stream_gate = sigmoidf_fast(
        stream_softness * (stream_pressure - 1.0f));
    const float stream_scale =
        1.0f - stream_gate * (1.0f - hard_stream_scale);

    return update_scale * stream_scale;
}

template <typename scalar_t>
__global__ void residual_controller_warp_kernel(
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t width,
    float update_ratio,
    float stream_ratio,
    float update_softness,
    float stream_softness,
    float eps) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    if (row >= rows) return;

    const int64_t base = row * width;
    float residual_sq = 0.0f;
    float update_sq = 0.0f;
    float residual_update = 0.0f;

    for (int64_t col = lane; col < width; col += 32) {
        const float r = static_cast<float>(residual[base + col]);
        const float u = static_cast<float>(update[base + col]);
        residual_sq = fmaf(r, r, residual_sq);
        update_sq = fmaf(u, u, update_sq);
        residual_update = fmaf(r, u, residual_update);
    }

    residual_sq = warp_reduce_sum(residual_sq);
    update_sq = warp_reduce_sum(update_sq);
    residual_update = warp_reduce_sum(residual_update);

    float combined_scale = 0.0f;
    if (lane == 0) {
        combined_scale = controller_scale_from_moments(
            residual_sq, update_sq, residual_update,
            1.0f / static_cast<float>(width),
            update_ratio, stream_ratio,
            update_softness, stream_softness, eps);
    }
    combined_scale = __shfl_sync(0xffffffff, combined_scale, 0);

    for (int64_t col = lane; col < width; col += 32) {
        const float r = static_cast<float>(residual[base + col]);
        const float u = static_cast<float>(update[base + col]);
        output[base + col] = static_cast<scalar_t>(fmaf(u, combined_scale, r));
    }
}

__device__ __forceinline__ void block_reduce_three(float& a, float& b, float& c) {
    __shared__ float warp_a[32];
    __shared__ float warp_b[32];
    __shared__ float warp_c[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + 31) >> 5;

    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    c = warp_reduce_sum(c);
    if (lane == 0) {
        warp_a[warp] = a;
        warp_b[warp] = b;
        warp_c[warp] = c;
    }
    __syncthreads();

    if (warp == 0) {
        a = lane < warp_count ? warp_a[lane] : 0.0f;
        b = lane < warp_count ? warp_b[lane] : 0.0f;
        c = lane < warp_count ? warp_c[lane] : 0.0f;
        a = warp_reduce_sum(a);
        b = warp_reduce_sum(b);
        c = warp_reduce_sum(c);
    }
}

template <typename scalar_t>
__global__ void residual_controller_block_kernel(
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ output,
    int64_t rows,
    int64_t width,
    float update_ratio,
    float stream_ratio,
    float update_softness,
    float stream_softness,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;
    const int64_t base = row * width;

    float residual_sq = 0.0f;
    float update_sq = 0.0f;
    float residual_update = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float r = static_cast<float>(residual[base + col]);
        const float u = static_cast<float>(update[base + col]);
        residual_sq = fmaf(r, r, residual_sq);
        update_sq = fmaf(u, u, update_sq);
        residual_update = fmaf(r, u, residual_update);
    }

    block_reduce_three(residual_sq, update_sq, residual_update);

    __shared__ float combined_scale_shared;
    if (threadIdx.x == 0) {
        combined_scale_shared = controller_scale_from_moments(
            residual_sq, update_sq, residual_update,
            1.0f / static_cast<float>(width),
            update_ratio, stream_ratio,
            update_softness, stream_softness, eps);
    }
    __syncthreads();

    const float combined_scale = combined_scale_shared;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float r = static_cast<float>(residual[base + col]);
        const float u = static_cast<float>(update[base + col]);
        output[base + col] = static_cast<scalar_t>(fmaf(u, combined_scale, r));
    }
}

int warps_per_block_for_rows(int64_t rows) {
    if (rows <= 1) return 1;
    if (rows <= 2) return 2;
    if (rows <= 4) return 4;
    return 8;
}

} // namespace

Tensor residual_controller_cuda(
    const Tensor& residual,
    const Tensor& update,
    double update_ratio,
    double stream_ratio,
    double update_softness,
    double stream_softness,
    double eps) {
    TORCH_CHECK(residual.is_cuda() && update.is_cuda(),
                "residual_controller_cuda expects CUDA tensors");
    TORCH_CHECK(residual.sizes() == update.sizes(), "residual/update shape mismatch");
    TORCH_CHECK(residual.scalar_type() == update.scalar_type(),
                "fused CUDA path requires residual/update to have the same dtype");
    TORCH_CHECK(residual.dim() > 0 && residual.size(-1) > 0,
                "last dimension must be non-empty for fused CUDA path");

    c10::cuda::CUDAGuard device_guard(residual.device());
    const auto residual_c = residual.contiguous();
    const auto update_c = update.contiguous();
    auto output = at::empty_like(residual_c);

    const int64_t width = residual_c.size(-1);
    const int64_t rows = residual_c.numel() / width;
    if (rows == 0) return output;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        residual_c.scalar_type(),
        "residualbrick_fused_controller_cuda",
        [&] {
            if (width <= 2048) {
                const int wpb = warps_per_block_for_rows(rows);
                const int threads = wpb * 32;
                const int64_t blocks64 = (rows + wpb - 1) / wpb;
                residual_controller_warp_kernel<scalar_t>
                    <<<static_cast<unsigned int>(blocks64), threads, 0, stream>>>(
                        residual_c.data_ptr<scalar_t>(),
                        update_c.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        rows, width,
                        static_cast<float>(update_ratio),
                        static_cast<float>(stream_ratio),
                        static_cast<float>(update_softness),
                        static_cast<float>(stream_softness),
                        static_cast<float>(eps));
            } else {
                residual_controller_block_kernel<scalar_t>
                    <<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
                        residual_c.data_ptr<scalar_t>(),
                        update_c.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        rows, width,
                        static_cast<float>(update_ratio),
                        static_cast<float>(stream_ratio),
                        static_cast<float>(update_softness),
                        static_cast<float>(stream_softness),
                        static_cast<float>(eps));
            }
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

} // namespace residualbrick
