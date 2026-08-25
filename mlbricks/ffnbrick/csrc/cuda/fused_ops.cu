#include "../ffnbrick.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ffnbrick {
namespace {

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float siluf_fast(float x) {
    return x * sigmoidf_fast(x);
}

template <typename scalar_t>
__global__ void silu_mul_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ out,
    int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float g = static_cast<float>(gate[idx]);
        const float v = static_cast<float>(value[idx]);
        out[idx] = static_cast<scalar_t>(siluf_fast(g) * v);
    }
}

template <typename scalar_t>
__global__ void silu_mul_packed_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t hidden_dim) {
    const int64_t n = rows * hidden_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int64_t row = idx / hidden_dim;
        const int64_t col = idx - row * hidden_dim;
        const int64_t base = row * (2 * hidden_dim);
        const float g = static_cast<float>(gate_up[base + col]);
        const float v = static_cast<float>(gate_up[base + hidden_dim + col]);
        out[idx] = static_cast<scalar_t>(siluf_fast(g) * v);
    }
}


template <typename scalar_t>
__global__ void transition_delta_kernel(
    const scalar_t* __restrict__ esa_update,
    const scalar_t* __restrict__ previous_esa,
    const scalar_t* __restrict__ candidate_transition,
    const scalar_t* __restrict__ write_transition,
    const scalar_t* __restrict__ delta_scale,
    scalar_t* __restrict__ candidate_esa,
    scalar_t* __restrict__ write_esa,
    scalar_t* __restrict__ scaled_delta,
    int64_t rows,
    int64_t width) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    const float candidate_alpha = static_cast<float>(candidate_transition[0]);
    const float write_alpha = static_cast<float>(write_transition[0]);
    const float magnitude_scale = static_cast<float>(delta_scale[0]);

    float local_sum = 0.0f;
    const int64_t base = row * width;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const int64_t idx = base + col;
        const float current = static_cast<float>(esa_update[idx]);
        const float previous = static_cast<float>(previous_esa[idx]);
        const float delta = current - previous;
        candidate_esa[idx] = static_cast<scalar_t>(current + candidate_alpha * delta);
        write_esa[idx] = static_cast<scalar_t>(current + write_alpha * delta);
        local_sum += delta * delta;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float mean_sq = scratch[0] / static_cast<float>(width);
        const float magnitude = sqrtf(mean_sq + 1.0e-6f);
        scaled_delta[row] = static_cast<scalar_t>(magnitude_scale * magnitude);
    }
}

template <typename scalar_t>
__global__ void state_mix_kernel(
    const scalar_t* __restrict__ candidate_pre,
    const scalar_t* __restrict__ write_pre,
    const scalar_t* __restrict__ previous_state,
    const scalar_t* __restrict__ scaled_delta,
    const scalar_t* __restrict__ retain_logit,
    const scalar_t* __restrict__ retain_delta_scale,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t width) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int64_t row = idx / width;
        const int64_t col = idx - row * width;
        const float candidate = tanhf(static_cast<float>(candidate_pre[idx]));
        const float write = sigmoidf_fast(static_cast<float>(write_pre[idx]));
        const float retain = sigmoidf_fast(
            static_cast<float>(retain_logit[col])
            - static_cast<float>(scaled_delta[row]) * static_cast<float>(retain_delta_scale[col]));
        const float prev = static_cast<float>(previous_state[idx]);
        out[idx] = static_cast<scalar_t>((1.0f - write) * (retain * prev) + write * candidate);
    }
}

template <typename scalar_t>
__global__ void read_mix_kernel(
    const scalar_t* __restrict__ value_pre,
    const scalar_t* __restrict__ next_state,
    const scalar_t* __restrict__ scaled_delta,
    const scalar_t* __restrict__ read_logit,
    const scalar_t* __restrict__ read_delta_scale,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t width) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int64_t row = idx / width;
        const int64_t col = idx - row * width;
        const float value = siluf_fast(static_cast<float>(value_pre[idx]));
        const float rg = sigmoidf_fast(
            static_cast<float>(read_logit[col])
            + static_cast<float>(scaled_delta[row]) * static_cast<float>(read_delta_scale[col]));
        out[idx] = static_cast<scalar_t>(static_cast<float>(next_state[idx]) * value * rg);
    }
}


template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t width,
    float eps) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    const int64_t base = row * width;
    float local_sum = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float v = static_cast<float>(state[base + col]);
        local_sum += v * v;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(width) + eps);
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const int64_t idx = base + col;
        const float v = static_cast<float>(state[idx]);
        const float w = static_cast<float>(weight[col]);
        out[idx] = static_cast<scalar_t>(v * inv_rms * w);
    }
}

template <typename scalar_t>
__global__ void condition_silu_kernel(
    const scalar_t* __restrict__ base,
    const scalar_t* __restrict__ x_condition,
    const scalar_t* __restrict__ esa_condition,
    const scalar_t* __restrict__ pass_embedding,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t width) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int64_t col = idx % width;
        const float v = static_cast<float>(base[idx])
                      + static_cast<float>(x_condition[idx])
                      + static_cast<float>(esa_condition[idx])
                      + static_cast<float>(pass_embedding[col]);
        out[idx] = static_cast<scalar_t>(siluf_fast(v));
    }
}

template <typename scalar_t>
__global__ void residual_gate_kernel(
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ update,
    const scalar_t* __restrict__ gate_logit,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t width) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const int64_t col = idx % width;
        const float gate = sigmoidf_fast(static_cast<float>(gate_logit[col]));
        out[idx] = static_cast<scalar_t>(static_cast<float>(state[idx]) + gate * static_cast<float>(update[idx]));
    }
}

inline dim3 blocks_for(int64_t n, int threads) {
    return dim3(static_cast<unsigned int>((n + threads - 1) / threads));
}

} // namespace

Tensor silu_mul_cuda(const Tensor& gate, const Tensor& value) {
    TORCH_CHECK(gate.is_cuda() && value.is_cuda(), "silu_mul_cuda expects CUDA tensors");
    TORCH_CHECK(gate.sizes() == value.sizes(), "gate/value shape mismatch");
    TORCH_CHECK(gate.scalar_type() == value.scalar_type(), "gate/value dtype mismatch");
    auto out = at::empty_like(gate);
    const auto n = gate.numel();
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, gate.scalar_type(), "ffnbrick_silu_mul_cuda", [&] {
        silu_mul_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
            gate.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

Tensor silu_mul_packed_cuda(const Tensor& gate_up, int64_t hidden_dim) {
    TORCH_CHECK(gate_up.is_cuda(), "silu_mul_packed_cuda expects a CUDA tensor");
    TORCH_CHECK(gate_up.is_contiguous(), "packed gate/value tensor must be contiguous");
    TORCH_CHECK(hidden_dim > 0, "hidden_dim must be positive");
    TORCH_CHECK(gate_up.size(-1) == 2 * hidden_dim, "packed gate/value width mismatch");

    const int64_t rows = gate_up.numel() / (2 * hidden_dim);
    auto out_sizes = gate_up.sizes().vec();
    out_sizes.back() = hidden_dim;
    auto out = at::empty(out_sizes, gate_up.options());

    const int64_t n = rows * hidden_dim;
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gate_up.scalar_type(),
        "ffnbrick_silu_mul_packed_cuda",
        [&] {
            silu_mul_packed_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
                gate_up.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows,
                hidden_dim);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


std::vector<Tensor> transition_delta_cuda(
    const Tensor& esa_update,
    const Tensor& previous_esa,
    const Tensor& candidate_transition,
    const Tensor& write_transition,
    const Tensor& delta_scale) {
    TORCH_CHECK(esa_update.is_cuda() && previous_esa.is_cuda(),
                "transition_delta_cuda expects CUDA tensors");
    TORCH_CHECK(esa_update.sizes() == previous_esa.sizes(),
                "ESA transition shape mismatch");
    TORCH_CHECK(esa_update.scalar_type() == previous_esa.scalar_type(),
                "ESA transition dtype mismatch");
    TORCH_CHECK(candidate_transition.numel() == 1 && write_transition.numel() == 1
                && delta_scale.numel() == 1,
                "transition scalars must contain exactly one value");
    TORCH_CHECK(candidate_transition.scalar_type() == esa_update.scalar_type()
                && write_transition.scalar_type() == esa_update.scalar_type()
                && delta_scale.scalar_type() == esa_update.scalar_type(),
                "transition scalar dtype mismatch");

    const int64_t width = esa_update.size(-1);
    TORCH_CHECK(width > 0, "ESA width must be positive");
    const int64_t rows = esa_update.numel() / width;

    auto candidate_esa = at::empty_like(esa_update);
    auto write_esa = at::empty_like(esa_update);
    auto delta_sizes = esa_update.sizes().vec();
    delta_sizes.back() = 1;
    auto scaled_delta = at::empty(delta_sizes, esa_update.options());

    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        esa_update.scalar_type(),
        "ffnbrick_transition_delta_cuda",
        [&] {
            transition_delta_kernel<scalar_t><<<
                dim3(static_cast<unsigned int>(rows)), threads, 0, stream>>>(
                esa_update.data_ptr<scalar_t>(),
                previous_esa.data_ptr<scalar_t>(),
                candidate_transition.data_ptr<scalar_t>(),
                write_transition.data_ptr<scalar_t>(),
                delta_scale.data_ptr<scalar_t>(),
                candidate_esa.data_ptr<scalar_t>(),
                write_esa.data_ptr<scalar_t>(),
                scaled_delta.data_ptr<scalar_t>(),
                rows,
                width);
        });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {candidate_esa, write_esa, scaled_delta};
}

Tensor state_mix_cuda(
    const Tensor& candidate_pre,
    const Tensor& write_pre,
    const Tensor& previous_state,
    const Tensor& scaled_delta,
    const Tensor& retain_logit,
    const Tensor& retain_delta_scale) {
    TORCH_CHECK(candidate_pre.is_cuda(), "state_mix_cuda expects CUDA tensors");
    TORCH_CHECK(candidate_pre.sizes() == write_pre.sizes() && candidate_pre.sizes() == previous_state.sizes(), "state tensor shape mismatch");
    TORCH_CHECK(candidate_pre.scalar_type() == write_pre.scalar_type() && candidate_pre.scalar_type() == previous_state.scalar_type(), "state tensor dtype mismatch");
    const int64_t width = candidate_pre.size(-1);
    TORCH_CHECK(scaled_delta.numel() == candidate_pre.numel() / width, "scaled_delta must be [...,1]");
    TORCH_CHECK(retain_logit.numel() == width && retain_delta_scale.numel() == width, "retain parameter width mismatch");
    auto out = at::empty_like(candidate_pre);
    const auto n = candidate_pre.numel();
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, candidate_pre.scalar_type(), "ffnbrick_state_mix_cuda", [&] {
        state_mix_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
            candidate_pre.data_ptr<scalar_t>(), write_pre.data_ptr<scalar_t>(), previous_state.data_ptr<scalar_t>(),
            scaled_delta.data_ptr<scalar_t>(), retain_logit.data_ptr<scalar_t>(), retain_delta_scale.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n, width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

Tensor read_mix_cuda(
    const Tensor& value_pre,
    const Tensor& next_state,
    const Tensor& scaled_delta,
    const Tensor& read_logit,
    const Tensor& read_delta_scale) {
    TORCH_CHECK(value_pre.is_cuda(), "read_mix_cuda expects CUDA tensors");
    TORCH_CHECK(value_pre.sizes() == next_state.sizes(), "read tensor shape mismatch");
    TORCH_CHECK(value_pre.scalar_type() == next_state.scalar_type(), "read tensor dtype mismatch");
    const int64_t width = value_pre.size(-1);
    TORCH_CHECK(scaled_delta.numel() == value_pre.numel() / width, "scaled_delta must be [...,1]");
    TORCH_CHECK(read_logit.numel() == width && read_delta_scale.numel() == width, "read parameter width mismatch");
    auto out = at::empty_like(value_pre);
    const auto n = value_pre.numel();
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, value_pre.scalar_type(), "ffnbrick_read_mix_cuda", [&] {
        read_mix_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
            value_pre.data_ptr<scalar_t>(), next_state.data_ptr<scalar_t>(), scaled_delta.data_ptr<scalar_t>(),
            read_logit.data_ptr<scalar_t>(), read_delta_scale.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


Tensor rms_norm_cuda(
    const Tensor& state,
    const Tensor& weight,
    double eps) {
    TORCH_CHECK(state.is_cuda() && weight.is_cuda(), "rms_norm_cuda expects CUDA tensors");
    TORCH_CHECK(state.scalar_type() == weight.scalar_type(), "RMSNorm dtype mismatch");
    const int64_t width = state.size(-1);
    TORCH_CHECK(width > 0 && weight.numel() == width, "RMSNorm width mismatch");
    const int64_t rows = state.numel() / width;
    auto out = at::empty_like(state);
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        state.scalar_type(),
        "ffnbrick_rms_norm_cuda",
        [&] {
            rms_norm_kernel<scalar_t><<<
                dim3(static_cast<unsigned int>(rows)), threads, 0, stream>>>(
                state.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                rows,
                width,
                static_cast<float>(eps));
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

Tensor condition_silu_cuda(
    const Tensor& base,
    const Tensor& x_condition,
    const Tensor& esa_condition,
    const Tensor& pass_embedding) {
    TORCH_CHECK(base.is_cuda() && x_condition.is_cuda() && esa_condition.is_cuda(),
                "condition_silu_cuda expects CUDA tensors");
    TORCH_CHECK(base.sizes() == x_condition.sizes() && base.sizes() == esa_condition.sizes(),
                "condition tensor shape mismatch");
    TORCH_CHECK(base.scalar_type() == x_condition.scalar_type()
                && base.scalar_type() == esa_condition.scalar_type()
                && base.scalar_type() == pass_embedding.scalar_type(),
                "condition tensor dtype mismatch");
    const int64_t width = base.size(-1);
    TORCH_CHECK(pass_embedding.numel() == width, "pass embedding width mismatch");
    auto out = at::empty_like(base);
    const auto n = base.numel();
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        base.scalar_type(),
        "ffnbrick_condition_silu_cuda",
        [&] {
            condition_silu_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
                base.data_ptr<scalar_t>(),
                x_condition.data_ptr<scalar_t>(),
                esa_condition.data_ptr<scalar_t>(),
                pass_embedding.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n,
                width);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

Tensor residual_gate_cuda(
    const Tensor& state,
    const Tensor& update,
    const Tensor& gate_logit) {
    TORCH_CHECK(state.is_cuda() && update.is_cuda(), "residual_gate_cuda expects CUDA tensors");
    TORCH_CHECK(state.sizes() == update.sizes(), "state/update shape mismatch");
    TORCH_CHECK(state.scalar_type() == update.scalar_type(), "state/update dtype mismatch");
    const int64_t width = state.size(-1);
    TORCH_CHECK(gate_logit.numel() == width, "gate_logit width mismatch");
    auto out = at::empty_like(state);
    const auto n = state.numel();
    constexpr int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, state.scalar_type(), "ffnbrick_residual_gate_cuda", [&] {
        residual_gate_kernel<scalar_t><<<blocks_for(n, threads), threads, 0, stream>>>(
            state.data_ptr<scalar_t>(), update.data_ptr<scalar_t>(), gate_logit.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n, width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

} // namespace ffnbrick
