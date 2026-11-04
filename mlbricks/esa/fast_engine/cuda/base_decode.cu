// BaseESA learned-bounds fused one-token CUDA decode.
// Original BaseESA recurrence: alpha=.9999*sigmoid(raw_alpha),
// beta=alpha+(.9999-alpha)*sigmoid(raw_beta), A=alpha+(beta-alpha)*sigmoid(g),
// E=A*E_prev+(1-A)*tanh(v); read=sigmoid(q)*RMSNorm(E).
// Adapted from Fast-EsaQ's Half-Life decode.cu; no Triton and no approximation.
#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace {
__device__ __forceinline__ float sigma(float z) {
  return z >= 0.f ? 1.f / (1.f + expf(-z)) : expf(z) / (1.f + expf(z));
}

template <typename IN, typename OUT, int THREADS>
__global__ void base_esa_decode_kernel(
    const IN* __restrict__ packed,
    const float* __restrict__ raw_alpha,
    const float* __restrict__ raw_beta,
    const float* __restrict__ previous_state,
    float* __restrict__ next_state,
    OUT* __restrict__ readout,
    int D, int Dh, bool has_previous_state) {
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float sums[THREADS];
  float partial = 0.f;
  const int64_t packed_offset = static_cast<int64_t>(b) * 3 * D;
  const int64_t state_offset = static_cast<int64_t>(b) * D;

  for (int d = tid; d < D; d += THREADS) {
    const int h = d / Dh;
    const float alpha = 0.9999f * sigma(raw_alpha[h]);
    const float beta = alpha + (0.9999f - alpha) * sigma(raw_beta[h]);
    const float g = static_cast<float>(packed[packed_offset + D + d]);
    const float v = static_cast<float>(packed[packed_offset + 2*D + d]);
    const float a = alpha + (beta - alpha) * sigma(g);
    const float old = has_previous_state ? previous_state[state_offset + d] : 0.f;
    const float e = a * old + (1.f - a) * tanhf(v);
    next_state[state_offset + d] = e;
    partial += e * e;
  }

  sums[tid] = partial;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sums[0] / static_cast<float>(D) + 1.e-5f);
  for (int d = tid; d < D; d += THREADS) {
    const float q = static_cast<float>(packed[packed_offset + d]);
    const float e = next_state[state_offset + d];
    readout[state_offset + d] = static_cast<OUT>(sigma(q) * e * inv_rms);
  }
}

template <typename IN, typename OUT>
void launch(const IN* packed, const float* a, const float* b,
            const float* old, float* next, OUT* y,
            int B, int D, int Dh, bool has_old, cudaStream_t stream) {
  constexpr int THREADS = 256;
  base_esa_decode_kernel<IN, OUT, THREADS><<<B, THREADS, 0, stream>>>(
      packed, a, b, old, next, y, D, Dh, has_old);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

std::vector<torch::Tensor> fused_base_decode_cuda(
    torch::Tensor qgv, torch::Tensor x,
    torch::Tensor raw_alpha, torch::Tensor raw_beta,
    torch::Tensor old_state, int64_t heads) {
  c10::cuda::CUDAGuard guard(qgv.device());
  const int B = static_cast<int>(x.size(0));
  const int D = static_cast<int>(x.size(2));
  const int Dh = D / static_cast<int>(heads);
  auto out = at::empty({B,1,D}, x.options());
  auto state = at::empty({B,heads,Dh}, raw_alpha.options());
  if (B == 0) return {out, state};
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bool has_old = old_state.numel() != 0;
  const float* old = has_old ? old_state.data_ptr<float>() : nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
      qgv.scalar_type(), "base_qgv_dispatch", [&] {
    using in_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
        x.scalar_type(), "base_out_dispatch", [&] {
      using out_t = scalar_t;
      launch<in_t,out_t>(qgv.data_ptr<in_t>(), raw_alpha.data_ptr<float>(),
          raw_beta.data_ptr<float>(), old, state.data_ptr<float>(),
          out.data_ptr<out_t>(), B, D, Dh, has_old, stream);
    });
  });
  return {out, state};
}
