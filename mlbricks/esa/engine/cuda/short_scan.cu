// MLBricks FAST SWQR-Short: one-launch CTA-local FP32 affine prefix scan.
// E[t] = A[t] * E[t-1] + U[t], optional initial state, shape [B,T,H,Dh].
// One warp owns 32 contiguous channels; W warps span disjoint time segments.
// Two passes through A,U: local segment summaries, then output recurrence.
// No global intermediate arrays, one CUDA kernel launch. T <= 2048.
#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {
template<int WARPS>
__global__ void short_scan_kernel(const float* __restrict__ A,
                                  const float* __restrict__ U,
                                  const float* __restrict__ state,
                                  float* __restrict__ output,
                                  int T, int D, bool has_state) {
    constexpr int LANES = 32;
    __shared__ float prod[WARPS][LANES];
    __shared__ float summ[WARPS][LANES];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = static_cast<int>(blockIdx.x) * LANES + lane;
    const int b = static_cast<int>(blockIdx.y);
    const bool valid = col < D;
    const int seg = (T + WARPS - 1) / WARPS;
    const int start = warp * seg;
    const int stop = (start + seg < T) ? start + seg : T;

    float p = 1.0f, s = 0.0f;
    if (valid) {
        for (int t = start; t < stop; ++t) {
            const int64_t idx = (static_cast<int64_t>(b) * T + t) * D + col;
            const float a = A[idx];
            s = a * s + U[idx];
            p *= a;
        }
    }
    prod[warp][lane] = p;
    summ[warp][lane] = s;
    __syncthreads();

    if (valid) {
        float carry = has_state ? state[b * D + col] : 0.0f;
        // Affine summaries are ordered: (p2,s2) o (p1,s1)
        //       = (p2*p1, p2*s1+s2).
        for (int w = 0; w < warp; ++w) {
            carry = prod[w][lane] * carry + summ[w][lane];
        }
        for (int t = start; t < stop; ++t) {
            const int64_t idx = (static_cast<int64_t>(b) * T + t) * D + col;
            carry = A[idx] * carry + U[idx];
            output[idx] = carry;
        }
    }
}

template<int WARPS>
void run(const float* A, const float* U, const float* state, float* output,
         int B, int T, int D, bool has_state, cudaStream_t stream) {
    dim3 grid(static_cast<unsigned>((D + 31) / 32), static_cast<unsigned>(B));
    short_scan_kernel<WARPS><<<grid, WARPS * 32, 0, stream>>>(
        A, U, state, output, T, D, has_state);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

torch::Tensor short_scan_cuda(torch::Tensor A, torch::Tensor U,
                                  torch::Tensor state, int64_t warps) {
    c10::cuda::CUDAGuard guard(A.device());
    const auto B64 = A.size(0), T64 = A.size(1), D64 = A.size(2) * A.size(3);
    TORCH_CHECK(T64 >= 1 && T64 <= 2048, "SWQR-Short supports 1..2048 tokens");
    TORCH_CHECK(B64 <= 65535 && D64 <= INT32_MAX, "SWQR-Short unsupported shape");
    const int B = static_cast<int>(B64), T = static_cast<int>(T64);
    const int D = static_cast<int>(D64);
    auto output = at::empty_like(A);
    if (B == 0 || D == 0) return output;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool has_state = state.numel() != 0;
    auto init = has_state ? state.data_ptr<float>() : nullptr;
    auto pa = A.data_ptr<float>(), pu = U.data_ptr<float>(), po = output.data_ptr<float>();
    if      (warps == 4)  run<4>(pa, pu, init, po, B, T, D, has_state, stream);
    else if (warps == 8)  run<8>(pa, pu, init, po, B, T, D, has_state, stream);
    else if (warps == 16) run<16>(pa, pu, init, po, B, T, D, has_state, stream);
    else TORCH_CHECK(false, "warps must be 4, 8 or 16");
    return output;
}
