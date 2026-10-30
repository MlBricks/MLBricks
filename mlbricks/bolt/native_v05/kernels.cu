#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cmath>

// Inference only: input and output storage must remain valid across graph replay.
template <typename T>
__global__ void gate_kernel(const T* __restrict__ packed,
                            T* __restrict__ out, int w, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t token = i / w;
    const int col = i % w;
    const int64_t base = token * 3 * (int64_t)w;
    const float u = (float)packed[base+w+col];
    const float g = (float)packed[base+2*w+col];
    T tanh_gate = (T)tanhf(g);
    T gate = (T)(1.f + (float)tanh_gate);
    out[i] = (T)(u * (float)gate);
}

void gate_out(torch::Tensor packed, torch::Tensor out) {
    TORCH_CHECK(packed.is_cuda() && out.is_cuda(), "gate requires CUDA tensors");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "gate tensors must be contiguous");
    TORCH_CHECK(packed.dim() == 3 && out.dim() == 3 && packed.size(1) == 1,
                "gate requires packed [B,1,3W] and out [B,1,W]");
    TORCH_CHECK(packed.size(0) == out.size(0) && out.size(1) == 1 &&
                packed.size(2) == 3*out.size(2), "gate shapes differ");
    TORCH_CHECK(packed.scalar_type() == out.scalar_type(), "gate dtype mismatch");
    TORCH_CHECK(packed.device() == out.device(), "gate device mismatch");
    c10::cuda::CUDAGuard guard(packed.device());
    const int64_t n = out.numel();
    const int w = out.size(2);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, packed.scalar_type(), "v05_gate", [&] {
        gate_kernel<scalar_t><<<(n + 255) / 256, 256, 0,
            at::cuda::getCurrentCUDAStream().stream()>>>(
            packed.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), w, n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void avq_kernel(const T* __restrict__ raw,
                           float* __restrict__ state,
                           float* __restrict__ features,
                           int w, float eps) {
    const int batch = blockIdx.x;
    const int i = threadIdx.x;
    extern __shared__ float sq[];
    float s = 0.f, read_gate = 0.f;
    if (i < w) {
        const int base = batch * 3 * w + i;
        const int st = batch * w + i;
        const float a = (float)(T)(1.f / (1.f + expf(-(float)raw[base])));
        const float tv = (float)(T)tanhf((float)raw[base + w]);
        const float v = (float)(T)(.1f * tv);
        read_gate = (float)(T)tanhf((float)raw[base + 2*w]);
        s = state[st] * a + v;
        state[st] = s;
    }
    sq[i] = (i < w ? s*s : 0.f);
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (i < stride) sq[i] += sq[i + stride];
        __syncthreads();
    }
    if (i < w) {
        features[batch*w + i] = s * rsqrtf(sq[0] / (float)w + eps) * (1.f + read_gate);
    }
}

void avq_out(torch::Tensor raw, torch::Tensor state,
             torch::Tensor features, double eps) {
    TORCH_CHECK(raw.is_cuda() && state.is_cuda() && features.is_cuda(), "AVQ requires CUDA tensors");
    TORCH_CHECK(raw.is_contiguous() && state.is_contiguous() && features.is_contiguous(), "AVQ tensors must be contiguous");
    TORCH_CHECK(raw.dim() == 3 && raw.size(1) == 1, "AVQ raw must be [B,1,3W]");
    TORCH_CHECK(state.dim() == 2 && features.dim() == 3 && features.size(1) == 1, "AVQ state/features shape invalid");
    const int w = state.size(1);
    TORCH_CHECK(w > 0 && w <= 1024, "AVQ width must be in [1, 1024]");
    TORCH_CHECK(raw.size(0) == state.size(0) && features.size(0) == state.size(0), "AVQ batch mismatch");
    TORCH_CHECK(raw.size(2) == 3*w && features.size(2) == w, "AVQ width mismatch");
    TORCH_CHECK(state.scalar_type() == at::kFloat && features.scalar_type() == at::kFloat,
                "AVQ state and output features must be FP32");
    TORCH_CHECK(raw.device() == state.device() && raw.device() == features.device(), "AVQ device mismatch");
    TORCH_CHECK(eps > 0., "AVQ epsilon must be positive");
    c10::cuda::CUDAGuard guard(raw.device());
    int threads = 1;
    while (threads < w) threads <<= 1;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, raw.scalar_type(), "v05_avq", [&] {
        avq_kernel<scalar_t><<<state.size(0), threads, threads*sizeof(float),
            at::cuda::getCurrentCUDAStream().stream()>>>(
            raw.data_ptr<scalar_t>(), state.data_ptr<float>(), features.data_ptr<float>(),
            w, (float)eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// BOLT v0.5g experimental recurrence-only CUDA kernel.
// Unlike avq_out, this kernel does NOT calculate RMSNorm, the read gate,
// or any reduction. PyTorch performs those operations in reference order.
template <typename T>
__global__ void avq_state_only_kernel(const T* __restrict__ raw,
                                       float* __restrict__ state,
                                       int w, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t batch = i / w;
    const int col = (int)(i % w);
    const int64_t base = batch * (int64_t)3 * w + col;
    const float a = (float)(T)(1.f / (1.f + expf(-(float)raw[base])));
    const float tv = (float)(T)tanhf((float)raw[base+w]);
    const float v = (float)(T)(.1f * tv);
    state[i] = state[i] * a + v;
}

void avq_state_only(torch::Tensor raw, torch::Tensor state) {
    TORCH_CHECK(raw.is_cuda() && state.is_cuda(), "AVQ recurrence needs CUDA");
    TORCH_CHECK(raw.is_contiguous() && state.is_contiguous(), "AVQ recurrence expects contiguous tensors");
    TORCH_CHECK(raw.dim() == 3 && raw.size(1) == 1, "raw shape must be [B,1,3W]");
    TORCH_CHECK(state.dim() == 2, "state shape must be [B,W]");
    TORCH_CHECK(raw.size(0) == state.size(0) && raw.size(2) == 3*state.size(1), "AVQ dimensions differ");
    TORCH_CHECK(state.scalar_type() == at::kFloat, "state must be FP32");
    TORCH_CHECK(raw.device() == state.device(), "AVQ devices differ");
    TORCH_CHECK(state.size(1) > 0 && state.size(1) <= 1024, "AVQ width must be in [1,1024]");
    c10::cuda::CUDAGuard guard(raw.device());
    const int64_t n = state.numel();
    const int w = (int)state.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, raw.scalar_type(), "v05g_state_only", [&] {
        avq_state_only_kernel<scalar_t><<<(n+255)/256,256,0,
            at::cuda::getCurrentCUDAStream().stream()>>>(
            raw.data_ptr<scalar_t>(), state.data_ptr<float>(), w, n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// BOLT v0.5h EXPERIMENTAL CUDA FP32 readout.
// No AVQ recurrence or RMS reduction here: both are performed identically
// to the already-parity-validated PyTorch pathway (state-only + PyTorch rsqrt).
// The pt variant also receives precomputed PyTorch tanh(raw) cast to FP32.
// Strict left-to-right FP32 arithmetic, explicit round-to-nearest and
// -fmad=false to avoid merging the normalized-state and gate products.
__global__ void readout_pt_kernel(const float* __restrict__ st,
                                   const float* __restrict__ r,
                                   const float* __restrict__ inv,
                                   float* __restrict__ out,
                                   int w, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t b = i / w;
    const float normalized = __fmul_rn(st[i], inv[b]);
    const float gate = __fadd_rn(1.0f, r[i]);
    out[i] = __fmul_rn(normalized, gate);
}

template<typename T>
__global__ void readout_raw_kernel(const float* __restrict__ st,
                                    const T* __restrict__ raw,
                                    const float* __restrict__ inv,
                                    float* __restrict__ out,
                                    int w, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t b = i / w;
    // Preserve the FP16/BF16 tanh output rounding boundary before float readout.
    const float r = (float)(T)tanhf((float)raw[b*(int64_t)3*w + 2*w + (i % w)]);
    const float normalized = __fmul_rn(st[i], inv[b]);
    const float gate = __fadd_rn(1.0f, r);
    out[i] = __fmul_rn(normalized, gate);
}

static void validate_readout(torch::Tensor st, torch::Tensor inv,
                             torch::Tensor out) {
    TORCH_CHECK(st.is_cuda() && inv.is_cuda() && out.is_cuda(), "readout CUDA required");
    TORCH_CHECK(st.is_contiguous() && inv.is_contiguous() && out.is_contiguous(), "readout tensors must be contiguous");
    TORCH_CHECK(st.dim() == 2 && inv.dim() == 3 && out.dim() == 3, "readout rank mismatch");
    TORCH_CHECK(st.size(1) > 0 && st.size(1) <= 1024, "readout width out of range");
    TORCH_CHECK(out.size(0) == st.size(0) && out.size(1) == 1 && out.size(2) == st.size(1), "readout features shape invalid");
    TORCH_CHECK(inv.size(0) == st.size(0) && inv.size(1) == 1 && inv.size(2) == 1, "readout inverse RMS shape invalid");
    TORCH_CHECK(st.scalar_type() == at::kFloat && inv.scalar_type() == at::kFloat && out.scalar_type() == at::kFloat, "readout expects float32 state/normalizer/output");
    TORCH_CHECK(st.device() == inv.device() && st.device() == out.device(), "readout device mismatch");
}

void avq_readout_pt(torch::Tensor st, torch::Tensor r,
                    torch::Tensor inv, torch::Tensor out) {
    validate_readout(st, inv, out);
    TORCH_CHECK(r.is_cuda() && r.device() == st.device() && r.is_contiguous() &&
                r.scalar_type() == at::kFloat && r.sizes() == out.sizes(),
                "readout r must be contiguous FP32 [B,1,W]");
    c10::cuda::CUDAGuard guard(st.device());
    const int64_t n = st.numel();
    readout_pt_kernel<<<(n+255)/256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        st.data_ptr<float>(), r.data_ptr<float>(), inv.data_ptr<float>(), out.data_ptr<float>(), (int)st.size(1), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void avq_readout_raw(torch::Tensor st, torch::Tensor raw,
                     torch::Tensor inv, torch::Tensor out) {
    validate_readout(st, inv, out);
    TORCH_CHECK(raw.is_cuda() && raw.device() == st.device() && raw.is_contiguous() &&
                raw.dim() == 3 && raw.size(0) == st.size(0) && raw.size(1) == 1 &&
                raw.size(2) == 3*st.size(1), "readout raw must be contiguous [B,1,3W]");
    c10::cuda::CUDAGuard guard(st.device());
    const int64_t n = st.numel();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, raw.scalar_type(), "v05h_readout_raw", [&] {
        readout_raw_kernel<scalar_t><<<(n+255)/256, 256, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
            st.data_ptr<float>(), raw.data_ptr<scalar_t>(), inv.data_ptr<float>(),
            out.data_ptr<float>(), (int)st.size(1), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
