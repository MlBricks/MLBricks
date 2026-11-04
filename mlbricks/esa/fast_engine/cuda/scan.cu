// MLBricks experimental inference-only FP32 SWQR scan.
// Associative recurrence: E_t = A_t*E_(t-1) + U_t.
// 1 kernel for T<=128; 3-stage hierarchical scan otherwise.
#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <algorithm>

// One thread handles a single (batch,channel) recurrence. Neighboring threads
// access neighboring channel elements, enabling coalesced memory operations.
__global__ void scan_serial_kernel(const float* __restrict__ a,
                                   const float* __restrict__ u,
                                   const float* __restrict__ initial,
                                   float* __restrict__ e,
                                   int64_t T,int64_t D,int64_t channels,
                                   bool has_initial) {
    int64_t ch=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(ch>=channels)return;
    int64_t b=ch/D, col=ch%D;
    float carry=has_initial?initial[ch]:0.0f;
    for(int64_t t=0;t<T;t++){
        int64_t idx=(b*T+t)*D+col;
        carry=a[idx]*carry+u[idx];
        e[idx]=carry;
    }
}

__global__ void chunk_summaries_kernel(const float* __restrict__ a,
                                      const float* __restrict__ u,
                                      float* __restrict__ prod,
                                      float* __restrict__ sum,
                                      int64_t T,int64_t D,int64_t G,int64_t C,
                                      int64_t tasks) {
    int64_t task=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=tasks)return;
    int64_t col=task%D;
    int64_t g=(task/D)%G;
    int64_t b=task/(D*G);
    int64_t t0=g*C,t1=(t0+C<T)?t0+C:T;
    float p=1.0f,s=0.0f;
    for(int64_t t=t0;t<t1;t++){
        const int64_t idx=(b*T+t)*D+col;
        float aa=a[idx];
        p=aa*p;
        s=aa*s+u[idx];
    }
    prod[task]=p;
    sum[task]=s;
}

__global__ void chunk_initials_kernel(const float* __restrict__ prod,
                                     const float* __restrict__ sum,
                                     const float* __restrict__ initial,
                                     float* __restrict__ carries,
                                     int64_t D,int64_t G,int64_t channels,
                                     bool has_initial) {
    int64_t ch=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(ch>=channels)return;
    int64_t col=ch%D,b=ch/D;
    float carry=has_initial?initial[ch]:0.0f;
    for(int64_t g=0;g<G;g++){
        int64_t i=(b*G+g)*D+col;
        carries[i]=carry;
        carry=prod[i]*carry+sum[i];
    }
}

__global__ void chunk_outputs_kernel(const float* __restrict__ a,
                                     const float* __restrict__ u,
                                     const float* __restrict__ carries,
                                     float* __restrict__ e,
                                     int64_t T,int64_t D,int64_t G,int64_t C,
                                     int64_t tasks) {
    int64_t task=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(task>=tasks)return;
    int64_t col=task%D,g=(task/D)%G,b=task/(D*G);
    int64_t t0=g*C,t1=(t0+C<T)?t0+C:T;
    float state=carries[task];
    for(int64_t t=t0;t<t1;t++){
        int64_t idx=(b*T+t)*D+col;
        state=a[idx]*state+u[idx];
        e[idx]=state;
    }
}

// Public binding shape [B,T,H,Dh]. The flattened last two dims form D.
torch::Tensor parallel_scan_cuda(torch::Tensor a,torch::Tensor u,
                                 torch::Tensor state,int64_t chunk) {
    c10::cuda::CUDAGuard guard(a.device());
    const int64_t B=a.size(0), T=a.size(1);
    const int64_t D=a.size(2)*a.size(3);
    const int64_t G=(T+chunk-1)/chunk;
    auto output=at::empty_like(a);
    if(B==0 || D==0 || T==0)return output;
    cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    constexpr int nt=256;
    bool has_initial=state.numel()!=0;
    const float* init=has_initial?state.data_ptr<float>():nullptr;
    int64_t channels=B*D;
    if(T<=128){
        scan_serial_kernel<<<(unsigned)((channels+nt-1)/nt),nt,0,stream>>>(
            a.data_ptr<float>(),u.data_ptr<float>(),init,output.data_ptr<float>(),
            T,D,channels,has_initial);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    auto buffer=at::empty({B,G,D},a.options());
    auto sum=at::empty_like(buffer);
    auto carry=at::empty_like(buffer);
    int64_t tasks=B*G*D;
    TORCH_CHECK(tasks<=(int64_t)INT32_MAX*nt, "CUDA launch grid too large");
    chunk_summaries_kernel<<<(unsigned)((tasks+nt-1)/nt),nt,0,stream>>>(
        a.data_ptr<float>(),u.data_ptr<float>(),buffer.data_ptr<float>(),
        sum.data_ptr<float>(),T,D,G,chunk,tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    chunk_initials_kernel<<<(unsigned)((channels+nt-1)/nt),nt,0,stream>>>(
        buffer.data_ptr<float>(),sum.data_ptr<float>(),init,carry.data_ptr<float>(),
        D,G,channels,has_initial);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    chunk_outputs_kernel<<<(unsigned)((tasks+nt-1)/nt),nt,0,stream>>>(
        a.data_ptr<float>(),u.data_ptr<float>(),carry.data_ptr<float>(),
        output.data_ptr<float>(),T,D,G,chunk,tasks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
