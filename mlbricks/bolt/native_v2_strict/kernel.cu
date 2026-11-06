// BOLT AVQ V2 strict, opt-in CUDA decoding path. No Triton.
// Based on the existing verified AVQ FP16 gate-rounding + FP32 state recurrence.
// Inputs: o[1,1,W] half, state[1,W] float, norm[W] float, gamma[1] float.
// V2: PyTorch Linear produces z[1,1,3W] then AVQ runs in-place.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cmath>

__device__ __forceinline__ float half_gate(float x) {
    return __half2float(__float2half_rn(x));
}

__global__ void update_inplace_kernel(
    __half* __restrict__ o,
    const __half* __restrict__ z,
    float* __restrict__ state,
    const float* __restrict__ nw,
    const float* __restrict__ gamma,
    int width, float eps) {
    const int i=threadIdx.x, lane=i&31, warp=i>>5;
    __shared__ float warp_sums[16];
    __shared__ float inv_norm;
    float new_value=0.f, r=0.f, square=0.f;
    if(i<width) {
        float ga=__half2float(z[i]);
        float gv=__half2float(z[width+i]);
        float gr=__half2float(z[2*width+i]);
        float a=half_gate(1.f/(1.f+expf(-ga)));
        float v=half_gate(0.1f*half_gate(tanhf(gv)));
        r=half_gate(tanhf(gr));
        new_value=__fadd_rn(__fmul_rn(a,state[i]),v);
        state[i]=new_value;       // same address as old state: safe per dimension
        square=new_value*new_value;
    }
    for(int off=16;off>0;off>>=1) square+=__shfl_down_sync(0xffffffff,square,off);
    if(lane==0) warp_sums[warp]=square;
    __syncthreads();
    if(i==0) {
        float sum=0.f;
        for(int w=0;w<(blockDim.x+31)/32;w++)sum+=warp_sums[w];
        inv_norm=rsqrtf(sum/(float)width+eps);
    }
    __syncthreads();
    if(i<width){
        float memory=new_value*inv_norm*nw[i]*(1.f+r);
        // Keep the old kernel's half-rounding semantics (including the residual).
        float mh=half_gate(memory), gh=half_gate(gamma[0]);
        float prod=half_gate(mh*gh);
        float old=__half2float(o[i]);
        o[i]=__float2half_rn(old+prod);
    }
}

void v2_launch(const at::Tensor& o,const at::Tensor& z,at::Tensor& state,
               const at::Tensor& nw,const at::Tensor& gamma,double eps){
    int width=state.size(1),threads=width<=256?256:512;
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    update_inplace_kernel<<<1,threads,0,stream>>>(
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(z.data_ptr<at::Half>()),
        state.data_ptr<float>(),nw.data_ptr<float>(),gamma.data_ptr<float>(),width,(float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
