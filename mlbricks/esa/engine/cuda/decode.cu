// Fast Half-Life Quality ESA: fused FP32 one-token recurrent update, learned state RMSNorm, and query readout.
// CUDA/C++ only, no Triton. GEMMs remain cuBLAS via PyTorch linear layers.
#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <climits>
namespace {
__device__ __forceinline__ float sigma(float z) {
  return z >= 0.f ? 1.f/(1.f+expf(-z)) : expf(z)/(1.f+expf(z));
}
__device__ __forceinline__ float softplus_stable(float z) {
  return z>20.f ? z : log1pf(expf(z));
}
template <typename IN,typename OUT,int THREADS>
__global__ void esa_decode(
  const IN* __restrict__ packed,
  const float* __restrict__ log_half,
  const float* __restrict__ decay_bias,
  const float* __restrict__ norm_weight,
  const float* __restrict__ old_state,
  float* __restrict__ new_state,
  OUT* __restrict__ readout,
  int D,int Dh,bool has_state) {
  int b=blockIdx.x,tid=threadIdx.x;
  __shared__ float sum_sq[THREADS];
  float mine=0.f;
  int64_t base=(int64_t)b*3*D;
  int64_t sb=(int64_t)b*D;
  for(int d=tid;d<D;d+=THREADS) {
    int h=d/Dh;
    float g=(float)packed[base+D+d];
    float v=(float)packed[base+2*D+d];
    // Exactly the PROGRESS hybrid half-life decay formulation.
    float half=expf(log_half[h]);
    float decay=softplus_stable(g+decay_bias[h]);
    float a=expf((-0.69314718055994530942f)*decay/half);
    float u=(1.0f-a)*tanhf(v);
    float previous=has_state ? old_state[sb+d] : 0.f;
    float e=a*previous+u;
    new_state[sb+d]=e;
    mine+=e*e;
  }
  sum_sq[tid]=mine;
  __syncthreads();
  for(int stride=THREADS/2;stride>0;stride>>=1) {
    if(tid<stride) sum_sq[tid]+=sum_sq[tid+stride];
    __syncthreads();
  }
  float inv=rsqrtf(sum_sq[0]/(float)D+1e-5f);
  for(int d=tid;d<D;d+=THREADS) {
    float q=(float)packed[base+d];
    float e=new_state[sb+d];
    float value=sigma(q)*(e*inv*norm_weight[d]);
    readout[sb+d]=(OUT)value;
  }
}
template<typename IN,typename OUT>
void launch(const IN* qgv,const float* hl,const float* bias,const float* norm,
            const float* old,float* new_state,OUT* output,
            int B,int D,int Dh,bool has_state,cudaStream_t stream) {
  constexpr int THREADS=256;
  esa_decode<IN,OUT,THREADS><<<B,THREADS,0,stream>>>(
      qgv,hl,bias,norm,old,new_state,output,D,Dh,has_state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace
std::vector<torch::Tensor> fused_decode_cuda(
    torch::Tensor qgv,torch::Tensor x,torch::Tensor log_half,
    torch::Tensor decay_bias,torch::Tensor norm_weight,
    torch::Tensor old_state,int64_t heads) {
  c10::cuda::CUDAGuard guard(qgv.device());
  int B=(int)x.size(0), D=(int)x.size(2), Dh=D/(int)heads;
  auto readout=at::empty({B,1,D},x.options());
  auto state=at::empty({B,heads,Dh},log_half.options());
  if(!B)return {readout,state};
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  bool has_state=old_state.numel()>0;
  const float* old=has_state?old_state.data_ptr<float>():nullptr;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,qgv.scalar_type(),
      "quality_qgv_dispatch",[&] {
    using in_t=scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,x.scalar_type(),
        "quality_output_dispatch",[&] {
      using out_t=scalar_t;
      launch<in_t,out_t>(qgv.data_ptr<in_t>(),log_half.data_ptr<float>(),
          decay_bias.data_ptr<float>(),norm_weight.data_ptr<float>(),old,
          state.data_ptr<float>(),readout.data_ptr<out_t>(),B,D,Dh,has_state,stream);
    });
  });
  return {readout,state};
}
