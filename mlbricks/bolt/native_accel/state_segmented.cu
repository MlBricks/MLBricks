// MLBricks BOLT AVQ v4: segmented, multi-CTA affine recurrence for training.
// Each CTA processes one time chunk across all state dimensions (coalesced).
// Stage 1: independent chunk-local states starting from zero + affine summary.
// Stage 2: short sequential scan across chunk summaries (one thread / dimension).
// Stage 3: correct the stored local states using only A (not recomputing V or local sums).
// Backward performs the same three-stage strategy in reverse time.
// Uncompiled eager PyTorch training. FP32 AVQ. No Triton or torch.compile.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

#ifndef AVQ_CHUNK
#define AVQ_CHUNK 256
#endif
constexpr int THREADS=256;

__global__ void seg_fwd_local(const float* __restrict__ a,
    const float* __restrict__ v, float* __restrict__ states,
    float* __restrict__ ca,float* __restrict__ cb,
    int T,int D,int NC) {
  int d=threadIdx.x, b=blockIdx.y, c=blockIdx.x;
  if(d>=D)return;
  int start=c*AVQ_CHUNK, stop=min(start+AVQ_CHUNK,T);
  float p=1.f, local=0.f;
  for(int t=start;t<stop;++t){
    int64_t k=((int64_t)b*T+t)*D+d;
    float gate=a[k];
    local=gate*local+v[k];
    p=gate*p;
    states[k]=local;
  }
  int64_t ci=((int64_t)b*NC+c)*D+d;
  ca[ci]=p; cb[ci]=local;
}
__global__ void seg_fwd_carries(const float* __restrict__ ca,
    const float* __restrict__ cb,const float* __restrict__ s0,
    float* __restrict__ carries,int D,int NC) {
  int d=blockIdx.x*blockDim.x+threadIdx.x;
  int b=blockIdx.y;
  if(d>=D)return;
  float state=s0[b*D+d];
  for(int c=0;c<NC;++c){
    int64_t i=((int64_t)b*NC+c)*D+d;
    carries[i]=state;
    state=ca[i]*state+cb[i];
  }
}
__global__ void seg_fwd_correct(const float* __restrict__ a,
    const float* __restrict__ carries,float* __restrict__ states,
    int T,int D,int NC) {
  int d=threadIdx.x,b=blockIdx.y,c=blockIdx.x;
  if(d>=D)return;
  float carry=carries[((int64_t)b*NC+c)*D+d], p=1.f;
  int start=c*AVQ_CHUNK,stop=min(start+AVQ_CHUNK,T);
  for(int t=start;t<stop;++t){
    int64_t k=((int64_t)b*T+t)*D+d;
    p=a[k]*p;
    states[k]=states[k]+p*carry;
  }
}
// Backward reverse recurrence: g_t=ds_t+a_(t+1)*g_(t+1).
// Local chunk stores g_t given incoming right carry=0, and p over a_(t+1)...a_(chunk_end).
// Aggregate summary produces outgoing left carry after the last a[t] multiplication.
__global__ void seg_bwd_local(const float* __restrict__ a,
    const float* __restrict__ ds,float* __restrict__ local_g,
    float* __restrict__ ca,float* __restrict__ cb,
    int T,int D,int NC) {
  int d=threadIdx.x,b=blockIdx.y,c=blockIdx.x;
  if(d>=D)return;
  int begin=c*AVQ_CHUNK,end=min(begin+AVQ_CHUNK,T);
  float carry=0.f,p=1.f;
  for(int t=end-1;t>=begin;--t){
    int64_t k=((int64_t)b*T+t)*D+d;
    float gt=ds[k]+carry;
    local_g[k]=gt;
    carry=gt*a[k];
    p=p*a[k];
  }
  int64_t ci=((int64_t)b*NC+c)*D+d;
  ca[ci]=p;cb[ci]=carry;
}
__global__ void seg_bwd_carries(const float* __restrict__ ca,
    const float* __restrict__ cb,float* __restrict__ carries,
    int D,int NC){
  int d=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;
  if(d>=D)return;
  float carry=0.f;
  for(int c=NC-1;c>=0;--c){
    int64_t ci=((int64_t)b*NC+c)*D+d;
    carries[ci]=carry;
    carry=ca[ci]*carry+cb[ci];
  }
}
__global__ void seg_bwd_correct(const float* __restrict__ a,
    const float* __restrict__ states,const float* __restrict__ s0,
    const float* __restrict__ local_g,const float* __restrict__ carries,
    float* __restrict__ ga,float* __restrict__ gv,float* __restrict__ g0,
    int T,int D,int NC) {
  int d=threadIdx.x,b=blockIdx.y,c=blockIdx.x;
  if(d>=D)return;
  int begin=c*AVQ_CHUNK,end=min(begin+AVQ_CHUNK,T);
  float carry=carries[((int64_t)b*NC+c)*D+d],p=1.f;
  for(int t=end-1;t>=begin;--t){
    int64_t k=((int64_t)b*T+t)*D+d;
    // p represents the gate product a[t+1]..a[end-1].
    float gt=local_g[k]+p*carry;
    float prev=t==0?s0[b*D+d]:states[k-D];
    ga[k]=gt*prev;
    gv[k]=gt;
    if(t==0)g0[b*D+d]=gt*a[k];
    p=p*a[k];
  }
}
static void check_inputs(const torch::Tensor& a,const torch::Tensor& v,const torch::Tensor& s0){
  TORCH_CHECK(a.is_cuda() && v.is_cuda() && s0.is_cuda(),"CUDA required");
  TORCH_CHECK(a.scalar_type()==torch::kFloat32 && v.scalar_type()==torch::kFloat32 && s0.scalar_type()==torch::kFloat32,"FP32 tensors required");
  TORCH_CHECK(a.is_contiguous()&&v.is_contiguous()&&s0.is_contiguous(),"contiguous required");
  TORCH_CHECK(a.dim()==3 && v.sizes()==a.sizes(),"a,v must be [B,T,D]");
  TORCH_CHECK(a.size(1)>0 && a.size(2)>0 && a.size(2)<=THREADS,"T positive; state dimension <=256");
  TORCH_CHECK(s0.dim()==2 && s0.size(0)==a.size(0) && s0.size(1)==a.size(2),"s0 must be [B,D]");
  TORCH_CHECK(a.device()==v.device()&&a.device()==s0.device(),"device mismatch");
}
torch::Tensor avq_segmented_forward(torch::Tensor a,torch::Tensor v,torch::Tensor s0){
  check_inputs(a,v,s0);
  c10::cuda::CUDAGuard guard(a.device());
  int B=a.size(0),T=a.size(1),D=a.size(2),NC=(T+AVQ_CHUNK-1)/AVQ_CHUNK;
  auto states=torch::empty_like(a);
  auto summaries=torch::empty({2,B,NC,D},a.options());
  auto ca=summaries.select(0,0),cb=summaries.select(0,1);
  auto carries=torch::empty({B,NC,D},a.options());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  dim3 blocks(NC,B);
  seg_fwd_local<<<blocks,THREADS,0,stream>>>(a.data_ptr<float>(),v.data_ptr<float>(),states.data_ptr<float>(),ca.data_ptr<float>(),cb.data_ptr<float>(),T,D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_fwd_carries<<<dim3((D+THREADS-1)/THREADS,B),THREADS,0,stream>>>(ca.data_ptr<float>(),cb.data_ptr<float>(),s0.data_ptr<float>(),carries.data_ptr<float>(),D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_fwd_correct<<<blocks,THREADS,0,stream>>>(a.data_ptr<float>(),carries.data_ptr<float>(),states.data_ptr<float>(),T,D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return states;
}
std::vector<torch::Tensor> avq_segmented_backward(torch::Tensor a,torch::Tensor states,torch::Tensor s0,torch::Tensor ds){
  check_inputs(a,states,s0);
  TORCH_CHECK(ds.sizes()==a.sizes()&&ds.is_cuda()&&ds.is_contiguous()&&ds.scalar_type()==torch::kFloat32&&ds.device()==a.device(),"ds invalid");
  c10::cuda::CUDAGuard guard(a.device());
  int B=a.size(0),T=a.size(1),D=a.size(2),NC=(T+AVQ_CHUNK-1)/AVQ_CHUNK;
  auto local_g=torch::empty_like(a),ga=torch::empty_like(a),gv=torch::empty_like(a),g0=torch::empty_like(s0);
  auto summaries=torch::empty({2,B,NC,D},a.options());
  auto ca=summaries.select(0,0),cb=summaries.select(0,1);
  auto carries=torch::empty({B,NC,D},a.options());
  auto stream=at::cuda::getCurrentCUDAStream().stream();
  dim3 blocks(NC,B);
  seg_bwd_local<<<blocks,THREADS,0,stream>>>(a.data_ptr<float>(),ds.data_ptr<float>(),local_g.data_ptr<float>(),ca.data_ptr<float>(),cb.data_ptr<float>(),T,D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_bwd_carries<<<dim3((D+THREADS-1)/THREADS,B),THREADS,0,stream>>>(ca.data_ptr<float>(),cb.data_ptr<float>(),carries.data_ptr<float>(),D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_bwd_correct<<<blocks,THREADS,0,stream>>>(a.data_ptr<float>(),states.data_ptr<float>(),s0.data_ptr<float>(),local_g.data_ptr<float>(),carries.data_ptr<float>(),ga.data_ptr<float>(),gv.data_ptr<float>(),g0.data_ptr<float>(),T,D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ga,gv,g0};
}
