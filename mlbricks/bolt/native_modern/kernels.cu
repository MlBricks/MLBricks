// BOLT-32 / AVQ-198 CUDA kernels. Other 16/192/198 widths are also supported.
// FP16/FP32 latent kernels, FP32 recurrent state and score accumulation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cmath>

namespace {
void check_cuda_contig(torch::Tensor t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
void check_f16_f32(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.scalar_type()==torch::kFloat16 || t.scalar_type()==torch::kFloat32,
                name, " requires float16 or float32");
}
template <typename scalar_t> __global__ void features_forward_kernel(
  const scalar_t *__restrict__ q, const scalar_t *__restrict__ u,
  const scalar_t *__restrict__ g, const float *__restrict__ p,
  float *__restrict__ q_out, scalar_t *__restrict__ c_out,
  int64_t n, int64_t heads, int64_t tokens, int64_t width) {
  const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx>=n)return;
  const int64_t pr= (idx/(tokens*width) % heads)*width + idx%width;
  const float uval=static_cast<float>(u[idx]);
  const float gval=static_cast<float>(g[idx]);
  const float qval=static_cast<float>(q[idx]);
  // Match PyTorch's intermediate casts for FP16 tanh -> add -> multiply.
  scalar_t gate=static_cast<scalar_t>(tanhf(gval));
  scalar_t gain=static_cast<scalar_t>(1.f+static_cast<float>(gate));
  c_out[idx]=static_cast<scalar_t>(uval*static_cast<float>(gain));
  q_out[idx]=qval*p[pr];
}
template <typename scalar_t> __global__ void features_backward_kernel(
  const float *__restrict__ gq, const scalar_t *__restrict__ gc,
  const scalar_t *__restrict__ q, const scalar_t *__restrict__ u,
  const scalar_t *__restrict__ g, const float *__restrict__ p,
  scalar_t *__restrict__ dq, scalar_t *__restrict__ du,
  scalar_t *__restrict__ dg, float *__restrict__ dp_parts,
  int64_t n, int64_t heads, int64_t tokens, int64_t width) {
  const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx>=n)return;
  const int64_t pr= (idx/(tokens*width) % heads)*width + idx%width;
  const float gv=static_cast<float>(g[idx]);
  const float gate=static_cast<float>(static_cast<scalar_t>(tanhf(gv)));
  const float factor=static_cast<float>(static_cast<scalar_t>(1.f+gate));
  const float cg=static_cast<float>(gc[idx]);
  dq[idx]=static_cast<scalar_t>(gq[idx]*p[pr]);
  du[idx]=static_cast<scalar_t>(cg*factor);
  dg[idx]=static_cast<scalar_t>(cg*static_cast<float>(u[idx])*(1.f-gate*gate));
  dp_parts[idx]=gq[idx]*static_cast<float>(q[idx]);
}
__global__ void avq_scan_kernel(const float *a,const float *v,float *s,
                                int64_t batch,int64_t tokens,int64_t width) {
    const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=batch*width)return;
    const int64_t b=index/width, w=index%width;
    const int64_t start=b*tokens*width+w;
    float state=0.f;
    for(int64_t t=0;t<tokens;++t) {
        const int64_t offset=start+t*width;
        state = a[offset]*state+v[offset];
        s[offset]=state;
    }
}
__global__ void avq_scan_backward_kernel(const float *ds,const float *a,const float *s,
                                         float *da,float *dv,int64_t batch,int64_t tokens,int64_t width) {
    const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=batch*width)return;
    const int64_t b=index/width, w=index%width;
    const int64_t start=b*tokens*width+w;
    float carry=0.f;
    for(int64_t t=tokens;t-->0;) {
        const int64_t offset=start+t*width;
        const float grad=ds[offset]+carry;
        da[offset]=grad*(t==0?0.f:s[offset-width]);
        dv[offset]=grad;
        carry=grad*a[offset];
    }
}
__global__ void avq_step_kernel(const float *a,const float *v,const float *last,float *out,int64_t n) {
    const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx<n)out[idx]=a[idx]*last[idx]+v[idx];
}
// Each CTA processes one (batch,head). Scores are buffered in scratch, per head.
// Streaming softmax with a stable max reduction. Exactly 1 append per call.
// Only supports no-RoPE and no dropout; other paths use SDPA.
template <typename scalar_t>
__global__ void bolt_decode_kernel(const float *__restrict__ q,
    const scalar_t *__restrict__ current,scalar_t *__restrict__ history,
    float *__restrict__ scratch,float *__restrict__ output,
    int64_t heads,int64_t capacity,int64_t width,int64_t position,float scale) {
    const int batchhead=blockIdx.x;
    const int tid=threadIdx.x;
    const int64_t base=int64_t(batchhead)*capacity*width;
    const int64_t now=int64_t(batchhead)*width;
    __shared__ float shared[128];
    for(int d=tid;d<width;d+=128)history[base+position*width+d]=current[now+d];
    __syncthreads();
    float local_max=-INFINITY;
    for(int64_t t=tid;t<=position;t+=128) {
        float dot=0.f;
        for(int d=0;d<width;++d)
            dot+=q[now+d]*static_cast<float>(history[base+t*width+d]);
        const float score=dot*scale;
        scratch[int64_t(batchhead)*capacity+t]=score;
        local_max=fmaxf(local_max,score);
    }
    shared[tid]=local_max;
    __syncthreads();
    for(int stride=64;stride>0;stride>>=1) {
        if(tid<stride)shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    const float max_score=shared[0];
    float partial=0.f;
    for(int64_t t=tid;t<=position;t+=128) {
        const float e=expf(scratch[int64_t(batchhead)*capacity+t]-max_score);
        scratch[int64_t(batchhead)*capacity+t]=e;
        partial+=e;
    }
    shared[tid]=partial;
    __syncthreads();
    for(int stride=64;stride>0;stride>>=1) {
        if(tid<stride)shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    const float inv_denom=1.f/shared[0];
    for(int d=tid;d<width;d+=128) {
        float weighted=0.f;
        for(int64_t t=0;t<=position;++t)
            weighted+=scratch[int64_t(batchhead)*capacity+t]*static_cast<float>(history[base+t*width+d]);
        output[now+d]=weighted*inv_denom;
    }
}
}

std::vector<torch::Tensor> bolt_features_forward_cuda(torch::Tensor q,torch::Tensor u,torch::Tensor g,torch::Tensor p) {
    c10::cuda::CUDAGuard device_guard(q.device());
    check_cuda_contig(q,"q");check_cuda_contig(u,"u");check_cuda_contig(g,"g");check_cuda_contig(p,"p");
    check_f16_f32(q,"q");TORCH_CHECK(q.scalar_type()==u.scalar_type() && q.scalar_type()==g.scalar_type(),"feature dtypes must match");
    TORCH_CHECK(p.scalar_type()==torch::kFloat32,"p must be float32");
    TORCH_CHECK(q.dim()==4 && q.sizes()==u.sizes() && q.sizes()==g.sizes(),"features require matching [B,H,T,R]");
    TORCH_CHECK(p.dim()==2 && p.size(0)==q.size(1) && p.size(1)==q.size(3),"p requires [H,R]");
    auto query=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    auto c=torch::empty_like(u);
    const int64_t n=q.numel();const int threads=256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),"bolt_features_forward",[&]{
      features_forward_kernel<scalar_t><<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        q.data_ptr<scalar_t>(),u.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),p.data_ptr<float>(),
        query.data_ptr<float>(),c.data_ptr<scalar_t>(),n,q.size(1),q.size(2),q.size(3));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {query,c};
}
std::vector<torch::Tensor> bolt_features_backward_cuda(torch::Tensor gq,torch::Tensor gc,torch::Tensor q,
                                                        torch::Tensor u,torch::Tensor g,torch::Tensor p) {
    c10::cuda::CUDAGuard device_guard(gq.device());
    check_cuda_contig(gq,"grad q");check_cuda_contig(gc,"grad c");
    check_cuda_contig(q,"q");check_cuda_contig(u,"u");check_cuda_contig(g,"g");check_cuda_contig(p,"p");
    TORCH_CHECK(gq.scalar_type()==torch::kFloat32,"grad q requires float32");
    TORCH_CHECK(gc.scalar_type()==q.scalar_type(),"grad c must match input dtype");
    TORCH_CHECK(gq.sizes()==q.sizes() && gc.sizes()==q.sizes(),"gradient shape mismatch");
    auto dq=torch::empty_like(q),du=torch::empty_like(u),dg=torch::empty_like(g);
    auto dp_parts=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    int64_t n=q.numel();int threads=256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),"bolt_features_backward",[&]{
      features_backward_kernel<scalar_t><<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        gq.data_ptr<float>(),gc.data_ptr<scalar_t>(),q.data_ptr<scalar_t>(),u.data_ptr<scalar_t>(),
        g.data_ptr<scalar_t>(),p.data_ptr<float>(),dq.data_ptr<scalar_t>(),du.data_ptr<scalar_t>(),
        dg.data_ptr<scalar_t>(),dp_parts.data_ptr<float>(),n,q.size(1),q.size(2),q.size(3));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq,du,dg,dp_parts.sum({0,2})};
}

torch::Tensor avq_scan_forward_cuda(torch::Tensor a,torch::Tensor v) {
    c10::cuda::CUDAGuard device_guard(a.device());
    check_cuda_contig(a,"a");check_cuda_contig(v,"v");
    TORCH_CHECK(a.scalar_type()==torch::kFloat32 && v.scalar_type()==torch::kFloat32,"AVQ requires FP32 components");
    TORCH_CHECK(a.sizes()==v.sizes() && a.dim()==3,"AVQ expects matching [B,T,W]");
    auto s=torch::empty_like(a);int64_t n=a.size(0)*a.size(2);int threads=128;
    avq_scan_kernel<<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        a.data_ptr<float>(),v.data_ptr<float>(),s.data_ptr<float>(),a.size(0),a.size(1),a.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return s;
}
std::vector<torch::Tensor> avq_scan_backward_cuda(torch::Tensor ds,torch::Tensor a,torch::Tensor s) {
    c10::cuda::CUDAGuard device_guard(ds.device());
    check_cuda_contig(ds,"gradient");check_cuda_contig(a,"a");check_cuda_contig(s,"state");
    TORCH_CHECK(a.sizes()==s.sizes() && a.sizes()==ds.sizes() && a.dim()==3,"AVQ backward shape mismatch");
    TORCH_CHECK(a.scalar_type()==torch::kFloat32 && ds.scalar_type()==torch::kFloat32,"AVQ backward requires FP32");
    auto da=torch::empty_like(a),dv=torch::empty_like(a);
    int64_t n=a.size(0)*a.size(2);int threads=128;
    avq_scan_backward_kernel<<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        ds.data_ptr<float>(),a.data_ptr<float>(),s.data_ptr<float>(),da.data_ptr<float>(),dv.data_ptr<float>(),
        a.size(0),a.size(1),a.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {da,dv};
}
torch::Tensor avq_decode_state_cuda(torch::Tensor a,torch::Tensor v,torch::Tensor previous) {
    c10::cuda::CUDAGuard device_guard(a.device());
    check_cuda_contig(a,"a");check_cuda_contig(v,"v");check_cuda_contig(previous,"previous");
    TORCH_CHECK(a.scalar_type()==torch::kFloat32 && a.sizes()==v.sizes() && a.sizes()==previous.sizes(),"AVQ step FP32 tensors must match");
    auto out=torch::empty_like(previous);int64_t n=out.numel();int threads=256;
    avq_step_kernel<<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        a.data_ptr<float>(),v.data_ptr<float>(),previous.data_ptr<float>(),out.data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor bolt_decode_append_cuda(torch::Tensor q,torch::Tensor current,torch::Tensor cache,
                                       torch::Tensor scores,int64_t position,double scale) {
    c10::cuda::CUDAGuard device_guard(q.device());
    check_cuda_contig(q,"q");check_cuda_contig(current,"current");
    check_cuda_contig(cache,"cache");check_cuda_contig(scores,"scores");
    TORCH_CHECK(q.scalar_type()==torch::kFloat32,"native BOLT decode expects FP32 scaled query");
    check_f16_f32(cache,"cache");
    TORCH_CHECK(current.scalar_type()==cache.scalar_type(),"current/cache dtype mismatch");
    TORCH_CHECK(q.dim()==4 && q.size(2)==1 && q.sizes()==current.sizes(),"decode q/current require [B,H,1,R]");
    TORCH_CHECK(cache.dim()==4 && cache.size(0)==q.size(0) && cache.size(1)==q.size(1) && cache.size(3)==q.size(3),"cache dimension mismatch");
    TORCH_CHECK(scores.dim()==3 && scores.size(0)==q.size(0) && scores.size(1)==q.size(1) && scores.size(2)==cache.size(2),"scratch shape mismatch");
    TORCH_CHECK(scores.scalar_type()==torch::kFloat32,"scratch must be FP32");
    TORCH_CHECK(position>=0 && position<cache.size(2),"decode position outside cache");
    TORCH_CHECK(q.size(3)==16 || q.size(3)==32,"native BOLT decode supports latent16/32");
    auto out=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    const int nblocks=q.size(0)*q.size(1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(),"bolt_decode",[&]{
       bolt_decode_kernel<scalar_t><<<nblocks,128,0,at::cuda::getCurrentCUDAStream()>>>(
          q.data_ptr<float>(),current.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),
          scores.data_ptr<float>(),out.data_ptr<float>(),q.size(1),cache.size(2),q.size(3),position,float(scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
