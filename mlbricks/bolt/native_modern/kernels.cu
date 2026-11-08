// BOLT-16/32 width-independent AVQ CUDA kernels. Default AVQ matches full attention width.
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
// Direct-Q features. The separately learned query scale P was removed from BOLT.
// This kernel is used only for inference; the training projection and gate
// remain in PyTorch's autograd path.
template <typename scalar_t> __global__ void features_forward_kernel(
  const scalar_t *__restrict__ q, const scalar_t *__restrict__ u,
  const scalar_t *__restrict__ g, float *__restrict__ q_out,
  scalar_t *__restrict__ c_out, int64_t n) {
  const int64_t idx=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(idx>=n)return;
  const float uval=static_cast<float>(u[idx]);
  const float gval=static_cast<float>(g[idx]);
  const float qval=static_cast<float>(q[idx]);
  // Match PyTorch's intermediate casts for FP16 tanh -> add -> multiply.
  scalar_t gate=static_cast<scalar_t>(tanhf(gval));
  scalar_t gain=static_cast<scalar_t>(1.f+static_cast<float>(gate));
  c_out[idx]=static_cast<scalar_t>(uval*static_cast<float>(gain));
  q_out[idx]=qval;
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

// Opt-in split-context exact online-softmax attention read for the existing
// Direct-Q cache. It preserves the public decode_append API and FP32 O output.
// Cross-CTA safety: append MUST launch separately on the same CUDA stream.
// Cache and scratch are reused; the split path allocates no extra GPU tensors.
// The source model and AVQ mathematics are unchanged.
template <typename scalar_t>
__global__ void bolt_split_append_kernel(const scalar_t* __restrict__ c_now,
                                         scalar_t* __restrict__ cache,
                                         int64_t capacity, int64_t width,
                                         int64_t position) {
    const int bh=blockIdx.x;
    const int lane=threadIdx.x;
    if (lane<width)
        cache[(int64_t(bh)*capacity+position)*width+lane]=c_now[int64_t(bh)*width+lane];
}

// Each CTA works on one (batch,head,chunk). Eight warps work on independent
// tokens, maintaining online-softmax (max, denominator, weighted accumulator).
// Scratch stores [max, denominator, weighted NUMERATOR] for every chunk.
template <int R, typename scalar_t>
__global__ void bolt_split_read_kernel(
    const float* __restrict__ q,
    const scalar_t* __restrict__ cache,
    float* __restrict__ partial,
    int64_t capacity, int64_t length, int chunks,
    int chunk_tokens, float scale) {
    constexpr int NW=8;
    const int chunk_id=int(blockIdx.x)%chunks;
    const int bh=int(blockIdx.x)/chunks;
    const int warp=threadIdx.x/32;
    const int lane=threadIdx.x%32;
    const int64_t start=int64_t(chunk_id)*chunk_tokens;
    const int64_t last=start+int64_t(chunk_tokens);
    const int64_t end=(last<length)?last:length;
    const int64_t base=int64_t(bh)*capacity*R;
    const float qv=(lane<R)?q[int64_t(bh)*R+lane]:0.f;
    float local_m=-INFINITY;
    float local_z=0.f;
    float local_a=0.f;
    for(int64_t t=start+warp;t<end;t+=NW){
        float val=(lane<R)?static_cast<float>(cache[base+t*R+lane]):0.f;
        float dot=qv*val;
#pragma unroll
        for(int off=16;off>0;off/=2)
            dot+=__shfl_down_sync(0xffffffff,dot,off);
        const float score=__shfl_sync(0xffffffff,dot,0)*scale;
        const float mm=fmaxf(local_m,score);
        const float fa=(local_m==-INFINITY)?0.f:expf(local_m-mm);
        const float fb=expf(score-mm);
        local_z=local_z*fa+fb;
        local_a=local_a*fa+fb*val;
        local_m=mm;
    }
    __shared__ float warp_m[NW],warp_z[NW],warp_a[NW][32];
    __shared__ float weights[NW];
    __shared__ float max_m, total_z;
    if(lane==0){warp_m[warp]=local_m;warp_z[warp]=local_z;}
    warp_a[warp][lane]=local_a;
    __syncthreads();
    if(threadIdx.x==0){
        float m=-INFINITY;
        for(int w=0;w<NW;++w) m=fmaxf(m,warp_m[w]);
        float z=0.f;
        for(int w=0;w<NW;++w){
            const float f=(warp_m[w]==-INFINITY)?0.f:expf(warp_m[w]-m);
            weights[w]=f;
            z+=f*warp_z[w];
        }
        max_m=m;total_z=z;
    }
    __syncthreads();
    float* dst=partial+int64_t(bh)*capacity+int64_t(chunk_id)*(R+2);
    if(threadIdx.x==0){dst[0]=max_m;dst[1]=total_z;}
    if(threadIdx.x<R){
        float weighted=0.f;
        for(int w=0;w<NW;++w)weighted+=weights[w]*warp_a[w][threadIdx.x];
        dst[2+threadIdx.x]=weighted;
    }
}

template <int R>
__global__ void bolt_split_merge_kernel(const float* __restrict__ partial,
                                        float* __restrict__ output,
                                        int64_t capacity,int chunks){
    const int bh=blockIdx.x;
    const int lane=threadIdx.x;
    const float* src=partial+int64_t(bh)*capacity;
    float m=-INFINITY;
    for(int k=0;k<chunks;++k)m=fmaxf(m,src[int64_t(k)*(R+2)]);
    float denom=0.f, numerator=0.f;
    for(int k=0;k<chunks;++k){
        const float mk=src[int64_t(k)*(R+2)];
        const float f=(mk==-INFINITY)?0.f:expf(mk-m);
        denom+=f*src[int64_t(k)*(R+2)+1];
        if(lane<R)numerator+=f*src[int64_t(k)*(R+2)+2+lane];
    }
    if(lane<R)output[int64_t(bh)*R+lane]=numerator/fmaxf(denom,1.0e-30f);
}

}

// No P, no query folding, and no native P-gradient kernel.
std::vector<torch::Tensor> bolt_features_forward_cuda(torch::Tensor q, torch::Tensor u, torch::Tensor g) {
    c10::cuda::CUDAGuard device_guard(q.device());
    check_cuda_contig(q,"q");check_cuda_contig(u,"u");check_cuda_contig(g,"g");
    check_f16_f32(q,"q");
    TORCH_CHECK(q.device()==u.device() && q.device()==g.device(),
                "q, u and g must be on the same CUDA device");
    TORCH_CHECK(q.scalar_type()==u.scalar_type() && q.scalar_type()==g.scalar_type(),
                "feature dtypes must match");
    TORCH_CHECK(q.dim()==4 && q.sizes()==u.sizes() && q.sizes()==g.sizes(),
                "features require matching [B,H,T,R]");
    auto query=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    auto c=torch::empty_like(u);
    const int64_t n=q.numel();const int threads=256;
    if (n==0) return {query,c};
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(),"bolt_features_forward",[&]{
      features_forward_kernel<scalar_t><<< (n+threads-1)/threads,threads,0,at::cuda::getCurrentCUDAStream() >>>(
        q.data_ptr<scalar_t>(),u.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),
        query.data_ptr<float>(),c.data_ptr<scalar_t>(),n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {query,c};
}

torch::Tensor avq_scan_forward_cuda(torch::Tensor a,torch::Tensor v) {
    c10::cuda::CUDAGuard device_guard(a.device());
    check_cuda_contig(a,"a");check_cuda_contig(v,"v");
    TORCH_CHECK(a.device()==v.device(),"a and v must be on the same CUDA device");
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
    TORCH_CHECK(ds.device()==a.device() && s.device()==a.device(),"AVQ backward tensors must share a CUDA device");
    TORCH_CHECK(a.sizes()==s.sizes() && a.sizes()==ds.sizes() && a.dim()==3,"AVQ backward shape mismatch");
    TORCH_CHECK(a.scalar_type()==torch::kFloat32 && ds.scalar_type()==torch::kFloat32 &&
                s.scalar_type()==torch::kFloat32,"AVQ backward requires FP32");
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
    TORCH_CHECK(a.device()==v.device() && a.device()==previous.device(),"AVQ step tensors must share a CUDA device");
    TORCH_CHECK(a.scalar_type()==torch::kFloat32 && v.scalar_type()==torch::kFloat32 &&
                previous.scalar_type()==torch::kFloat32 && a.sizes()==v.sizes() &&
                a.sizes()==previous.sizes(),"AVQ step FP32 tensors must match");
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
    TORCH_CHECK(q.device()==current.device() && q.device()==cache.device() &&
                q.device()==scores.device(),"native BOLT decode tensors must share a CUDA device");
    TORCH_CHECK(q.scalar_type()==torch::kFloat32,"native BOLT decode expects FP32 query");
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


// Separate entry point: default decode_append remains untouched.
// scratch [B,H,capacity] FP32 is repurposed as chunk partials only in this path.
torch::Tensor bolt_decode_append_split_cuda(torch::Tensor q, torch::Tensor current,
    torch::Tensor cache, torch::Tensor scores, int64_t position,
    double scale, int64_t chunk_tokens) {
    c10::cuda::CUDAGuard device_guard(q.device());
    check_cuda_contig(q,"q"); check_cuda_contig(current,"current");
    check_cuda_contig(cache,"cache"); check_cuda_contig(scores,"scratch");
    TORCH_CHECK(q.device()==current.device() && q.device()==cache.device() &&
                q.device()==scores.device(),"split decode inputs must share device");
    TORCH_CHECK(q.scalar_type()==torch::kFloat32,"split decode query must be FP32");
    check_f16_f32(cache,"cache");
    TORCH_CHECK(current.scalar_type()==cache.scalar_type(),"current and cache dtype mismatch");
    TORCH_CHECK(q.dim()==4 && q.size(2)==1 && q.sizes()==current.sizes(),
                "split decode q/current require matching [B,H,1,R]");
    TORCH_CHECK(cache.dim()==4 && cache.size(0)==q.size(0) &&
                cache.size(1)==q.size(1) && cache.size(3)==q.size(3),
                "split decode cache dimension mismatch");
    TORCH_CHECK(scores.dim()==3 && scores.size(0)==q.size(0) &&
                scores.size(1)==q.size(1) && scores.size(2)==cache.size(2) &&
                scores.scalar_type()==torch::kFloat32,"scratch requires FP32 [B,H,capacity]");
    TORCH_CHECK(position>=0 && position<cache.size(2),"position outside cache");
    TORCH_CHECK(q.size(3)==16 || q.size(3)==32,"split decode requires BOLT-16/32");
    TORCH_CHECK(chunk_tokens>=128 && chunk_tokens<=4096 && chunk_tokens%8==0,
                "chunk must be 128..4096 and divisible by 8");
    const int64_t length=position+1;
    const int64_t chunks=(length+chunk_tokens-1)/chunk_tokens;
    TORCH_CHECK(chunks*(q.size(3)+2)<=scores.size(2),
                "scratch capacity insufficient for chunk partials");
    TORCH_CHECK(q.size(0)*q.size(1)*chunks<=2147483647LL,"too many CUDA blocks");
    auto output=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    const int bh=int(q.size(0)*q.size(1));
    const int R=int(q.size(3));
    const int64_t capacity=cache.size(2);
    cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(),"bolt_split_decode",[&]{
        // Cross-block memory dependency requires this independent kernel launch.
        bolt_split_append_kernel<scalar_t><<<bh,32,0,stream>>>(
            current.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),capacity,R,position);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if(R==16){
            bolt_split_read_kernel<16,scalar_t><<<bh*chunks,256,0,stream>>>(
                q.data_ptr<float>(),cache.data_ptr<scalar_t>(),scores.data_ptr<float>(),
                capacity,length,int(chunks),int(chunk_tokens),float(scale));
        }else{
            bolt_split_read_kernel<32,scalar_t><<<bh*chunks,256,0,stream>>>(
                q.data_ptr<float>(),cache.data_ptr<scalar_t>(),scores.data_ptr<float>(),
                capacity,length,int(chunks),int(chunk_tokens),float(scale));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    if(R==16){
        bolt_split_merge_kernel<16><<<bh,32,0,stream>>>(
            scores.data_ptr<float>(),output.data_ptr<float>(),capacity,int(chunks));
    }else{
        bolt_split_merge_kernel<32><<<bh,32,0,stream>>>(
            scores.data_ptr<float>(),output.data_ptr<float>(),capacity,int(chunks));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
