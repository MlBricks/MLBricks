// MLBricks BOLT dynamic-position CUDA Graph experimental decoder.
// Source semantics follow the qualified native Direct-Q reader.
// No Triton, no V3 GEMV, no change to AVQ V2 recurrence.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

namespace {
void checked(torch::Tensor x, const char *which) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), which, " must be contiguous CUDA");
}

// One CTA per B,H. Same operations/order as the original native single-CTA
// decoder, but position is read from persistent device memory at replay time.
template<typename scalar_t>
__global__ void dynamic_single(const float* __restrict__ q,
  const scalar_t* __restrict__ current, scalar_t* __restrict__ cache,
  float* __restrict__ scratch, float* __restrict__ output,
  const int64_t* __restrict__ position_ptr,
  int64_t capacity, int64_t width, float scale) {
    int bh=blockIdx.x;
    int tid=threadIdx.x;
    int64_t pos=position_ptr[0];
    if(pos<0 || pos>=capacity) return;
    const int64_t base=int64_t(bh)*capacity*width;
    const int64_t now=int64_t(bh)*width;
    __shared__ float shared[128];
    for(int d=tid;d<width;d+=128)
        cache[base+pos*width+d]=current[now+d];
    __syncthreads();
    float local_max=-INFINITY;
    for(int64_t t=tid;t<=pos;t+=128){
        float dot=0.f;
        for(int d=0;d<width;++d)
            dot+=q[now+d]*static_cast<float>(cache[base+t*width+d]);
        float score=dot*scale;
        scratch[int64_t(bh)*capacity+t]=score;
        local_max=fmaxf(local_max,score);
    }
    shared[tid]=local_max; __syncthreads();
    for(int stride=64;stride>0;stride>>=1){
        if(tid<stride)shared[tid]=fmaxf(shared[tid],shared[tid+stride]);
        __syncthreads();
    }
    float max_score=shared[0];
    float partial=0.f;
    for(int64_t t=tid;t<=pos;t+=128){
        float e=expf(scratch[int64_t(bh)*capacity+t]-max_score);
        scratch[int64_t(bh)*capacity+t]=e;
        partial+=e;
    }
    shared[tid]=partial; __syncthreads();
    for(int stride=64;stride>0;stride>>=1){
        if(tid<stride)shared[tid]+=shared[tid+stride];
        __syncthreads();
    }
    float inv_denom=1.f/shared[0];
    for(int d=tid;d<width;d+=128){
        float weighted=0.f;
        for(int64_t t=0;t<=pos;++t)
            weighted+=scratch[int64_t(bh)*capacity+t]*static_cast<float>(cache[base+t*width+d]);
        output[now+d]=weighted*inv_denom;
    }
}

template<typename scalar_t>
__global__ void append_current(const scalar_t* __restrict__ current,
    scalar_t* __restrict__ cache, const int64_t* __restrict__ position_ptr,
    int64_t capacity, int R){
    int bh=blockIdx.x, lane=threadIdx.x;
    int64_t pos=position_ptr[0];
    if(pos>=0 && pos<capacity && lane<R)
        cache[(int64_t(bh)*capacity+pos)*R+lane]=current[int64_t(bh)*R+lane];
}

// Fixed CUDA launch grid (max_chunks), but actual valid token count and
// corresponding active chunks are calculated from the live GPU position.
// Inactive chunks explicitly write neutral partials, avoiding stale scratch.
template<int R,typename scalar_t>
__global__ void dynamic_split_read(const float* __restrict__ q,
    const scalar_t* __restrict__ cache, float* __restrict__ partial,
    const int64_t* __restrict__ position_ptr, int64_t capacity,
    int max_chunks, int chunk_tokens, float scale){
    constexpr int NW=8;
    int chunk_id=blockIdx.x%max_chunks;
    int bh=blockIdx.x/max_chunks;
    int warp=threadIdx.x/32;
    int lane=threadIdx.x%32;
    int64_t pos=position_ptr[0];
    int64_t length=pos+1;
    int64_t start=int64_t(chunk_id)*chunk_tokens;
    int64_t last=start+int64_t(chunk_tokens);
    int64_t end=(last<length)?last:length;
    float *dst=partial+int64_t(bh)*capacity+int64_t(chunk_id)*(R+2);
    if(start>=length){
        if(threadIdx.x==0){dst[0]=-INFINITY;dst[1]=0.f;}
        if(threadIdx.x<R)dst[2+threadIdx.x]=0.f;
        return;
    }
    int64_t base=int64_t(bh)*capacity*R;
    float qv=(lane<R)?q[int64_t(bh)*R+lane]:0.f;
    float local_m=-INFINITY, local_z=0.f, local_a=0.f;
    for(int64_t t=start+warp;t<end;t+=NW){
        float val=(lane<R)?static_cast<float>(cache[base+t*R+lane]):0.f;
        float dot=qv*val;
        #pragma unroll
        for(int off=16;off>0;off/=2)dot+=__shfl_down_sync(0xffffffff,dot,off);
        float score=__shfl_sync(0xffffffff,dot,0)*scale;
        float mm=fmaxf(local_m,score);
        float fa=(local_m==-INFINITY)?0.f:expf(local_m-mm);
        float fb=expf(score-mm);
        local_z=local_z*fa+fb;
        local_a=local_a*fa+fb*val;
        local_m=mm;
    }
    __shared__ float warp_m[NW],warp_z[NW],warp_a[NW][32];
    __shared__ float weights[NW];
    if(lane==0){warp_m[warp]=local_m;warp_z[warp]=local_z;}
    warp_a[warp][lane]=local_a; __syncthreads();
    if(threadIdx.x==0){
        float m=-INFINITY;
        for(int w=0;w<NW;++w)m=fmaxf(m,warp_m[w]);
        float z=0.f;
        for(int w=0;w<NW;++w){
            float f=(warp_m[w]==-INFINITY)?0.f:expf(warp_m[w]-m);
            weights[w]=f;
            z+=f*warp_z[w];
        }
        dst[0]=m;dst[1]=z;
    }
    __syncthreads();
    if(threadIdx.x<R){
        float weighted=0.f;
        for(int w=0;w<NW;++w)weighted+=weights[w]*warp_a[w][threadIdx.x];
        dst[2+threadIdx.x]=weighted;
    }
}

template<int R>
__global__ void dynamic_split_merge(const float* __restrict__ partial,
    float* __restrict__ output, int64_t capacity,int max_chunks){
    int bh=blockIdx.x, lane=threadIdx.x;
    const float* src=partial+int64_t(bh)*capacity;
    float m=-INFINITY;
    for(int k=0;k<max_chunks;++k)m=fmaxf(m,src[int64_t(k)*(R+2)]);
    float denom=0.f, numerator=0.f;
    for(int k=0;k<max_chunks;++k){
        float mk=src[int64_t(k)*(R+2)];
        float f=(mk==-INFINITY)?0.f:expf(mk-m);
        denom+=f*src[int64_t(k)*(R+2)+1];
        if(lane<R)numerator+=f*src[int64_t(k)*(R+2)+2+lane];
    }
    if(lane<R)output[int64_t(bh)*R+lane]=numerator/fmaxf(denom,1.0e-30f);
}
}

torch::Tensor bolt_dynamic_cuda(torch::Tensor q,torch::Tensor current,
    torch::Tensor cache,torch::Tensor scratch,torch::Tensor position,
    double scale,int64_t chunk_tokens,int64_t max_chunks,bool use_split){
    c10::cuda::CUDAGuard guard(q.device());
    checked(q,"q");checked(current,"current");checked(cache,"cache");
    checked(scratch,"scratch");checked(position,"position");
    TORCH_CHECK(q.device()==current.device() && q.device()==cache.device() &&
                q.device()==scratch.device() && q.device()==position.device(),
                "all tensors must be on one CUDA device");
    TORCH_CHECK(q.scalar_type()==torch::kFloat32,"query must be FP32");
    TORCH_CHECK(cache.scalar_type()==torch::kFloat16 || cache.scalar_type()==torch::kFloat32,
                "cache must be FP16/FP32");
    TORCH_CHECK(current.scalar_type()==cache.scalar_type(),"current/cache dtype mismatch");
    TORCH_CHECK(position.scalar_type()==torch::kInt64 && position.numel()==1,
                "position must be persistent CUDA int64[1]");
    TORCH_CHECK(q.dim()==4 && q.size(2)==1 && q.sizes()==current.sizes(),
                "q/current require matching [batch, heads, 1, latent]");
    TORCH_CHECK(q.size(0)==1 && (q.size(3)==16 || q.size(3)==32),
                "experimental dynamic CUDA path supports batch1 BOLT16/32");
    TORCH_CHECK(cache.dim()==4 && cache.size(0)==q.size(0) && cache.size(1)==q.size(1)
                && cache.size(3)==q.size(3),"cache shape mismatch");
    TORCH_CHECK(scratch.dim()==3 && scratch.scalar_type()==torch::kFloat32 &&
                scratch.size(0)==q.size(0) && scratch.size(1)==q.size(1) &&
                scratch.size(2)==cache.size(2),"scratch must be FP32 [B,H,capacity]");
    TORCH_CHECK(chunk_tokens>=128 && chunk_tokens<=4096 && chunk_tokens%8==0,
                "chunk_tokens must be a multiple of 8 between 128 and 4096");
    TORCH_CHECK(max_chunks>=1 &&
                max_chunks*(q.size(3)+2)<=scratch.size(2),
                "scratch insufficient for split partials");
    TORCH_CHECK(max_chunks<= (cache.size(2)+chunk_tokens-1)/chunk_tokens,
                "max_chunks exceeds capacity");
    const int bh=int(q.size(0)*q.size(1));
    const int R=int(q.size(3));
    const int64_t capacity=cache.size(2);
    auto output=torch::empty(q.sizes(),q.options().dtype(torch::kFloat32));
    cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cache.scalar_type(),"dynamic_BOLT_CUDA",[&]{
        if(use_split){
            append_current<scalar_t><<<bh,32,0,stream>>>(
                current.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),
                position.data_ptr<int64_t>(),capacity,R);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if(R==16)
                dynamic_split_read<16,scalar_t><<<bh*int(max_chunks),256,0,stream>>>(
                    q.data_ptr<float>(),cache.data_ptr<scalar_t>(),scratch.data_ptr<float>(),
                    position.data_ptr<int64_t>(),capacity,int(max_chunks),int(chunk_tokens),float(scale));
            else
                dynamic_split_read<32,scalar_t><<<bh*int(max_chunks),256,0,stream>>>(
                    q.data_ptr<float>(),cache.data_ptr<scalar_t>(),scratch.data_ptr<float>(),
                    position.data_ptr<int64_t>(),capacity,int(max_chunks),int(chunk_tokens),float(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }else{
            dynamic_single<scalar_t><<<bh,128,0,stream>>>(
                q.data_ptr<float>(),current.data_ptr<scalar_t>(),cache.data_ptr<scalar_t>(),
                scratch.data_ptr<float>(),output.data_ptr<float>(),position.data_ptr<int64_t>(),
                capacity,R,float(scale));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
    if(use_split){
        if(R==16)
            dynamic_split_merge<16><<<bh,32,0,stream>>>(
                scratch.data_ptr<float>(),output.data_ptr<float>(),capacity,int(max_chunks));
        else
            dynamic_split_merge<32><<<bh,32,0,stream>>>(
                scratch.data_ptr<float>(),output.data_ptr<float>(),capacity,int(max_chunks));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}
