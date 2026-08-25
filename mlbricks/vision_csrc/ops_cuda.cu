#include <torch/extension.h>
#include <ATen/AccumulateType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace mlbricks_vision {

__device__ __forceinline__ int64_t base_order_index_dev(int64_t t, int64_t h, int64_t w, bool vertical) {
    if (!vertical) {
        const int64_t row=t/w, off=t-row*w;
        const int64_t col=(row&1)?(w-1-off):off;
        return row*w+col;
    }
    const int64_t col=t/h, off=t-col*h;
    const int64_t row=(col&1)?(h-1-off):off;
    return row*w+col;
}

__device__ __forceinline__ int64_t order_index_dev(int64_t t,int64_t h,int64_t w,int64_t kind,int64_t phase){
    const int64_t n=h*w; bool rev=false, vertical=false, raster=false;
    if(kind==3){raster=true;rev=(phase&1);} else if(kind==1){rev=(phase&1);} else if(kind==2){vertical=true;rev=(phase&1);} else {
        int64_t p=phase%4; if(p<0)p+=4; vertical=p>=2; rev=(p==1||p==3);
    }
    const int64_t k=rev?(n-1-t):t;
    return raster?k:base_order_index_dev(k,h,w,vertical);
}

template<typename scalar_t>
__global__ void scan_reorder_kernel(const scalar_t* src, scalar_t* dst,int64_t B,int64_t N,int64_t D,int64_t h,int64_t w,int64_t kind,int64_t phase,bool inverse){
    const int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    const int64_t total=B*N*D; if(idx>=total)return;
    int64_t r=idx; const int64_t d=r%D; r/=D; const int64_t t=r%N; const int64_t b=r/N;
    const int64_t canonical=order_index_dev(t,h,w,kind,phase);
    if(!inverse) dst[idx]=src[(b*N+canonical)*D+d];
    else dst[(b*N+canonical)*D+d]=src[idx];
}

static inline int blocks_for(int64_t n,int threads){return (int)((n+threads-1)/threads);}

torch::Tensor scan_reorder_cuda(const torch::Tensor& x_in,int64_t h,int64_t w,int64_t kind,int64_t phase,bool inverse){
    c10::cuda::CUDAGuard guard(x_in.device()); auto x=x_in.contiguous(); auto out=torch::empty_like(x);
    const int64_t total=x.numel(); if(total==0)return out; constexpr int threads=256; cudaStream_t stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,x.scalar_type(),"vision_scan_reorder_cuda",[&]{
        scan_reorder_kernel<scalar_t><<<blocks_for(total,threads),threads,0,stream>>>(x.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),x.size(0),x.size(1),x.size(2),h,w,kind,phase,inverse);
    }); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

__device__ __forceinline__ float coord_value_dev(int64_t pos,int64_t length,int64_t channel,int64_t channels){
    if(channels<=0)return 0.f; const int64_t pairs=channels/2;
    if(pairs==0)return channel==0?(float)pos/(float)(length>1?length-1:1):0.f;
    if(channel>=2*pairs)return 0.f; const bool cosine=channel>=pairs; const int64_t j=cosine?channel-pairs:channel;
    const float inv=expf(-logf(10000.f)*(float)j/(float)(pairs>0?pairs:1)); const float a=(float)pos*inv; return cosine?cosf(a):sinf(a);
}

template<typename scalar_t>
__global__ void sincos2d_kernel(scalar_t* out,int64_t h,int64_t w,int64_t D){
    const int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; const int64_t total=h*w*D; if(idx>=total)return;
    const int64_t d=idx%D,t=idx/D,row=t/w,col=t-row*w,ydim=D/2,xdim=D-ydim;
    const float v=d<ydim?coord_value_dev(row,h,d,ydim):coord_value_dev(col,w,d-ydim,xdim); out[idx]=(scalar_t)v;
}

template<typename scalar_t>
__global__ void add_sincos2d_kernel(const scalar_t* x,scalar_t* out,int64_t B,int64_t h,int64_t w,int64_t D){
    const int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; const int64_t N=h*w,total=B*N*D; if(idx>=total)return;
    int64_t r=idx; const int64_t d=r%D; r/=D; const int64_t t=r%N; const int64_t row=t/w,col=t-row*w,ydim=D/2,xdim=D-ydim;
    const float v=d<ydim?coord_value_dev(row,h,d,ydim):coord_value_dev(col,w,d-ydim,xdim); out[idx]=(scalar_t)((float)x[idx]+v);
}

torch::Tensor sincos2d_cuda(const torch::Tensor& ref,int64_t h,int64_t w){
    c10::cuda::CUDAGuard guard(ref.device()); const int64_t D=ref.size(-1),total=h*w*D; auto out=torch::empty({h*w,D},ref.options()); if(total==0)return out;
    constexpr int threads=256; cudaStream_t stream=c10::cuda::getCurrentCUDAStream(ref.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,ref.scalar_type(),"vision_sincos2d_cuda",[&]{sincos2d_kernel<scalar_t><<<blocks_for(total,threads),threads,0,stream>>>(out.data_ptr<scalar_t>(),h,w,D);});
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor add_sincos2d_cuda(const torch::Tensor& x_in,int64_t h,int64_t w){
    c10::cuda::CUDAGuard guard(x_in.device()); auto x=x_in.contiguous(); auto out=torch::empty_like(x); const int64_t total=x.numel(); if(total==0)return out;
    constexpr int threads=256; cudaStream_t stream=c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,x.scalar_type(),"vision_add_sincos2d_cuda",[&]{add_sincos2d_kernel<scalar_t><<<blocks_for(total,threads),threads,0,stream>>>(x.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),x.size(0),h,w,x.size(2));});
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

template<typename scalar_t>
__global__ void unpatchify_kernel(const scalar_t* patches,scalar_t* out,int64_t B,int64_t C,int64_t gh,int64_t gw,int64_t p){
    const int64_t H=gh*p,W=gw*p,N=gh*gw,F=p*p*C,total=B*C*H*W; const int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=total)return;
    int64_t r=idx; const int64_t x=r%W; r/=W; const int64_t y=r%H; r/=H; const int64_t ch=r%C; const int64_t b=r/C;
    const int64_t pr=y/p,pc=x/p,iy=y%p,ix=x%p,t=pr*gw+pc,f=(iy*p+ix)*C+ch; out[idx]=patches[(b*N+t)*F+f];
}

template<typename scalar_t>
__global__ void patchify_layout_kernel(const scalar_t* image,scalar_t* out,int64_t B,int64_t C,int64_t H,int64_t W,int64_t p){
    const int64_t gh=H/p,gw=W/p,N=gh*gw,F=p*p*C,total=B*N*F; const int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=total)return;
    int64_t r=idx; const int64_t f=r%F; r/=F; const int64_t t=r%N; const int64_t b=r/N; const int64_t ch=f%C,pix=f/C,iy=pix/p,ix=pix%p,pr=t/gw,pc=t%gw,y=pr*p+iy,x=pc*p+ix;
    out[idx]=image[((b*C+ch)*H+y)*W+x];
}

torch::Tensor unpatchify_cuda(const torch::Tensor& patches_in,int64_t gh,int64_t gw,int64_t p,int64_t c){
    c10::cuda::CUDAGuard guard(patches_in.device()); auto patches=patches_in.contiguous(); auto out=torch::empty({patches.size(0),c,gh*p,gw*p},patches.options()); const int64_t total=out.numel(); if(total==0)return out; constexpr int threads=256; cudaStream_t stream=c10::cuda::getCurrentCUDAStream(patches.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,patches.scalar_type(),"vision_unpatchify_cuda",[&]{unpatchify_kernel<scalar_t><<<blocks_for(total,threads),threads,0,stream>>>(patches.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),patches.size(0),c,gh,gw,p);}); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

torch::Tensor patchify_layout_cuda(const torch::Tensor& image_in,int64_t p){
    c10::cuda::CUDAGuard guard(image_in.device()); auto image=image_in.contiguous(); const int64_t B=image.size(0),C=image.size(1),H=image.size(2),W=image.size(3),N=(H/p)*(W/p),F=p*p*C; auto out=torch::empty({B,N,F},image.options()); const int64_t total=out.numel(); if(total==0)return out; constexpr int threads=256; cudaStream_t stream=c10::cuda::getCurrentCUDAStream(image.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,image.scalar_type(),"vision_patchify_layout_cuda",[&]{patchify_layout_kernel<scalar_t><<<blocks_for(total,threads),threads,0,stream>>>(image.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),B,C,H,W,p);}); C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}


template<typename scalar_t>
__global__ void bolt_gate_rho_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ g,
    scalar_t* __restrict__ c,
    float* __restrict__ rho,
    int64_t rows,
    int64_t R,
    float eps) {
    const int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int64_t r = threadIdx.x; r < R; r += blockDim.x) {
        const int64_t idx = row * R + r;
        const float cv = (float)u[idx] * (1.0f + tanhf((float)g[idx]));
        c[idx] = (scalar_t)cv;
        sum += cv * cv;
    }
    // Block reduction; R<=64 and blockDim=64.
    __shared__ float smem[64];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) rho[row] = rsqrtf(smem[0] / (float)R + eps);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template<typename scalar_t>
__global__ void bolt_attention_online_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ c,
    const float* __restrict__ rho,
    scalar_t* __restrict__ out,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t R,
    float scale,
    bool causal) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t query_linear = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    const int64_t total_queries = B * H * T;
    if (query_linear >= total_queries) return;

    int64_t tmp = query_linear;
    const int64_t t = tmp % T; tmp /= T;
    const int64_t h = tmp % H; const int64_t b = tmp / H;

    const int64_t q_base = ((b * T + t) * H + h) * R;
    float acc0 = 0.0f, acc1 = 0.0f;
    float m = -INFINITY, l = 0.0f;
    const int64_t limit = causal ? (t + 1) : T;

    for (int64_t s = 0; s < limit; ++s) {
        const int64_t c_base = ((b * T + s) * H + h) * R;
        float dot = 0.0f;
        if (lane < R) dot += (float)q[q_base + lane] * (float)c[c_base + lane];
        if (lane + 32 < R) dot += (float)q[q_base + lane + 32] * (float)c[c_base + lane + 32];
        dot = warp_sum(dot);
        dot = __shfl_sync(0xffffffffu, dot, 0);
        const float score = dot * rho[(b * T + s) * H + h] * scale;
        const float new_m = fmaxf(m, score);
        const float alpha = isinf(m) ? 0.0f : expf(m - new_m);
        const float beta = expf(score - new_m);
        l = l * alpha + beta;
        if (lane < R) acc0 = acc0 * alpha + beta * (float)c[c_base + lane];
        if (lane + 32 < R) acc1 = acc1 * alpha + beta * (float)c[c_base + lane + 32];
        m = new_m;
    }
    const float inv_l = 1.0f / fmaxf(l, 1e-20f);
    if (lane < R) out[q_base + lane] = (scalar_t)(acc0 * inv_l);
    if (lane + 32 < R) out[q_base + lane + 32] = (scalar_t)(acc1 * inv_l);
}

torch::Tensor bolt_full_fused_cuda(
    const torch::Tensor& q_in,
    const torch::Tensor& u_in,
    const torch::Tensor& g_in,
    int64_t heads,
    int64_t latent_dim,
    int64_t head_dim,
    double eps,
    bool causal) {
    TORCH_CHECK(q_in.is_cuda() && u_in.is_cuda() && g_in.is_cuda(), "fused Bolt requires CUDA tensors");
    TORCH_CHECK(q_in.dim()==3 && q_in.sizes()==u_in.sizes() && q_in.sizes()==g_in.sizes(), "q/u/g shapes must match [B,T,H*R]");
    TORCH_CHECK(q_in.scalar_type()==u_in.scalar_type() && q_in.scalar_type()==g_in.scalar_type(), "q/u/g dtypes must match");
    TORCH_CHECK(q_in.size(2)==heads*latent_dim, "q last dimension mismatch");
    TORCH_CHECK(latent_dim > 0 && latent_dim <= 64, "latent_dim must be in 1..64");
    c10::cuda::CUDAGuard guard(q_in.device());
    auto q=q_in.contiguous(), u=u_in.contiguous(), g=g_in.contiguous();
    auto c=torch::empty_like(u); auto rho=torch::empty({q.size(0),q.size(1),heads}, q.options().dtype(torch::kFloat)); auto out=torch::empty_like(q);
    const int64_t rows=q.size(0)*q.size(1)*heads;
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,q.scalar_type(),"vision_bolt_full_fused_cuda",[&]{
        bolt_gate_rho_kernel<scalar_t><<<rows,64,0,stream>>>(u.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),c.data_ptr<scalar_t>(),rho.data_ptr<float>(),rows,latent_dim,(float)eps);
        const int threads=128, warps=threads/32; const int64_t total_q=q.size(0)*heads*q.size(1); const int blocks=(int)((total_q+warps-1)/warps);
        bolt_attention_online_kernel<scalar_t><<<blocks,threads,0,stream>>>(q.data_ptr<scalar_t>(),c.data_ptr<scalar_t>(),rho.data_ptr<float>(),out.data_ptr<scalar_t>(),q.size(0),q.size(1),heads,latent_dim,1.0f/sqrtf((float)head_dim),causal);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}

} // namespace mlbricks_vision
