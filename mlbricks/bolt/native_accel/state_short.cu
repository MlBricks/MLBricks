// BOLT AVQ v7 experimental: no all-FP32 [A,V,R] promotion, no full-width dz zeros.
// A = sigmoid(za), V = 0.1*tanh(zv), E[t] = A[t]*E[t-1] + V[t].
// Two execution paths: SWQR-inspired single CTA short scan (T<=512) and
// 256-token multi-CTA segmented scan with 256-channel tiling. Both have exact reverse-time gradients.
// FP32 recurrent states; FP16/FP32 input logits and output gradients.
// No torch.compile. No Triton. Native CUDA C++.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <algorithm>
constexpr int CHUNK=256;
constexpr int THREADS=256;
__device__ __forceinline__ float avq_sigmoid(float x){return 1.f/(1.f+__expf(-x));}

template<typename scalar_t>
__device__ __forceinline__ float rd(const scalar_t* z,int b,int t,int d,int64_t sb,int64_t st){
 return static_cast<float>(z[(int64_t)b*sb+(int64_t)t*st+d]);
}
template<typename scalar_t>
__device__ __forceinline__ float gate(const scalar_t* z,int b,int t,int d,int64_t sb,int64_t st){return avq_sigmoid(rd(z,b,t,d,sb,st));}
template<typename scalar_t>
__device__ __forceinline__ float upd(const scalar_t* z,int b,int t,int d,int64_t sb,int64_t st){return 0.1f*tanhf(rd(z,b,t,d,sb,st));}

template<typename scalar_t,int W>
__global__ void short_fwd(const scalar_t* __restrict__ za,const scalar_t* __restrict__ zv,
 const float* __restrict__ s0,float* __restrict__ out,int T,int D,
 int64_t asb,int64_t ast,int64_t vsb,int64_t vst){
 __shared__ float prod[W][32],bias[W][32];
 int lane=threadIdx.x&31,w=threadIdx.x>>5,d=blockIdx.x*32+lane,b=blockIdx.y;
 bool valid=d<D; int seg=(T+W-1)/W,start=w*seg,stop=min(T,start+seg);
 float p=1.f,s=0.f;
 if(valid)for(int t=start;t<stop;++t){float a=gate(za,b,t,d,asb,ast);s=a*s+upd(zv,b,t,d,vsb,vst);p*=a;}
 prod[w][lane]=p;bias[w][lane]=s;__syncthreads();
 if(!valid)return;
 float carry=s0[b*D+d];
 for(int j=0;j<w;++j)carry=prod[j][lane]*carry+bias[j][lane];
 for(int t=start;t<stop;++t){carry=gate(za,b,t,d,asb,ast)*carry+upd(zv,b,t,d,vsb,vst);out[((int64_t)b*T+t)*D+d]=carry;}
}

template<typename scalar_t,int W>
__global__ void short_bwd(const scalar_t* __restrict__ za,const scalar_t* __restrict__ zv,
 const float* __restrict__ states,const float* __restrict__ s0,const float* __restrict__ de,
 scalar_t* __restrict__ dza,scalar_t* __restrict__ dzv,float* __restrict__ d0,
 int T,int D,int64_t asb,int64_t ast,int64_t vsb,int64_t vst){
 __shared__ float prod[W][32],bias[W][32];
 int lane=threadIdx.x&31,w=threadIdx.x>>5,d=blockIdx.x*32+lane,b=blockIdx.y;
 bool valid=d<D;int seg=(T+W-1)/W,start=w*seg,stop=min(T,start+seg);
 float p=1.f,carry=0.f;
 if(valid)for(int t=stop-1;t>=start;--t){
  int64_t k=((int64_t)b*T+t)*D+d;
  float a=gate(za,b,t,d,asb,ast);
  float gt=de[k]+carry;carry=gt*a;p*=a;
 }
 prod[w][lane]=p;bias[w][lane]=carry;__syncthreads();
 if(!valid)return;
 carry=0.f;
 for(int j=W-1;j>w;--j)carry=prod[j][lane]*carry+bias[j][lane];
 for(int t=stop-1;t>=start;--t){
  int64_t k=((int64_t)b*T+t)*D+d;
  float a=gate(za,b,t,d,asb,ast);
  float gt=de[k]+carry;
  float prev=t==0?s0[b*D+d]:states[k-D];
  float u=tanhf(rd(zv,b,t,d,vsb,vst));
  dza[k]=static_cast<scalar_t>(gt*prev*a*(1.f-a));
  dzv[k]=static_cast<scalar_t>(gt*0.1f*(1.f-u*u));
  if(t==0)d0[b*D+d]=gt*a;
  carry=gt*a;
 }
}

template<typename scalar_t>
__global__ void seg_fwd_local(const scalar_t* za,const scalar_t* zv,float* states,float* ca,float* cb,
 int T,int D,int NC,int64_t asb,int64_t ast,int64_t vsb,int64_t vst){
 int d=blockIdx.z*blockDim.x+threadIdx.x,b=blockIdx.y,c=blockIdx.x;if(d>=D)return;
 float p=1.f,s=0.f;int stop=min((c+1)*CHUNK,T);
 for(int t=c*CHUNK;t<stop;++t){float a=gate(za,b,t,d,asb,ast);s=a*s+upd(zv,b,t,d,vsb,vst);p*=a;states[((int64_t)b*T+t)*D+d]=s;}
 int64_t i=((int64_t)b*NC+c)*D+d;ca[i]=p;cb[i]=s;
}
__global__ void seg_fwd_carry(const float* ca,const float* cb,const float* s0,float* inc,int D,int NC){
 int d=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(d>=D)return;
 float s=s0[b*D+d];for(int c=0;c<NC;++c){int64_t i=((int64_t)b*NC+c)*D+d;inc[i]=s;s=ca[i]*s+cb[i];}
}
template<typename scalar_t>
__global__ void seg_fwd_correct(const scalar_t* za,const float* inc,float* states,int T,int D,int NC,int64_t asb,int64_t ast){
 int d=blockIdx.z*blockDim.x+threadIdx.x,b=blockIdx.y,c=blockIdx.x;if(d>=D)return;
 float s=inc[((int64_t)b*NC+c)*D+d],p=1.f;
 for(int t=c*CHUNK;t<min((c+1)*CHUNK,T);++t){int64_t k=((int64_t)b*T+t)*D+d;p*=gate(za,b,t,d,asb,ast);states[k]+=p*s;}
}
template<typename scalar_t>
__global__ void seg_bwd_local(const scalar_t* za,const float* de,float* local,float* ca,float* cb,
 int T,int D,int NC,int64_t asb,int64_t ast){
 int d=blockIdx.z*blockDim.x+threadIdx.x,b=blockIdx.y,c=blockIdx.x;if(d>=D)return;
 int start=c*CHUNK,stop=min((c+1)*CHUNK,T);float p=1.f,carry=0.f;
 for(int t=stop-1;t>=start;--t){int64_t k=((int64_t)b*T+t)*D+d;float gt=de[k]+carry;local[k]=gt;float a=gate(za,b,t,d,asb,ast);carry=gt*a;p*=a;}
 int64_t i=((int64_t)b*NC+c)*D+d;ca[i]=p;cb[i]=carry;
}
__global__ void seg_bwd_carry(const float* ca,const float* cb,float* inc,int D,int NC){
 int d=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(d>=D)return;
 float s=0.f;for(int c=NC-1;c>=0;--c){int64_t i=((int64_t)b*NC+c)*D+d;inc[i]=s;s=ca[i]*s+cb[i];}
}
template<typename scalar_t>
__global__ void seg_bwd_correct(const scalar_t* za,const scalar_t* zv,const float* states,const float* s0,
 const float* local,const float* inc,scalar_t* dza,scalar_t* dzv,float* d0,
 int T,int D,int NC,int64_t asb,int64_t ast,int64_t vsb,int64_t vst){
 int d=blockIdx.z*blockDim.x+threadIdx.x,b=blockIdx.y,c=blockIdx.x;if(d>=D)return;
 float carry=inc[((int64_t)b*NC+c)*D+d],p=1.f;
 for(int t=min((c+1)*CHUNK,T)-1;t>=c*CHUNK;--t){
  int64_t k=((int64_t)b*T+t)*D+d;
  float a=gate(za,b,t,d,asb,ast);
  float gt=local[k]+p*carry;
  float prev=t==0?s0[b*D+d]:states[k-D];
  float u=tanhf(rd(zv,b,t,d,vsb,vst));
  dza[k]=static_cast<scalar_t>(gt*prev*a*(1.f-a));
  dzv[k]=static_cast<scalar_t>(gt*.1f*(1.f-u*u));
  if(t==0)d0[b*D+d]=gt*a;
  p*=a;
 }
}

static void valid(const torch::Tensor& a,const torch::Tensor& v,const torch::Tensor& s0){
 TORCH_CHECK(a.is_cuda() && v.is_cuda() && s0.is_cuda(),"CUDA required");
 TORCH_CHECK(a.scalar_type()==v.scalar_type(),"A/V dtype mismatch");
 TORCH_CHECK(a.scalar_type()==torch::kFloat16 || a.scalar_type()==torch::kFloat32,"A/V logits must be FP16 or FP32");
 TORCH_CHECK(s0.scalar_type()==torch::kFloat32 && s0.is_contiguous(),"state0 must be contiguous FP32");
 TORCH_CHECK(a.dim()==3 && v.sizes()==a.sizes(),"A/V logits [B,T,D]");
 TORCH_CHECK(a.size(0)>0 && a.size(0)<=65535 && a.size(1)>0 && a.size(2)>0 && a.size(2)<=int64_t(THREADS)*65535,"B in 1..65535, T>0, channel-tiled D");
 TORCH_CHECK(a.stride(2)==1 && v.stride(2)==1,"last-dimension contiguous required");
 TORCH_CHECK(s0.dim()==2 && s0.size(0)==a.size(0) && s0.size(1)==a.size(2),"state0 [B,D]");
 TORCH_CHECK(a.device()==v.device() && a.device()==s0.device(),"device mismatch");
 TORCH_CHECK(a.size(0)<=65535,"batch grid limit");
}
template<typename scalar_t>
static torch::Tensor launch_fwd(torch::Tensor a,torch::Tensor v,torch::Tensor s0,int w,bool use_short){
 int B=a.size(0),T=a.size(1),D=a.size(2),NC=(T+CHUNK-1)/CHUNK;
 auto states=torch::empty({B,T,D},s0.options());auto st=at::cuda::getCurrentCUDAStream().stream();
 const scalar_t* pa = a.data_ptr<scalar_t>();
 const scalar_t* pv = v.data_ptr<scalar_t>();
 const float* p0 = s0.data_ptr<float>();
 float* po = states.data_ptr<float>();
 if(use_short){dim3 grid((D+31)/32,B);
  if(w==4)short_fwd<scalar_t,4><<<grid,128,0,st>>>(pa,pv,p0,po,T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
  else if(w==8)short_fwd<scalar_t,8><<<grid,256,0,st>>>(pa,pv,p0,po,T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
  else short_fwd<scalar_t,16><<<grid,512,0,st>>>(pa,pv,p0,po,T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
 }else{
  auto summary=torch::empty({2,B,NC,D},s0.options());auto ca=summary.select(0,0),cb=summary.select(0,1);
  auto carries=torch::empty({B,NC,D},s0.options());dim3 grid(NC,B,(D+THREADS-1)/THREADS);
  seg_fwd_local<scalar_t><<<grid,THREADS,0,st>>>(pa,pv,po,ca.data_ptr<float>(),cb.data_ptr<float>(),T,D,NC,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_fwd_carry<<<dim3((D+THREADS-1)/THREADS,B),THREADS,0,st>>>(ca.data_ptr<float>(),cb.data_ptr<float>(),p0,carries.data_ptr<float>(),D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_fwd_correct<scalar_t><<<grid,THREADS,0,st>>>(pa,carries.data_ptr<float>(),po,T,D,NC,a.stride(0),a.stride(1));
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return states;
}
template<typename scalar_t>
static std::vector<torch::Tensor> launch_bwd(torch::Tensor a,torch::Tensor v,torch::Tensor states,torch::Tensor s0,torch::Tensor de,int w,bool use_short){
 int B=a.size(0),T=a.size(1),D=a.size(2),NC=(T+CHUNK-1)/CHUNK;
 auto da=torch::empty({B,T,D},a.options()),dv=torch::empty({B,T,D},v.options()),d0=torch::empty_like(s0);
 auto st=at::cuda::getCurrentCUDAStream().stream();
 const scalar_t* pa = a.data_ptr<scalar_t>();
 const scalar_t* pv = v.data_ptr<scalar_t>();
 if(use_short){dim3 grid((D+31)/32,B);
  if(w==4)short_bwd<scalar_t,4><<<grid,128,0,st>>>(pa,pv,states.data_ptr<float>(),s0.data_ptr<float>(),de.data_ptr<float>(),da.data_ptr<scalar_t>(),dv.data_ptr<scalar_t>(),d0.data_ptr<float>(),T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
  else if(w==8)short_bwd<scalar_t,8><<<grid,256,0,st>>>(pa,pv,states.data_ptr<float>(),s0.data_ptr<float>(),de.data_ptr<float>(),da.data_ptr<scalar_t>(),dv.data_ptr<scalar_t>(),d0.data_ptr<float>(),T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
  else short_bwd<scalar_t,16><<<grid,512,0,st>>>(pa,pv,states.data_ptr<float>(),s0.data_ptr<float>(),de.data_ptr<float>(),da.data_ptr<scalar_t>(),dv.data_ptr<scalar_t>(),d0.data_ptr<float>(),T,D,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
 }else{
  auto summary=torch::empty({2,B,NC,D},s0.options());auto ca=summary.select(0,0),cb=summary.select(0,1);
  auto carries=torch::empty({B,NC,D},s0.options());auto local=torch::empty_like(states);dim3 grid(NC,B,(D+THREADS-1)/THREADS);
  seg_bwd_local<scalar_t><<<grid,THREADS,0,st>>>(pa,de.data_ptr<float>(),local.data_ptr<float>(),ca.data_ptr<float>(),cb.data_ptr<float>(),T,D,NC,a.stride(0),a.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_bwd_carry<<<dim3((D+THREADS-1)/THREADS,B),THREADS,0,st>>>(ca.data_ptr<float>(),cb.data_ptr<float>(),carries.data_ptr<float>(),D,NC);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  seg_bwd_correct<scalar_t><<<grid,THREADS,0,st>>>(pa,pv,states.data_ptr<float>(),s0.data_ptr<float>(),local.data_ptr<float>(),carries.data_ptr<float>(),da.data_ptr<scalar_t>(),dv.data_ptr<scalar_t>(),d0.data_ptr<float>(),T,D,NC,a.stride(0),a.stride(1),v.stride(0),v.stride(1));
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {da,dv,d0};
}
torch::Tensor avq_v7_forward(torch::Tensor za,torch::Tensor zv,torch::Tensor s0,int64_t warps,bool short_path){
 valid(za,zv,s0);TORCH_CHECK(!short_path || za.size(1)<=512,"short path T<=512");
 TORCH_CHECK(warps==4 || warps==8 || warps==16,"warps 4/8/16");c10::cuda::CUDAGuard guard(za.device());
 if(za.scalar_type()==torch::kFloat16)return launch_fwd<at::Half>(za,zv,s0,warps,short_path);
 return launch_fwd<float>(za,zv,s0,warps,short_path);
}
std::vector<torch::Tensor> avq_v7_backward(torch::Tensor za,torch::Tensor zv,torch::Tensor states,torch::Tensor s0,torch::Tensor de,int64_t warps,bool short_path){
 valid(za,zv,s0);TORCH_CHECK(states.sizes()==za.sizes() && states.scalar_type()==torch::kFloat32 && states.is_contiguous(),"states FP32 contiguous [B,T,D]");
 TORCH_CHECK(de.sizes()==za.sizes() && de.scalar_type()==torch::kFloat32 && de.is_contiguous(),"dE FP32 contiguous [B,T,D]");
 TORCH_CHECK(de.device()==za.device() && states.device()==za.device(),"AVQ tensors must share the same CUDA device");
 TORCH_CHECK(!short_path || za.size(1)<=512,"short backward path T<=512");
 TORCH_CHECK(warps==4||warps==8||warps==16,"warps 4/8/16");c10::cuda::CUDAGuard guard(za.device());
 if(za.scalar_type()==torch::kFloat16)return launch_bwd<at::Half>(za,zv,states,s0,de,warps,short_path);
 return launch_bwd<float>(za,zv,states,s0,de,warps,short_path);
}
