
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

static constexpr unsigned FULL_MASK = 0xffffffffu;
static constexpr int TILE_WARPS = 8;
static constexpr int TILE_THREADS = TILE_WARPS * 32;
static constexpr int MAX_W = 64;

__device__ __forceinline__ float warp_sum(float x) {
    x += __shfl_down_sync(FULL_MASK, x, 16);
    x += __shfl_down_sync(FULL_MASK, x, 8);
    x += __shfl_down_sync(FULL_MASK, x, 4);
    x += __shfl_down_sync(FULL_MASK, x, 2);
    x += __shfl_down_sync(FULL_MASK, x, 1);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
    x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, 16));
    x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, 8));
    x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, 4));
    x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, 2));
    x = fmaxf(x, __shfl_down_sync(FULL_MASK, x, 1));
    return x;
}


// ============================================================
// FUSED GAUSS CACHE CREATION
// (U,G) -> (C,rho)
// u,g,c: [B,H,R] FP16
// rho:   [B,H]   FP16
// one warp per B*H
// ============================================================

__global__ void gauss_gate_rho_kernel(
    const half* __restrict__ u,
    const half* __restrict__ g,
    half* __restrict__ c,
    half* __restrict__ rho,
    int BH,
    int R,
    float eps
) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;

    if (bh >= BH) return;

    const int d0 = lane;
    const int d1 = lane + 32;

    const long base = (long)bh * R;

    float ss = 0.0f;

    if (d0 < R) {
        const float uf = __half2float(u[base + d0]);
        const float gf = __half2float(g[base + d0]);
        const float cf = uf * (1.0f + tanhf(gf));
        const half ch = __float2half_rn(cf);
        c[base + d0] = ch;
        const float cr = __half2float(ch);
        ss += cr * cr;
    }

    if (d1 < R) {
        const float uf = __half2float(u[base + d1]);
        const float gf = __half2float(g[base + d1]);
        const float cf = uf * (1.0f + tanhf(gf));
        const half ch = __float2half_rn(cf);
        c[base + d1] = ch;
        const float cr = __half2float(ch);
        ss += cr * cr;
    }

    ss = warp_sum(ss);

    if (lane == 0) {
        const float r =
            rsqrtf(ss / (float)R + eps);

        rho[bh] = __float2half_rn(r);
    }
}


// ============================================================
// STREAMING BASELINE PARTIAL
// one warp per sequence split
// ============================================================

__global__ void baseline_stream_partial(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int D,
    int splits,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;

    if (bh >= BH) return;

    const int lane = threadIdx.x;

    const int start =
        (int)(((long long)T * s) / splits);

    const int end =
        (int)(((long long)T * (s + 1)) / splits);

    const int d0 = lane;
    const int d1 = lane + 32;

    const long qbase = (long)bh * D;
    const long hbase = (long)bh * Tstride * D;

    const float q0 =
        d0 < D ? __half2float(q[qbase+d0]) : 0.0f;

    const float q1 =
        d1 < D ? __half2float(q[qbase+d1]) : 0.0f;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int t=start; t<end; ++t) {
        const long base = hbase + (long)t * D;

        const float k0 =
            d0 < D ? __half2float(k[base+d0]) : 0.0f;

        const float k1 =
            d1 < D ? __half2float(k[base+d1]) : 0.0f;

        const float v0 =
            d0 < D ? __half2float(v[base+d0]) : 0.0f;

        const float v1 =
            d1 < D ? __half2float(v[base+d1]) : 0.0f;

        float dot = warp_sum(q0*k0 + q1*k1);

        float alpha = 0.0f;
        float beta = 0.0f;

        if (lane == 0) {
            const float score = dot * scale;
            const float nm = fmaxf(m, score);

            alpha =
                (m == -FLT_MAX)
                ? 0.0f
                : __expf(m-nm);

            beta = __expf(score-nm);

            l = l*alpha + beta;
            m = nm;
        }

        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        beta  = __shfl_sync(FULL_MASK, beta, 0);

        if (d0 < D) a0 = a0*alpha + beta*v0;
        if (d1 < D) a1 = a1*alpha + beta*v1;
    }

    if (lane == 0) {
        pm[block] = m;
        pl[block] = l;
    }

    const long obase = (long)block * D;

    if (d0 < D) po[obase+d0] = a0;
    if (d1 < D) po[obase+d1] = a1;
}


// ============================================================
// STREAMING GAUSS PARTIAL
// ============================================================

__global__ void gauss_stream_partial(
    const half* __restrict__ q,
    const half* __restrict__ c,
    const half* __restrict__ rho,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;

    if (bh >= BH) return;

    const int lane = threadIdx.x;

    const int start =
        (int)(((long long)T * s) / splits);

    const int end =
        (int)(((long long)T * (s + 1)) / splits);

    const int d0 = lane;
    const int d1 = lane + 32;

    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;

    const float q0 =
        d0 < R ? __half2float(q[qbase+d0]) : 0.0f;

    const float q1 =
        d1 < R ? __half2float(q[qbase+d1]) : 0.0f;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int t=start; t<end; ++t) {
        const long base = hbase + (long)t * R;

        const float c0 =
            d0 < R ? __half2float(c[base+d0]) : 0.0f;

        const float c1 =
            d1 < R ? __half2float(c[base+d1]) : 0.0f;

        float dot = warp_sum(q0*c0 + q1*c1);

        float alpha = 0.0f;
        float beta = 0.0f;

        if (lane == 0) {
            const float rr =
                __half2float(rho[rbase+t]);

            const float score =
                dot * rr * scale;

            const float nm =
                fmaxf(m,score);

            alpha =
                (m == -FLT_MAX)
                ? 0.0f
                : __expf(m-nm);

            beta =
                __expf(score-nm);

            l = l*alpha + beta;
            m = nm;
        }

        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        beta  = __shfl_sync(FULL_MASK, beta, 0);

        if (d0 < R) a0 = a0*alpha + beta*c0;
        if (d1 < R) a1 = a1*alpha + beta*c1;
    }

    if (lane == 0) {
        pm[block] = m;
        pl[block] = l;
    }

    const long obase = (long)block * R;

    if (d0 < R) po[obase+d0] = a0;
    if (d1 < R) po[obase+d1] = a1;
}


// ============================================================
// TILED BASELINE PARTIAL
// 8 warps score 8 tokens in parallel
// ============================================================

__global__ void baseline_tiled_partial(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int D,
    int splits,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;

    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int start =
        (int)(((long long)T * s) / splits);

    const int end =
        (int)(((long long)T * (s+1)) / splits);

    const int d0 = lane;
    const int d1 = lane + 32;

    const long qbase = (long)bh * D;
    const long hbase = (long)bh * Tstride * D;

    const float q0 =
        d0 < D ? __half2float(q[qbase+d0]) : 0.0f;

    const float q1 =
        d1 < D ? __half2float(q[qbase+d1]) : 0.0f;

    __shared__ float scores[TILE_WARPS];
    __shared__ float weights[TILE_WARPS];
    __shared__ float vals[TILE_WARPS][MAX_W];
    __shared__ float alpha_s;
    __shared__ float beta_s;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int tile=start; tile<end; tile+=TILE_WARPS) {
        const int t = tile + warp;
        const bool valid = t < end;

        float vv0 = 0.0f;
        float vv1 = 0.0f;
        float score = -FLT_MAX;

        if (valid) {
            const long base = hbase + (long)t * D;

            const float k0 =
                d0 < D ? __half2float(k[base+d0]) : 0.0f;

            const float k1 =
                d1 < D ? __half2float(k[base+d1]) : 0.0f;

            vv0 =
                d0 < D ? __half2float(v[base+d0]) : 0.0f;

            vv1 =
                d1 < D ? __half2float(v[base+d1]) : 0.0f;

            const float dot =
                warp_sum(q0*k0 + q1*k1);

            if (lane == 0) {
                score = dot * scale;
            }
        }

        if (lane == 0) scores[warp] = score;
        if (d0 < D) vals[warp][d0] = valid ? vv0 : 0.0f;
        if (d1 < D) vals[warp][d1] = valid ? vv1 : 0.0f;

        __syncthreads();

        if (tid == 0) {
            float tm = -FLT_MAX;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w)
                tm = fmaxf(tm, scores[w]);

            float tl = 0.0f;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float sw = scores[w];

                const float ew =
                    sw == -FLT_MAX
                    ? 0.0f
                    : __expf(sw-tm);

                weights[w] = ew;
                tl += ew;
            }

            const float nm = fmaxf(m,tm);

            alpha_s =
                m == -FLT_MAX
                ? 0.0f
                : __expf(m-nm);

            beta_s =
                tm == -FLT_MAX
                ? 0.0f
                : __expf(tm-nm);

            l = l*alpha_s + tl*beta_s;
            m = nm;
        }

        __syncthreads();

        if (warp == 0) {
            float ta0 = 0.0f;
            float ta1 = 0.0f;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float wt = weights[w];

                if (d0 < D)
                    ta0 += wt * vals[w][d0];

                if (d1 < D)
                    ta1 += wt * vals[w][d1];
            }

            if (d0 < D)
                a0 = a0*alpha_s + ta0*beta_s;

            if (d1 < D)
                a1 = a1*alpha_s + ta1*beta_s;
        }

        __syncthreads();
    }

    if (tid == 0) {
        pm[block] = m;
        pl[block] = l;
    }

    if (warp == 0) {
        const long obase = (long)block * D;

        if (d0 < D) po[obase+d0] = a0;
        if (d1 < D) po[obase+d1] = a1;
    }
}


// ============================================================
// TILED GAUSS PARTIAL
// ============================================================

__global__ void gauss_tiled_partial(
    const half* __restrict__ q,
    const half* __restrict__ c,
    const half* __restrict__ rho,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;

    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int start =
        (int)(((long long)T * s) / splits);

    const int end =
        (int)(((long long)T * (s+1)) / splits);

    const int d0 = lane;
    const int d1 = lane + 32;

    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;

    const float q0 =
        d0 < R ? __half2float(q[qbase+d0]) : 0.0f;

    const float q1 =
        d1 < R ? __half2float(q[qbase+d1]) : 0.0f;

    __shared__ float scores[TILE_WARPS];
    __shared__ float weights[TILE_WARPS];
    __shared__ float vals[TILE_WARPS][MAX_W];
    __shared__ float alpha_s;
    __shared__ float beta_s;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int tile=start; tile<end; tile+=TILE_WARPS) {
        const int t = tile + warp;
        const bool valid = t < end;

        float cv0 = 0.0f;
        float cv1 = 0.0f;
        float score = -FLT_MAX;

        if (valid) {
            const long base = hbase + (long)t * R;

            cv0 =
                d0 < R ? __half2float(c[base+d0]) : 0.0f;

            cv1 =
                d1 < R ? __half2float(c[base+d1]) : 0.0f;

            const float dot =
                warp_sum(q0*cv0 + q1*cv1);

            if (lane == 0) {
                const float rr =
                    __half2float(rho[rbase+t]);

                score = dot * rr * scale;
            }
        }

        if (lane == 0) scores[warp] = score;
        if (d0 < R) vals[warp][d0] = valid ? cv0 : 0.0f;
        if (d1 < R) vals[warp][d1] = valid ? cv1 : 0.0f;

        __syncthreads();

        if (tid == 0) {
            float tm = -FLT_MAX;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w)
                tm = fmaxf(tm, scores[w]);

            float tl = 0.0f;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float sw = scores[w];

                const float ew =
                    sw == -FLT_MAX
                    ? 0.0f
                    : __expf(sw-tm);

                weights[w] = ew;
                tl += ew;
            }

            const float nm = fmaxf(m,tm);

            alpha_s =
                m == -FLT_MAX
                ? 0.0f
                : __expf(m-nm);

            beta_s =
                tm == -FLT_MAX
                ? 0.0f
                : __expf(tm-nm);

            l = l*alpha_s + tl*beta_s;
            m = nm;
        }

        __syncthreads();

        if (warp == 0) {
            float ta0 = 0.0f;
            float ta1 = 0.0f;

            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float wt = weights[w];

                if (d0 < R)
                    ta0 += wt * vals[w][d0];

                if (d1 < R)
                    ta1 += wt * vals[w][d1];
            }

            if (d0 < R)
                a0 = a0*alpha_s + ta0*beta_s;

            if (d1 < R)
                a1 = a1*alpha_s + ta1*beta_s;
        }

        __syncthreads();
    }

    if (tid == 0) {
        pm[block] = m;
        pl[block] = l;
    }

    if (warp == 0) {
        const long obase = (long)block * R;

        if (d0 < R) po[obase+d0] = a0;
        if (d1 < R) po[obase+d1] = a1;
    }
}



// ============================================================
// APPEND-AWARE PARTIALS
// The current token is consumed directly from k_now/v_now or c_now/rho_now
// while one designated split writes it into the persistent cache for the next
// decode step.  No separate cache-write CUDA launch is required.
// ============================================================

__global__ void baseline_stream_append_partial(
    const half* __restrict__ q,
    half* __restrict__ k,
    half* __restrict__ v,
    const half* __restrict__ k_now,
    const half* __restrict__ v_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int D,
    int splits,
    int position,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int lane = threadIdx.x;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s + 1)) / splits);
    const int d0 = lane;
    const int d1 = lane + 32;
    const long qbase = (long)bh * D;
    const long hbase = (long)bh * Tstride * D;
    const long write_base = hbase + (long)position * D;

    // Exactly one split persists the current token. All splits that need the
    // current token consume k_now/v_now directly, so no cross-block barrier is
    // needed inside this kernel launch.
    if (s == 0) {
        if (d0 < D) {
            k[write_base + d0] = k_now[qbase + d0];
            v[write_base + d0] = v_now[qbase + d0];
        }
        if (d1 < D) {
            k[write_base + d1] = k_now[qbase + d1];
            v[write_base + d1] = v_now[qbase + d1];
        }
    }

    const float q0 = d0 < D ? __half2float(q[qbase+d0]) : 0.0f;
    const float q1 = d1 < D ? __half2float(q[qbase+d1]) : 0.0f;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int t=start; t<end; ++t) {
        const bool current = (t == position);
        const long base = hbase + (long)t * D;
        const float k0 = d0 < D ? __half2float(current ? k_now[qbase+d0] : k[base+d0]) : 0.0f;
        const float k1 = d1 < D ? __half2float(current ? k_now[qbase+d1] : k[base+d1]) : 0.0f;
        const float v0 = d0 < D ? __half2float(current ? v_now[qbase+d0] : v[base+d0]) : 0.0f;
        const float v1 = d1 < D ? __half2float(current ? v_now[qbase+d1] : v[base+d1]) : 0.0f;

        float dot = warp_sum(q0*k0 + q1*k1);
        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float score = dot * scale;
            const float nm = fmaxf(m, score);
            alpha = (m == -FLT_MAX) ? 0.0f : __expf(m-nm);
            beta = __expf(score-nm);
            l = l*alpha + beta;
            m = nm;
        }
        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        beta = __shfl_sync(FULL_MASK, beta, 0);
        if (d0 < D) a0 = a0*alpha + beta*v0;
        if (d1 < D) a1 = a1*alpha + beta*v1;
    }

    if (lane == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    const long obase = (long)block * D;
    if (d0 < D) po[obase+d0] = a0;
    if (d1 < D) po[obase+d1] = a1;
}

__global__ void gauss_stream_append_partial(
    const half* __restrict__ q,
    half* __restrict__ c,
    half* __restrict__ rho,
    const half* __restrict__ c_now,
    const half* __restrict__ rho_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    int position,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int lane = threadIdx.x;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s + 1)) / splits);
    const int d0 = lane;
    const int d1 = lane + 32;
    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;
    const long write_base = hbase + (long)position * R;

    if (s == 0) {
        if (d0 < R) c[write_base+d0] = c_now[qbase+d0];
        if (d1 < R) c[write_base+d1] = c_now[qbase+d1];
        if (lane == 0) rho[rbase+position] = rho_now[bh];
    }

    const float q0 = d0 < R ? __half2float(q[qbase+d0]) : 0.0f;
    const float q1 = d1 < R ? __half2float(q[qbase+d1]) : 0.0f;
    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int t=start; t<end; ++t) {
        const bool current = (t == position);
        const long base = hbase + (long)t * R;
        const float c0 = d0 < R ? __half2float(current ? c_now[qbase+d0] : c[base+d0]) : 0.0f;
        const float c1 = d1 < R ? __half2float(current ? c_now[qbase+d1] : c[base+d1]) : 0.0f;
        float dot = warp_sum(q0*c0 + q1*c1);
        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float rr = __half2float(current ? rho_now[bh] : rho[rbase+t]);
            const float score = dot * rr * scale;
            const float nm = fmaxf(m,score);
            alpha = (m == -FLT_MAX) ? 0.0f : __expf(m-nm);
            beta = __expf(score-nm);
            l = l*alpha + beta;
            m = nm;
        }
        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        beta = __shfl_sync(FULL_MASK, beta, 0);
        if (d0 < R) a0 = a0*alpha + beta*c0;
        if (d1 < R) a1 = a1*alpha + beta*c1;
    }

    if (lane == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    const long obase = (long)block * R;
    if (d0 < R) po[obase+d0] = a0;
    if (d1 < R) po[obase+d1] = a1;
}

__global__ void baseline_tiled_append_partial(
    const half* __restrict__ q,
    half* __restrict__ k,
    half* __restrict__ v,
    const half* __restrict__ k_now,
    const half* __restrict__ v_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int D,
    int splits,
    int position,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s+1)) / splits);
    const int d0 = lane;
    const int d1 = lane + 32;
    const long qbase = (long)bh * D;
    const long hbase = (long)bh * Tstride * D;
    const long write_base = hbase + (long)position * D;

    if (s == 0 && warp == 0) {
        if (d0 < D) {
            k[write_base+d0] = k_now[qbase+d0];
            v[write_base+d0] = v_now[qbase+d0];
        }
        if (d1 < D) {
            k[write_base+d1] = k_now[qbase+d1];
            v[write_base+d1] = v_now[qbase+d1];
        }
    }

    const float q0 = d0 < D ? __half2float(q[qbase+d0]) : 0.0f;
    const float q1 = d1 < D ? __half2float(q[qbase+d1]) : 0.0f;

    __shared__ float scores[TILE_WARPS];
    __shared__ float weights[TILE_WARPS];
    __shared__ float vals[TILE_WARPS][MAX_W];
    __shared__ float alpha_s;
    __shared__ float beta_s;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int tile=start; tile<end; tile+=TILE_WARPS) {
        const int t = tile + warp;
        const bool valid = t < end;
        float vv0 = 0.0f;
        float vv1 = 0.0f;
        float score = -FLT_MAX;

        if (valid) {
            const bool current = (t == position);
            const long base = hbase + (long)t * D;
            const float k0 = d0 < D ? __half2float(current ? k_now[qbase+d0] : k[base+d0]) : 0.0f;
            const float k1 = d1 < D ? __half2float(current ? k_now[qbase+d1] : k[base+d1]) : 0.0f;
            vv0 = d0 < D ? __half2float(current ? v_now[qbase+d0] : v[base+d0]) : 0.0f;
            vv1 = d1 < D ? __half2float(current ? v_now[qbase+d1] : v[base+d1]) : 0.0f;
            const float dot = warp_sum(q0*k0 + q1*k1);
            if (lane == 0) score = dot * scale;
        }

        if (lane == 0) scores[warp] = score;
        if (d0 < D) vals[warp][d0] = valid ? vv0 : 0.0f;
        if (d1 < D) vals[warp][d1] = valid ? vv1 : 0.0f;
        __syncthreads();

        if (tid == 0) {
            float tm = -FLT_MAX;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) tm = fmaxf(tm, scores[w]);
            float tl = 0.0f;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float sw = scores[w];
                const float ew = sw == -FLT_MAX ? 0.0f : __expf(sw-tm);
                weights[w] = ew;
                tl += ew;
            }
            const float nm = fmaxf(m,tm);
            alpha_s = m == -FLT_MAX ? 0.0f : __expf(m-nm);
            beta_s = tm == -FLT_MAX ? 0.0f : __expf(tm-nm);
            l = l*alpha_s + tl*beta_s;
            m = nm;
        }
        __syncthreads();

        if (warp == 0) {
            float ta0 = 0.0f;
            float ta1 = 0.0f;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float wt = weights[w];
                if (d0 < D) ta0 += wt * vals[w][d0];
                if (d1 < D) ta1 += wt * vals[w][d1];
            }
            if (d0 < D) a0 = a0*alpha_s + ta0*beta_s;
            if (d1 < D) a1 = a1*alpha_s + ta1*beta_s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    if (warp == 0) {
        const long obase = (long)block * D;
        if (d0 < D) po[obase+d0] = a0;
        if (d1 < D) po[obase+d1] = a1;
    }
}

__global__ void gauss_tiled_append_partial(
    const half* __restrict__ q,
    half* __restrict__ c,
    half* __restrict__ rho,
    const half* __restrict__ c_now,
    const half* __restrict__ rho_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    int position,
    float scale
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s+1)) / splits);
    const int d0 = lane;
    const int d1 = lane + 32;
    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;
    const long write_base = hbase + (long)position * R;

    if (s == 0 && warp == 0) {
        if (d0 < R) c[write_base+d0] = c_now[qbase+d0];
        if (d1 < R) c[write_base+d1] = c_now[qbase+d1];
        if (lane == 0) rho[rbase+position] = rho_now[bh];
    }

    const float q0 = d0 < R ? __half2float(q[qbase+d0]) : 0.0f;
    const float q1 = d1 < R ? __half2float(q[qbase+d1]) : 0.0f;

    __shared__ float scores[TILE_WARPS];
    __shared__ float weights[TILE_WARPS];
    __shared__ float vals[TILE_WARPS][MAX_W];
    __shared__ float alpha_s;
    __shared__ float beta_s;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;

    for (int tile=start; tile<end; tile+=TILE_WARPS) {
        const int t = tile + warp;
        const bool valid = t < end;
        float cv0 = 0.0f;
        float cv1 = 0.0f;
        float score = -FLT_MAX;

        if (valid) {
            const bool current = (t == position);
            const long base = hbase + (long)t * R;
            cv0 = d0 < R ? __half2float(current ? c_now[qbase+d0] : c[base+d0]) : 0.0f;
            cv1 = d1 < R ? __half2float(current ? c_now[qbase+d1] : c[base+d1]) : 0.0f;
            const float dot = warp_sum(q0*cv0 + q1*cv1);
            if (lane == 0) {
                const float rr = __half2float(current ? rho_now[bh] : rho[rbase+t]);
                score = dot * rr * scale;
            }
        }

        if (lane == 0) scores[warp] = score;
        if (d0 < R) vals[warp][d0] = valid ? cv0 : 0.0f;
        if (d1 < R) vals[warp][d1] = valid ? cv1 : 0.0f;
        __syncthreads();

        if (tid == 0) {
            float tm = -FLT_MAX;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) tm = fmaxf(tm, scores[w]);
            float tl = 0.0f;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float sw = scores[w];
                const float ew = sw == -FLT_MAX ? 0.0f : __expf(sw-tm);
                weights[w] = ew;
                tl += ew;
            }
            const float nm = fmaxf(m,tm);
            alpha_s = m == -FLT_MAX ? 0.0f : __expf(m-nm);
            beta_s = tm == -FLT_MAX ? 0.0f : __expf(tm-nm);
            l = l*alpha_s + tl*beta_s;
            m = nm;
        }
        __syncthreads();

        if (warp == 0) {
            float ta0 = 0.0f;
            float ta1 = 0.0f;
            #pragma unroll
            for (int w=0; w<TILE_WARPS; ++w) {
                const float wt = weights[w];
                if (d0 < R) ta0 += wt * vals[w][d0];
                if (d1 < R) ta1 += wt * vals[w][d1];
            }
            if (d0 < R) a0 = a0*alpha_s + ta0*beta_s;
            if (d1 < R) a1 = a1*alpha_s + ta1*beta_s;
        }
        __syncthreads();
    }

    if (tid == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    if (warp == 0) {
        const long obase = (long)block * R;
        if (d0 < R) po[obase+d0] = a0;
        if (d1 < R) po[obase+d1] = a1;
    }
}


// ============================================================
// GAUSS + RoPE PARTIALS
// q is already RoPE-positioned for the current query position.  C remains
// raw in the compact cache/value path.  The key view is rotated on-the-fly
// inside the score kernel, avoiding materialization of RoPE(C[0:T]).
// Each lane owns one adjacent even/odd pair so the rotary transform is local.
// ============================================================

template <bool APPEND>
__global__ void gauss_rope_stream_partial(
    const half* __restrict__ q,
    half* __restrict__ c,
    half* __restrict__ rho,
    const half* __restrict__ c_now,
    const half* __restrict__ rho_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    int position,
    float scale,
    float rope_log_base,
    int rope_dim
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int lane = threadIdx.x;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s + 1)) / splits);
    const int d0 = lane * 2;
    const int d1 = d0 + 1;
    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;

    if (APPEND && s == 0) {
        const long write_base = hbase + (long)position * R;
        if (d0 < R) c[write_base + d0] = c_now[qbase + d0];
        if (d1 < R) c[write_base + d1] = c_now[qbase + d1];
        if (lane == 0) rho[rbase + position] = rho_now[bh];
    }

    const float q0 = d0 < R ? __half2float(q[qbase + d0]) : 0.0f;
    const float q1 = d1 < R ? __half2float(q[qbase + d1]) : 0.0f;

    const bool rotary = d1 < rope_dim;
    float theta = 0.0f;
    float sin_a = 0.0f, cos_a = 1.0f;
    float sin_step = 0.0f, cos_step = 1.0f;
    if (rotary) {
        theta = __expf(-rope_log_base * ((float)d0 / (float)rope_dim));
        sincosf((float)start * theta, &sin_a, &cos_a);
        sincosf(theta, &sin_step, &cos_step);
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    int iter = 0;

    for (int t = start; t < end; ++t, ++iter) {
        const bool current = APPEND && (t == position);
        const long base = hbase + (long)t * R;
        const float cv0 = d0 < R
            ? __half2float(current ? c_now[qbase + d0] : c[base + d0]) : 0.0f;
        const float cv1 = d1 < R
            ? __half2float(current ? c_now[qbase + d1] : c[base + d1]) : 0.0f;

        float k0 = cv0;
        float k1 = cv1;
        if (rotary) {
            // Match MLBricks RoPE key semantics, including FP16 materialized
            // key rounding, without ever allocating a rotated cache tensor.
            k0 = __half2float(__float2half_rn(cv0 * cos_a - cv1 * sin_a));
            k1 = __half2float(__float2half_rn(cv0 * sin_a + cv1 * cos_a));
        }

        const float dot = warp_sum(q0 * k0 + q1 * k1);
        float alpha = 0.0f;
        float beta = 0.0f;
        if (lane == 0) {
            const float rr = __half2float(current ? rho_now[bh] : rho[rbase + t]);
            const float score = dot * rr * scale;
            const float nm = fmaxf(m, score);
            alpha = (m == -FLT_MAX) ? 0.0f : __expf(m - nm);
            beta = __expf(score - nm);
            l = l * alpha + beta;
            m = nm;
        }
        alpha = __shfl_sync(FULL_MASK, alpha, 0);
        beta = __shfl_sync(FULL_MASK, beta, 0);
        if (d0 < R) a0 = a0 * alpha + beta * cv0;
        if (d1 < R) a1 = a1 * alpha + beta * cv1;

        if (rotary && t + 1 < end) {
            if (((iter + 1) & 63) == 0) {
                sincosf((float)(t + 1) * theta, &sin_a, &cos_a);
            } else {
                const float nc = cos_a * cos_step - sin_a * sin_step;
                const float ns = sin_a * cos_step + cos_a * sin_step;
                cos_a = nc;
                sin_a = ns;
            }
        }
    }

    if (lane == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    const long obase = (long)block * R;
    if (d0 < R) po[obase + d0] = a0;
    if (d1 < R) po[obase + d1] = a1;
}


template <bool APPEND>
__global__ void gauss_rope_tiled_partial(
    const half* __restrict__ q,
    half* __restrict__ c,
    half* __restrict__ rho,
    const half* __restrict__ c_now,
    const half* __restrict__ rho_now,
    float* __restrict__ pm,
    float* __restrict__ pl,
    float* __restrict__ po,
    int BH,
    int T,
    int Tstride,
    int R,
    int splits,
    int position,
    float scale,
    float rope_log_base,
    int rope_dim
) {
    const int block = blockIdx.x;
    const int bh = block / splits;
    const int s = block - bh * splits;
    if (bh >= BH) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int start = (int)(((long long)T * s) / splits);
    const int end = (int)(((long long)T * (s + 1)) / splits);
    const int d0 = lane * 2;
    const int d1 = d0 + 1;
    const long qbase = (long)bh * R;
    const long hbase = (long)bh * Tstride * R;
    const long rbase = (long)bh * Tstride;

    if (APPEND && s == 0 && warp == 0) {
        const long write_base = hbase + (long)position * R;
        if (d0 < R) c[write_base + d0] = c_now[qbase + d0];
        if (d1 < R) c[write_base + d1] = c_now[qbase + d1];
        if (lane == 0) rho[rbase + position] = rho_now[bh];
    }

    const float q0 = d0 < R ? __half2float(q[qbase + d0]) : 0.0f;
    const float q1 = d1 < R ? __half2float(q[qbase + d1]) : 0.0f;

    const bool rotary = d1 < rope_dim;
    float theta = 0.0f;
    float sin_a = 0.0f, cos_a = 1.0f;
    float sin_step = 0.0f, cos_step = 1.0f;
    const int first_t = start + warp;
    if (rotary) {
        theta = __expf(-rope_log_base * ((float)d0 / (float)rope_dim));
        sincosf((float)first_t * theta, &sin_a, &cos_a);
        sincosf((float)TILE_WARPS * theta, &sin_step, &cos_step);
    }

    __shared__ float scores[TILE_WARPS];
    __shared__ float weights[TILE_WARPS];
    __shared__ float vals[TILE_WARPS][MAX_W];
    __shared__ float alpha_s;
    __shared__ float beta_s;

    float m = -FLT_MAX;
    float l = 0.0f;
    float a0 = 0.0f;
    float a1 = 0.0f;
    int iter = 0;

    for (int tile = start; tile < end; tile += TILE_WARPS, ++iter) {
        const int t = tile + warp;
        const bool valid = t < end;
        float cv0 = 0.0f;
        float cv1 = 0.0f;
        float score = -FLT_MAX;

        if (valid) {
            const bool current = APPEND && (t == position);
            const long base = hbase + (long)t * R;
            cv0 = d0 < R
                ? __half2float(current ? c_now[qbase + d0] : c[base + d0]) : 0.0f;
            cv1 = d1 < R
                ? __half2float(current ? c_now[qbase + d1] : c[base + d1]) : 0.0f;
            float k0 = cv0;
            float k1 = cv1;
            if (rotary) {
                k0 = __half2float(__float2half_rn(cv0 * cos_a - cv1 * sin_a));
                k1 = __half2float(__float2half_rn(cv0 * sin_a + cv1 * cos_a));
            }
            const float dot = warp_sum(q0 * k0 + q1 * k1);
            if (lane == 0) {
                const float rr = __half2float(current ? rho_now[bh] : rho[rbase + t]);
                score = dot * rr * scale;
            }
        }

        if (lane == 0) scores[warp] = score;
        if (d0 < R) vals[warp][d0] = valid ? cv0 : 0.0f;
        if (d1 < R) vals[warp][d1] = valid ? cv1 : 0.0f;
        __syncthreads();

        if (tid == 0) {
            float tm = -FLT_MAX;
            #pragma unroll
            for (int w = 0; w < TILE_WARPS; ++w) tm = fmaxf(tm, scores[w]);
            float tl = 0.0f;
            #pragma unroll
            for (int w = 0; w < TILE_WARPS; ++w) {
                const float sw = scores[w];
                const float ew = sw == -FLT_MAX ? 0.0f : __expf(sw - tm);
                weights[w] = ew;
                tl += ew;
            }
            const float nm = fmaxf(m, tm);
            alpha_s = m == -FLT_MAX ? 0.0f : __expf(m - nm);
            beta_s = tm == -FLT_MAX ? 0.0f : __expf(tm - nm);
            l = l * alpha_s + tl * beta_s;
            m = nm;
        }
        __syncthreads();

        if (warp == 0) {
            float ta0 = 0.0f;
            float ta1 = 0.0f;
            #pragma unroll
            for (int w = 0; w < TILE_WARPS; ++w) {
                const float wt = weights[w];
                if (d0 < R) ta0 += wt * vals[w][d0];
                if (d1 < R) ta1 += wt * vals[w][d1];
            }
            if (d0 < R) a0 = a0 * alpha_s + ta0 * beta_s;
            if (d1 < R) a1 = a1 * alpha_s + ta1 * beta_s;
        }
        __syncthreads();

        if (rotary && tile + TILE_WARPS < end) {
            if (((iter + 1) & 63) == 0) {
                sincosf((float)(tile + TILE_WARPS + warp) * theta, &sin_a, &cos_a);
            } else {
                const float nc = cos_a * cos_step - sin_a * sin_step;
                const float ns = sin_a * cos_step + cos_a * sin_step;
                cos_a = nc;
                sin_a = ns;
            }
        }
    }

    if (tid == 0) {
        pm[block] = m;
        pl[block] = l;
    }
    if (warp == 0) {
        const long obase = (long)block * R;
        if (d0 < R) po[obase + d0] = a0;
        if (d1 < R) po[obase + d1] = a1;
    }
}

// ============================================================
// MERGE SPLITS
// ============================================================

__global__ void merge_partial(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ po,
    half* __restrict__ out,
    int BH,
    int W,
    int splits
) {
    const int bh = blockIdx.x;
    if (bh >= BH) return;

    const int lane = threadIdx.x;
    const int d0 = lane;
    const int d1 = lane + 32;

    const long sbase = (long)bh * splits;

    float gm = -FLT_MAX;

    if (lane == 0) {
        for (int s=0; s<splits; ++s)
            gm = fmaxf(gm, pm[sbase+s]);
    }

    gm = __shfl_sync(FULL_MASK, gm, 0);

    float denom = 0.0f;

    if (lane == 0) {
        for (int s=0; s<splits; ++s) {
            denom +=
                pl[sbase+s]
                * __expf(pm[sbase+s]-gm);
        }
    }

    denom = __shfl_sync(FULL_MASK, denom, 0);

    float n0 = 0.0f;
    float n1 = 0.0f;

    for (int s=0; s<splits; ++s) {
        const float w =
            __expf(pm[sbase+s]-gm);

        const long obase =
            ((long)bh*splits+s)*W;

        if (d0 < W)
            n0 += po[obase+d0]*w;

        if (d1 < W)
            n1 += po[obase+d1]*w;
    }

    const long outbase = (long)bh * W;

    if (d0 < W)
        out[outbase+d0] =
            __float2half_rn(n0/denom);

    if (d1 < W)
        out[outbase+d1] =
            __float2half_rn(n1/denom);
}


// ============================================================
// MERGE SPLITS + OUTPUT PROJECTION (NO GLOBAL O)
//
// One block owns one batch element. Exact split-softmax summaries are merged
// into a shared-memory O vector [H*R], then the ordinary Linear weight
// [D,H*R] is applied before any O tensor is written to global memory.
// This is inference-only; training keeps the saved-O + SDPA path because O is
// required by backward.
// ============================================================

__global__ void merge_project_partial(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ po,
    const half* __restrict__ weight,
    half* __restrict__ y,
    int B,
    int H,
    int R,
    int D,
    int splits,
    int blocks_per_batch
) {
    const int tile = blockIdx.x % blocks_per_batch;
    const int b = blockIdx.x / blocks_per_batch;
    const int tid = threadIdx.x;
    if (b >= B) return;

    extern __shared__ unsigned char smem_raw[];
    float* gm = reinterpret_cast<float*>(smem_raw);
    float* denom = gm + H;
    half* osh = reinterpret_cast<half*>(denom + H);

    // One thread computes the exact global max/denominator for each head.
    for (int h = tid; h < H; h += blockDim.x) {
        const int bh = b * H + h;
        const long sbase = (long)bh * splits;
        float m = -FLT_MAX;
        for (int sp = 0; sp < splits; ++sp)
            m = fmaxf(m, pm[sbase + sp]);
        float d = 0.0f;
        for (int sp = 0; sp < splits; ++sp)
            d += pl[sbase + sp] * __expf(pm[sbase + sp] - m);
        gm[h] = m;
        denom[h] = d;
    }
    __syncthreads();

    const int HR = H * R;
    // Merge each O component once into shared memory. No global O store.
    for (int k = tid; k < HR; k += blockDim.x) {
        const int h = k / R;
        const int r = k - h * R;
        const int bh = b * H + h;
        const long sbase = (long)bh * splits;
        float n = 0.0f;
        for (int sp = 0; sp < splits; ++sp) {
            const float w = __expf(pm[sbase + sp] - gm[h]);
            const long obase = ((long)bh * splits + sp) * R;
            n += po[obase + r] * w;
        }
        osh[k] = __float2half_rn(n / denom[h]);
    }
    __syncthreads();

    // F.linear(O, weight): weight is the native PyTorch [D, H*R] layout.
    // Tile D across blocks so large model widths do not collapse to one block.
    constexpr int OUT_TILE = 128;
    const int d_begin = tile * OUT_TILE;
    const int d_end = min(D, d_begin + OUT_TILE);
    for (int d = d_begin + tid; d < d_end; d += blockDim.x) {
        float acc = 0.0f;
        const long wbase = (long)d * HR;
        for (int k = 0; k < HR; ++k)
            acc += __half2float(osh[k]) * __half2float(weight[wbase + k]);
        y[(long)b * D + d] = __float2half_rn(acc);
    }
}

static inline size_t merge_project_smem_bytes(int H, int R) {
    return (size_t)(2 * H) * sizeof(float) + (size_t)(H * R) * sizeof(half);
}


// Fast standalone no-O reducer port.
//
// This is the exact execution shape that won the standalone T4 generation
// experiment: H=4, R=16, D=128, <=32 context splits. One warp merges each
// BOLT head, producing the 64-wide O only in shared memory. The same block
// immediately applies PyTorch's native Linear layout weight[D,64] and writes Y.
// Other shapes continue through merge_project_partial above.
__global__ void merge_project_h4_r16_d128(
    const float* __restrict__ pm,
    const float* __restrict__ pl,
    const float* __restrict__ po,
    const half* __restrict__ weight,
    half* __restrict__ y,
    int B,
    int splits
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (b >= B) return;

    __shared__ half osh[64];

    if (warp < 4) {
        const int h = warp;
        const int bh = b * 4 + h;
        const long sbase = (long)bh * splits;

        float gm = -FLT_MAX;
        if (lane < splits)
            gm = pm[sbase + lane];
        gm = warp_max(gm);
        gm = __shfl_sync(FULL_MASK, gm, 0);

        float denom = 0.0f;
        float numer = 0.0f;

        for (int sp = 0; sp < splits; ++sp) {
            const float a = __expf(pm[sbase + sp] - gm);
            if (lane == 0)
                denom += a * pl[sbase + sp];
            if (lane < 16)
                numer += a * po[((long)bh * splits + sp) * 16 + lane];
        }

        denom = __shfl_sync(FULL_MASK, denom, 0);
        if (lane < 16)
            osh[h * 16 + lane] = __float2half_rn(numer / denom);
    }

    __syncthreads();

    // Native nn.Linear stores weight as [out_features, in_features] = [128,64].
    if (tid < 128) {
        float acc = 0.0f;
        const long wbase = (long)tid * 64;
        #pragma unroll
        for (int k = 0; k < 64; ++k)
            acc += __half2float(osh[k]) * __half2float(weight[wbase + k]);
        y[(long)b * 128 + tid] = __float2half_rn(acc);
    }
}

static inline bool use_fast_no_o_merge(int H, int R, int D, int splits) {
    return H == 4 && R == 16 && D == 128 && splits >= 1 && splits <= 32;
}

static inline void launch_merge_project(
    const float* pm,
    const float* pl,
    const float* po,
    const half* weight,
    half* y,
    int B,
    int H,
    int R,
    int D,
    int splits,
    cudaStream_t stream
) {
    if (use_fast_no_o_merge(H, R, D, splits)) {
        merge_project_h4_r16_d128<<<B,128,0,stream>>>(
            pm, pl, po, weight, y, B, splits
        );
        return;
    }

    const int project_blocks = (D + 127) / 128;
    merge_project_partial<<<B*project_blocks,128,merge_project_smem_bytes(H,R),stream>>>(
        pm, pl, po, weight, y, B, H, R, D, splits, project_blocks
    );
}



// ============================================================
// HOST HELPERS
// ============================================================

static inline void check_half(
    const torch::Tensor& x,
    const char* name
) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        x.scalar_type()==at::kHalf,
        name,
        " must be FP16"
    );
    TORCH_CHECK(
        x.is_contiguous(),
        name,
        " must be contiguous"
    );
}

static inline void check_same_device(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const char* a_name,
    const char* b_name
) {
    TORCH_CHECK(
        a.device() == b.device(),
        a_name, " and ", b_name, " must be on the same CUDA device"
    );
}

static inline int valid_splits(int64_t splits) {
    TORCH_CHECK(
        splits >= 1 && splits <= 128,
        "splits must be in [1,128]"
    );
    return (int)splits;
}


// ============================================================
// PUBLIC: FUSED PREPROCESS
// ============================================================

std::vector<torch::Tensor> gauss_gate_rho_cuda(
    torch::Tensor u,
    torch::Tensor g,
    double eps
) {
    check_half(u,"u");
    check_half(g,"g");

    TORCH_CHECK(
        u.sizes()==g.sizes(),
        "u and g shape mismatch"
    );
    check_same_device(u, g, "u", "g");
    TORCH_CHECK(
        u.dim()==3,
        "u/g must be [B,H,R]"
    );

    const int B = u.size(0);
    const int H = u.size(1);
    const int R = u.size(2);

    TORCH_CHECK(
        R>0 && R<=64,
        "R must be <=64"
    );

    c10::cuda::CUDAGuard guard(u.device());

    auto c = torch::empty_like(u);

    auto rho = torch::empty(
        {B,H},
        torch::TensorOptions()
            .device(u.device())
            .dtype(torch::kFloat16)
    );

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            u.get_device()
        ).stream();

    gauss_gate_rho_kernel<<<
        B*H,
        32,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(
            u.data_ptr<at::Half>()
        ),
        reinterpret_cast<const half*>(
            g.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            c.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            rho.data_ptr<at::Half>()
        ),
        B*H,
        R,
        (float)eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {c,rho};
}


// ============================================================
// PUBLIC: BASELINE
// mode 0 stream, mode 1 tiled8
// ============================================================

torch::Tensor baseline_decode_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale,
    int64_t mode,
    int64_t splits_i64
) {
    check_half(q,"q");
    check_half(k,"k");
    check_half(v,"v");

    const int B=q.size(0);
    const int H=q.size(1);
    const int D=q.size(2);
    const int T=k.size(2);
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,D]");
    TORCH_CHECK(k.dim()==4, "k must be [B,H,T,D]");
    TORCH_CHECK(v.dim()==4, "v must be [B,H,T,D]");
    TORCH_CHECK(B>0 && H>0 && T>0, "B/H/T must be positive");
    TORCH_CHECK(D>0 && D<=64, "D must be in [1,64]");
    TORCH_CHECK(k.size(0)==B && k.size(1)==H && k.size(3)==D, "k shape mismatch");
    TORCH_CHECK(v.sizes()==k.sizes(), "v shape mismatch");
    check_same_device(q, k, "q", "k");
    check_same_device(q, v, "q", "v");

    c10::cuda::CUDAGuard guard(q.device());

    auto fopts =
        torch::TensorOptions()
        .device(q.device())
        .dtype(torch::kFloat32);

    auto pm=torch::empty({BH,splits},fopts);
    auto pl=torch::empty({BH,splits},fopts);
    auto po=torch::empty({BH,splits,D},fopts);
    auto out=torch::empty_like(q);

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            q.get_device()
        ).stream();

    if (mode==0) {
        baseline_stream_partial<<<
            BH*splits,
            32,
            0,
            stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,D,splits,(float)scale
        );
    } else if (mode==1) {
        baseline_tiled_partial<<<
            BH*splits,
            TILE_THREADS,
            0,
            stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,D,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false, "unknown mode");
    }

    merge_partial<<<
        BH,
        32,
        0,
        stream
    >>>(
        pm.data_ptr<float>(),
        pl.data_ptr<float>(),
        po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        BH,D,splits
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


// ============================================================
// PUBLIC: GAUSS
// ============================================================

torch::Tensor gauss_decode_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    double scale,
    int64_t mode,
    int64_t splits_i64
) {
    check_half(q,"q");
    check_half(c,"c");
    check_half(rho,"rho");

    const int B=q.size(0);
    const int H=q.size(1);
    const int R=q.size(2);
    const int T=c.size(2);
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c.dim()==4, "c must be [B,H,T,R]");
    TORCH_CHECK(rho.dim()==3, "rho must be [B,H,T]");
    TORCH_CHECK(B>0 && H>0 && T>0, "B/H/T must be positive");
    TORCH_CHECK(R>0 && R<=64, "R must be in [1,64]");
    TORCH_CHECK(c.size(0)==B && c.size(1)==H && c.size(3)==R, "c shape mismatch");
    TORCH_CHECK(rho.size(0)==B && rho.size(1)==H && rho.size(2)==T, "rho shape mismatch");
    check_same_device(q, c, "q", "c");
    check_same_device(q, rho, "q", "rho");

    c10::cuda::CUDAGuard guard(q.device());

    auto fopts =
        torch::TensorOptions()
        .device(q.device())
        .dtype(torch::kFloat32);

    auto pm=torch::empty({BH,splits},fopts);
    auto pl=torch::empty({BH,splits},fopts);
    auto po=torch::empty({BH,splits,R},fopts);
    auto out=torch::empty_like(q);

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            q.get_device()
        ).stream();

    if (mode==0) {
        gauss_stream_partial<<<
            BH*splits,
            32,
            0,
            stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,R,splits,(float)scale
        );
    } else if (mode==1) {
        gauss_tiled_partial<<<
            BH*splits,
            TILE_THREADS,
            0,
            stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,R,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false, "unknown mode");
    }

    merge_partial<<<
        BH,
        32,
        0,
        stream
    >>>(
        pm.data_ptr<float>(),
        pl.data_ptr<float>(),
        po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        BH,R,splits
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


// ============================================================
// ZERO-ALLOCATION UNPACK / PREPROCESS KERNELS
// ============================================================

// ============================================================
// FULL-SEQUENCE BOLT STAGE-1 TRAINING
// packed qcg [B,T,3*H*R] -> q,c,rho (+ gate saved for backward)
// One warp per token/head. U/G are never materialized as public tensors.
// ============================================================

__global__ void gauss_stage1_forward_kernel(
    const half* __restrict__ qcg,
    half* __restrict__ q_out,
    half* __restrict__ c_out,
    float* __restrict__ rho_out,
    half* __restrict__ gate_out,
    int B,
    int T,
    int H,
    int R,
    float eps
) {
    const int bth = blockIdx.x;
    const int lane = threadIdx.x;
    const int BTH = B * T * H;
    if (bth >= BTH) return;

    const int h = bth % H;
    const int bt = bth / H;
    const int t = bt % T;
    const int b = bt / T;
    const int HR = H * R;
    const int d0 = lane;
    const int d1 = lane + 32;
    const long row = ((long)b * T + t) * (3 * HR);
    const long head_off = (long)h * R;
    const long out_base = (((long)b * H + h) * T + t) * R;

    float ss = 0.0f;

    if (d0 < R) {
        const float qf = __half2float(qcg[row + head_off + d0]);
        const float uf = __half2float(qcg[row + HR + head_off + d0]);
        const float gf = __half2float(qcg[row + 2 * HR + head_off + d0]);
        const float a = 1.0f + tanhf(gf);
        const half ch = __float2half_rn(uf * a);
        q_out[out_base + d0] = __float2half_rn(qf);
        c_out[out_base + d0] = ch;
        gate_out[out_base + d0] = __float2half_rn(a);
        const float cr = __half2float(ch);
        ss += cr * cr;
    }
    if (d1 < R) {
        const float qf = __half2float(qcg[row + head_off + d1]);
        const float uf = __half2float(qcg[row + HR + head_off + d1]);
        const float gf = __half2float(qcg[row + 2 * HR + head_off + d1]);
        const float a = 1.0f + tanhf(gf);
        const half ch = __float2half_rn(uf * a);
        q_out[out_base + d1] = __float2half_rn(qf);
        c_out[out_base + d1] = ch;
        gate_out[out_base + d1] = __float2half_rn(a);
        const float cr = __half2float(ch);
        ss += cr * cr;
    }

    ss = warp_sum(ss);
    if (lane == 0) {
        rho_out[((long)b * H + h) * T + t] =
            rsqrtf(ss / (float)R + eps);
    }
}

__global__ void gauss_stage1_backward_kernel(
    const half* __restrict__ dq,
    const half* __restrict__ dc,
    const float* __restrict__ drho,
    const half* __restrict__ c,
    const float* __restrict__ rho,
    const half* __restrict__ gate,
    half* __restrict__ dqcg,
    int B,
    int T,
    int H,
    int R
) {
    const int bth = blockIdx.x;
    const int lane = threadIdx.x;
    const int BTH = B * T * H;
    if (bth >= BTH) return;

    const int h = bth % H;
    const int bt = bth / H;
    const int t = bt % T;
    const int b = bt / T;
    const int HR = H * R;
    const long row = ((long)b * T + t) * (3 * HR);
    const long head_off = (long)h * R;
    const long in_base = (((long)b * H + h) * T + t) * R;
    const long rho_idx = ((long)b * H + h) * T + t;
    const float rv = rho[rho_idx];
    const float rg = drho[rho_idx];

    const int d0 = lane;
    const int d1 = lane + 32;

    if (d0 < R) {
        const long idx = in_base + d0;
        const float cv = __half2float(c[idx]);
        const float a = __half2float(gate[idx]);
        const float dct = __half2float(dc[idx])
            - rg * (rv * rv * rv) * cv / (float)R;
        dqcg[row + head_off + d0] = dq[idx];
        dqcg[row + HR + head_off + d0] = __float2half_rn(dct * a);
        dqcg[row + 2 * HR + head_off + d0] = __float2half_rn(dct * cv * (2.0f - a));
    }
    if (d1 < R) {
        const long idx = in_base + d1;
        const float cv = __half2float(c[idx]);
        const float a = __half2float(gate[idx]);
        const float dct = __half2float(dc[idx])
            - rg * (rv * rv * rv) * cv / (float)R;
        dqcg[row + head_off + d1] = dq[idx];
        dqcg[row + HR + head_off + d1] = __float2half_rn(dct * a);
        dqcg[row + 2 * HR + head_off + d1] = __float2half_rn(dct * cv * (2.0f - a));
    }
}

// qcg: [B, 3*H*R], row-major FP16.
// Writes contiguous q_out [B,H,R], c_out [B,H,R], rho_out [B,H].
__global__ void gauss_unpack_gate_rho_out_kernel(
    const half* __restrict__ qcg,
    half* __restrict__ q_out,
    half* __restrict__ c_out,
    half* __restrict__ rho_out,
    int B,
    int H,
    int R,
    float eps
) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;

    const int BH = B * H;
    if (bh >= BH) return;

    const int b = bh / H;
    const int h = bh - b * H;
    const int HR = H * R;

    const int d0 = lane;
    const int d1 = lane + 32;

    const long row = (long)b * (3 * HR);
    const long head_off = (long)h * R;
    const long out_base = (long)bh * R;

    float ss = 0.0f;

    if (d0 < R) {
        const float qf = __half2float(
            qcg[row + head_off + d0]
        );
        const float uf = __half2float(
            qcg[row + HR + head_off + d0]
        );
        const float gf = __half2float(
            qcg[row + 2*HR + head_off + d0]
        );

        const float cf =
            uf * (1.0f + tanhf(gf));

        q_out[out_base + d0] =
            __float2half_rn(qf);

        const half ch = __float2half_rn(cf);
        c_out[out_base + d0] = ch;
        const float cr = __half2float(ch);
        ss += cr * cr;
    }

    if (d1 < R) {
        const float qf = __half2float(
            qcg[row + head_off + d1]
        );
        const float uf = __half2float(
            qcg[row + HR + head_off + d1]
        );
        const float gf = __half2float(
            qcg[row + 2*HR + head_off + d1]
        );

        const float cf =
            uf * (1.0f + tanhf(gf));

        q_out[out_base + d1] =
            __float2half_rn(qf);

        const half ch = __float2half_rn(cf);
        c_out[out_base + d1] = ch;
        const float cr = __half2float(ch);
        ss += cr * cr;
    }

    ss = warp_sum(ss);

    if (lane == 0) {
        rho_out[bh] =
            __float2half_rn(
                rsqrtf(
                    ss / (float)R + eps
                )
            );
    }
}


// qkv: [B, 3*H*D].
// Writes contiguous q/k/v current-token buffers.
__global__ void baseline_unpack_qkv_out_kernel(
    const half* __restrict__ qkv,
    half* __restrict__ q_out,
    half* __restrict__ k_now,
    half* __restrict__ v_now,
    int B,
    int H,
    int D
) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;

    const int BH = B * H;
    if (bh >= BH) return;

    const int b = bh / H;
    const int h = bh - b * H;
    const int HD = H * D;

    const int d0 = lane;
    const int d1 = lane + 32;

    const long row = (long)b * (3 * HD);
    const long head_off = (long)h * D;
    const long out_base = (long)bh * D;

    if (d0 < D) {
        q_out[out_base+d0] =
            qkv[row + head_off + d0];

        k_now[out_base+d0] =
            qkv[row + HD + head_off + d0];

        v_now[out_base+d0] =
            qkv[row + 2*HD + head_off + d0];
    }

    if (d1 < D) {
        q_out[out_base+d1] =
            qkv[row + head_off + d1];

        k_now[out_base+d1] =
            qkv[row + HD + head_off + d1];

        v_now[out_base+d1] =
            qkv[row + 2*HD + head_off + d1];
    }
}


// ============================================================
// ZERO-ALLOCATION HOST API
// ============================================================

static inline void check_float(
    const torch::Tensor& x,
    const char* name
) {
    TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(
        x.scalar_type()==at::kFloat,
        name,
        " must be FP32"
    );
    TORCH_CHECK(
        x.is_contiguous(),
        name,
        " must be contiguous"
    );
}


std::vector<torch::Tensor> gauss_stage1_forward_cuda(
    torch::Tensor qcg,
    int64_t heads,
    int64_t latent,
    double eps
) {
    check_half(qcg, "qcg");
    TORCH_CHECK(qcg.dim() == 3, "qcg must be [B,T,3*H*R]");
    const int B = qcg.size(0);
    const int T = qcg.size(1);
    const int H = (int)heads;
    const int R = (int)latent;
    TORCH_CHECK(R > 0 && R <= 64, "latent must be in [1,64]");
    TORCH_CHECK(qcg.size(2) == 3 * H * R, "qcg shape mismatch");

    auto q = torch::empty({B,H,T,R}, qcg.options());
    auto c = torch::empty_like(q);
    auto rho = torch::empty({B,H,T}, qcg.options().dtype(at::kFloat));
    auto gate = torch::empty_like(q);

    c10::cuda::CUDAGuard guard(qcg.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(qcg.get_device()).stream();
    gauss_stage1_forward_kernel<<<B*T*H,32,0,stream>>>(
        reinterpret_cast<const half*>(qcg.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()),
        rho.data_ptr<float>(),
        reinterpret_cast<half*>(gate.data_ptr<at::Half>()),
        B,T,H,R,(float)eps
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q,c,rho,gate};
}

torch::Tensor gauss_stage1_backward_cuda(
    torch::Tensor dq,
    torch::Tensor dc,
    torch::Tensor drho,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor gate,
    int64_t heads,
    int64_t latent
) {
    check_half(dq, "dq");
    check_half(dc, "dc");
    check_float(drho, "drho");
    check_half(c, "c");
    check_float(rho, "rho");
    check_half(gate, "gate");
    TORCH_CHECK(c.dim() == 4, "c must be [B,H,T,R]");
    TORCH_CHECK(dq.sizes() == c.sizes() && dc.sizes() == c.sizes(), "dq/dc shape mismatch");
    TORCH_CHECK(gate.sizes() == c.sizes(), "gate shape mismatch");
    const int B = c.size(0);
    const int H = (int)heads;
    const int T = c.size(2);
    const int R = (int)latent;
    TORCH_CHECK(c.size(1) == H && c.size(3) == R, "c head/latent mismatch");
    TORCH_CHECK(rho.dim() == 3 && rho.size(0) == B && rho.size(1) == H && rho.size(2) == T, "rho shape mismatch");
    TORCH_CHECK(drho.sizes() == rho.sizes(), "drho shape mismatch");

    auto dqcg = torch::empty({B,T,3*H*R}, c.options());
    c10::cuda::CUDAGuard guard(c.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(c.get_device()).stream();
    gauss_stage1_backward_kernel<<<B*T*H,32,0,stream>>>(
        reinterpret_cast<const half*>(dq.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(dc.data_ptr<at::Half>()),
        drho.data_ptr<float>(),
        reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
        rho.data_ptr<float>(),
        reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dqcg.data_ptr<at::Half>()),
        B,T,H,R
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dqcg;
}

void gauss_unpack_gate_rho_out_cuda(
    torch::Tensor qcg,
    torch::Tensor q_out,
    torch::Tensor c_out,
    torch::Tensor rho_out,
    double eps
) {
    check_half(qcg, "qcg");
    check_half(q_out, "q_out");
    check_half(c_out, "c_out");
    check_half(rho_out, "rho_out");

    TORCH_CHECK(qcg.dim()==2, "qcg must be [B,3HR]");
    TORCH_CHECK(q_out.dim()==3, "q_out must be [B,H,R]");
    TORCH_CHECK(c_out.sizes()==q_out.sizes(), "c_out shape mismatch");
    TORCH_CHECK(rho_out.dim()==2, "rho_out must be [B,H]");
    check_same_device(qcg, q_out, "qcg", "q_out");
    check_same_device(qcg, c_out, "qcg", "c_out");
    check_same_device(qcg, rho_out, "qcg", "rho_out");

    const int B = q_out.size(0);
    const int H = q_out.size(1);
    const int R = q_out.size(2);

    TORCH_CHECK(
        qcg.size(0)==B
        && qcg.size(1)==3*H*R,
        "qcg shape mismatch"
    );

    TORCH_CHECK(
        rho_out.size(0)==B
        && rho_out.size(1)==H,
        "rho_out shape mismatch"
    );

    c10::cuda::CUDAGuard guard(qcg.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            qcg.get_device()
        ).stream();

    gauss_unpack_gate_rho_out_kernel<<<
        B*H,
        32,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(
            qcg.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            q_out.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            c_out.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            rho_out.data_ptr<at::Half>()
        ),
        B,H,R,(float)eps
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void baseline_unpack_qkv_out_cuda(
    torch::Tensor qkv,
    torch::Tensor q_out,
    torch::Tensor k_now,
    torch::Tensor v_now
) {
    check_half(qkv, "qkv");
    check_half(q_out, "q_out");
    check_half(k_now, "k_now");
    check_half(v_now, "v_now");

    TORCH_CHECK(qkv.dim()==2, "qkv must be [B,3HD]");
    TORCH_CHECK(q_out.dim()==3, "q_out must be [B,H,D]");
    TORCH_CHECK(k_now.sizes()==q_out.sizes(), "k_now shape mismatch");
    TORCH_CHECK(v_now.sizes()==q_out.sizes(), "v_now shape mismatch");
    check_same_device(qkv, q_out, "qkv", "q_out");
    check_same_device(qkv, k_now, "qkv", "k_now");
    check_same_device(qkv, v_now, "qkv", "v_now");

    const int B = q_out.size(0);
    const int H = q_out.size(1);
    const int D = q_out.size(2);

    TORCH_CHECK(
        qkv.size(0)==B
        && qkv.size(1)==3*H*D,
        "qkv shape mismatch"
    );

    c10::cuda::CUDAGuard guard(qkv.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            qkv.get_device()
        ).stream();

    baseline_unpack_qkv_out_kernel<<<
        B*H,
        32,
        0,
        stream
    >>>(
        reinterpret_cast<const half*>(
            qkv.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            q_out.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            k_now.data_ptr<at::Half>()
        ),
        reinterpret_cast<half*>(
            v_now.data_ptr<at::Half>()
        ),
        B,H,D
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void baseline_decode_out_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64
) {
    check_half(q,"q");
    check_half(k,"k");
    check_half(v,"v");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    const int B=q.size(0);
    const int H=q.size(1);
    const int D=q.size(2);
    const int T=k.size(2);
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,D]");
    TORCH_CHECK(k.dim()==4 && v.dim()==4, "k/v must be [B,H,T,D]");
    TORCH_CHECK(B>0 && H>0 && T>0 && D>0 && D<=64, "invalid baseline dimensions");
    TORCH_CHECK(k.size(0)==B && k.size(1)==H && k.size(3)==D, "k shape mismatch");
    TORCH_CHECK(v.sizes()==k.sizes(), "v shape mismatch");
    check_same_device(q,k,"q","k");
    check_same_device(q,v,"q","v");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(
        po.size(0)==BH
        && po.size(1)==splits
        && po.size(2)==D,
        "po shape mismatch"
    );

    c10::cuda::CUDAGuard guard(q.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            q.get_device()
        ).stream();

    if (mode==0) {
        baseline_stream_partial<<<
            BH*splits,32,0,stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,D,splits,(float)scale
        );
    } else if (mode==1) {
        baseline_tiled_partial<<<
            BH*splits,TILE_THREADS,0,stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,D,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }

    merge_partial<<<
        BH,32,0,stream
    >>>(
        pm.data_ptr<float>(),
        pl.data_ptr<float>(),
        po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        BH,D,splits
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void gauss_decode_out_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64
) {
    check_half(q,"q");
    check_half(c,"c");
    check_half(rho,"rho");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    const int B=q.size(0);
    const int H=q.size(1);
    const int R=q.size(2);
    const int T=c.size(2);
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c.dim()==4, "c must be [B,H,T,R]");
    TORCH_CHECK(rho.dim()==3, "rho must be [B,H,T]");
    TORCH_CHECK(B>0 && H>0 && T>0 && R>0 && R<=64, "invalid Gauss dimensions");
    TORCH_CHECK(c.size(0)==B && c.size(1)==H && c.size(3)==R, "c shape mismatch");
    TORCH_CHECK(rho.size(0)==B && rho.size(1)==H && rho.size(2)==T, "rho shape mismatch");
    check_same_device(q,c,"q","c");
    check_same_device(q,rho,"q","rho");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(
        po.size(0)==BH
        && po.size(1)==splits
        && po.size(2)==R,
        "po shape mismatch"
    );

    c10::cuda::CUDAGuard guard(q.device());

    cudaStream_t stream =
        c10::cuda::getCurrentCUDAStream(
            q.get_device()
        ).stream();

    if (mode==0) {
        gauss_stream_partial<<<
            BH*splits,32,0,stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,R,splits,(float)scale
        );
    } else if (mode==1) {
        gauss_tiled_partial<<<
            BH*splits,TILE_THREADS,0,stream
        >>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),
            pl.data_ptr<float>(),
            po.data_ptr<float>(),
            BH,T,T,R,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }

    merge_partial<<<
        BH,32,0,stream
    >>>(
        pm.data_ptr<float>(),
        pl.data_ptr<float>(),
        po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        BH,R,splits
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}



// ============================================================
// FUSED CURRENT-TOKEN CACHE APPEND
// One CUDA launch writes both members of each cache pair.  The append+decode
// host APIs below enqueue this write and the attention kernels on the same
// stream, removing Python/PyTorch cache-copy dispatch from the token loop.
// ============================================================

__global__ void baseline_append_cache_kernel(
    const half* __restrict__ k_now,
    const half* __restrict__ v_now,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int BH,
    int capacity,
    int D,
    int position
) {
    const int bh = blockIdx.x;
    const int d = threadIdx.x;
    if (bh >= BH || d >= D) return;

    const long src = (long)bh * D + d;
    const long dst = ((long)bh * capacity + position) * D + d;
    k_cache[dst] = k_now[src];
    v_cache[dst] = v_now[src];
}

__global__ void gauss_append_cache_kernel(
    const half* __restrict__ c_now,
    const half* __restrict__ rho_now,
    half* __restrict__ c_cache,
    half* __restrict__ rho_cache,
    int BH,
    int capacity,
    int R,
    int position
) {
    const int bh = blockIdx.x;
    const int d = threadIdx.x;
    if (bh >= BH) return;

    if (d < R) {
        const long src = (long)bh * R + d;
        const long dst = ((long)bh * capacity + position) * R + d;
        c_cache[dst] = c_now[src];
    }
    if (d == 0) {
        rho_cache[(long)bh * capacity + position] = rho_now[bh];
    }
}

// ============================================================
// FIXED-CAPACITY CACHE DECODE
// Logical T may be smaller than the physical cache capacity.  This lets the
// Python runtime append tokens in-place without torch.cat or cache copies.
// ============================================================

void baseline_decode_out_used_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t used_i64
) {
    check_half(q,"q");
    check_half(k,"k");
    check_half(v,"v");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    const int B=q.size(0);
    const int H=q.size(1);
    const int D=q.size(2);
    const int Tstride=k.size(2);
    const int T=(int)used_i64;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,D]");
    TORCH_CHECK(k.dim()==4 && v.dim()==4, "k/v must be [B,H,capacity,D]");
    TORCH_CHECK(T>=1 && T<=Tstride, "used length must be in [1, cache capacity]");
    TORCH_CHECK(D>0 && D<=64, "D must be in [1,64]");
    TORCH_CHECK(k.size(0)==B && k.size(1)==H && k.size(3)==D, "k shape mismatch");
    TORCH_CHECK(v.sizes()==k.sizes(), "v shape mismatch");
    check_same_device(q,k,"q","k");
    check_same_device(q,v,"q","v");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==D, "po shape mismatch");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    if (mode==0) {
        baseline_stream_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,Tstride,D,splits,(float)scale
        );
    } else if (mode==1) {
        baseline_tiled_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,Tstride,D,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }
    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,D,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void gauss_decode_out_used_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t used_i64
) {
    check_half(q,"q");
    check_half(c,"c");
    check_half(rho,"rho");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    const int B=q.size(0);
    const int H=q.size(1);
    const int R=q.size(2);
    const int Tstride=c.size(2);
    const int T=(int)used_i64;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c.dim()==4 && rho.dim()==3, "c/rho cache rank mismatch");
    TORCH_CHECK(T>=1 && T<=Tstride, "used length must be in [1, cache capacity]");
    TORCH_CHECK(R>0 && R<=64, "R must be in [1,64]");
    TORCH_CHECK(c.size(0)==B && c.size(1)==H && c.size(3)==R, "c shape mismatch");
    TORCH_CHECK(rho.size(0)==B && rho.size(1)==H && rho.size(2)==Tstride, "rho shape mismatch");
    check_same_device(q,c,"q","c");
    check_same_device(q,rho,"q","rho");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    if (mode==0) {
        gauss_stream_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,Tstride,R,splits,(float)scale
        );
    } else if (mode==1) {
        gauss_tiled_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,Tstride,R,splits,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }
    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,R,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================
// PUBLIC: DIRECT C+rho DECODE + W_O (NO GLOBAL O)
// ============================================================

static torch::Tensor make_project_output(torch::Tensor q, torch::Tensor weight) {
    TORCH_CHECK(weight.dim()==2, "output projection weight must be [D,H*R]");
    return torch::empty({q.size(0), weight.size(0)}, q.options());
}

static void check_project_weight(torch::Tensor q, torch::Tensor weight, int H, int R) {
    check_half(weight, "weight");
    TORCH_CHECK(weight.dim()==2, "output projection weight must be [D,H*R]");
    TORCH_CHECK(weight.size(1)==(long)H*R, "output projection input width mismatch");
    TORCH_CHECK(weight.size(0)>0, "output projection output width must be positive");
    TORCH_CHECK(weight.is_contiguous(), "output projection weight must be contiguous");
    check_same_device(q,weight,"q","weight");
}

torch::Tensor gauss_decode_project_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits_i64, int64_t used_i64
) {
    check_half(q,"q"); check_half(c,"c"); check_half(rho,"rho");
    check_float(pm,"pm"); check_float(pl,"pl"); check_float(po,"po");
    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c.dim()==4 && rho.dim()==3, "c/rho cache rank mismatch");
    const int B=(int)q.size(0), H=(int)q.size(1), R=(int)q.size(2);
    const int capacity=(int)c.size(2), T=(int)used_i64, BH=B*H;
    const int splits=valid_splits(splits_i64);
    TORCH_CHECK(B>0 && H>0 && R>0 && R<=64, "invalid Gauss dimensions");
    TORCH_CHECK(T>=1 && T<=capacity, "used length must be in [1, cache capacity]");
    TORCH_CHECK(c.size(0)==B && c.size(1)==H && c.size(3)==R, "c shape mismatch");
    TORCH_CHECK(rho.size(0)==B && rho.size(1)==H && rho.size(2)==capacity, "rho shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_same_device(q,c,"q","c"); check_same_device(q,rho,"q","rho");
    check_same_device(q,pm,"q","pm"); check_same_device(q,pl,"q","pl"); check_same_device(q,po,"q","po");
    check_project_weight(q,weight,H,R);
    auto y=make_project_output(q,weight);
    const int D=(int)weight.size(0);
    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if(mode==0){
        gauss_stream_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,(float)scale);
    } else if(mode==1){
        gauss_tiled_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,(float)scale);
    } else TORCH_CHECK(false,"unknown mode");
    launch_merge_project(
        pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        B, H, R, D, splits, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor gauss_decode_project_out_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits_i64
) {
    return gauss_decode_project_out_used_cuda(
        q,c,rho,pm,pl,po,weight,scale,mode,splits_i64,c.size(2));
}

torch::Tensor gauss_decode_append_project_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits_i64, int64_t position_i64
) {
    check_half(q,"q"); check_half(c_now,"c_now"); check_half(rho_now,"rho_now");
    check_half(c_cache,"c_cache"); check_half(rho_cache,"rho_cache");
    check_float(pm,"pm"); check_float(pl,"pl"); check_float(po,"po");
    TORCH_CHECK(q.dim()==3 && c_now.dim()==3 && rho_now.dim()==2, "current projected state rank mismatch");
    TORCH_CHECK(c_cache.dim()==4 && rho_cache.dim()==3, "cache rank mismatch");
    const int B=(int)q.size(0), H=(int)q.size(1), R=(int)q.size(2);
    const int capacity=(int)c_cache.size(2), position=(int)position_i64, T=position+1, BH=B*H;
    const int splits=valid_splits(splits_i64);
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    TORCH_CHECK(c_now.sizes()==q.sizes(), "current C shape mismatch");
    TORCH_CHECK(rho_now.size(0)==B && rho_now.size(1)==H, "current rho shape mismatch");
    TORCH_CHECK(c_cache.size(0)==B && c_cache.size(1)==H && c_cache.size(3)==R, "C cache shape mismatch");
    TORCH_CHECK(rho_cache.size(0)==B && rho_cache.size(1)==H && rho_cache.size(2)==capacity, "rho cache shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_same_device(q,c_now,"q","c_now"); check_same_device(q,rho_now,"q","rho_now");
    check_same_device(q,c_cache,"q","c_cache"); check_same_device(q,rho_cache,"q","rho_cache");
    check_project_weight(q,weight,H,R);
    auto y=make_project_output(q,weight); const int D=(int)weight.size(0);
    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if(mode==0){
        gauss_stream_append_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale);
    } else if(mode==1){
        gauss_tiled_append_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale);
    } else TORCH_CHECK(false,"unknown mode");
    launch_merge_project(
        pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        B, H, R, D, splits, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ============================================================
// PUBLIC: CACHE APPEND ONLY
// Useful when the mathematical decode must stay in PyTorch (for example
// Gauss + RoPE), while still avoiding two tiny Python-level copy_ launches.
// ============================================================

void baseline_append_cache_cuda(
    torch::Tensor k_now,
    torch::Tensor v_now,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t position_i64
) {
    check_half(k_now, "k_now");
    check_half(v_now, "v_now");
    check_half(k_cache, "k_cache");
    check_half(v_cache, "v_cache");
    TORCH_CHECK(k_now.dim()==3 && v_now.dim()==3, "k_now/v_now must be [B,H,D]");
    TORCH_CHECK(k_cache.dim()==4 && v_cache.dim()==4, "k_cache/v_cache must be [B,H,capacity,D]");
    TORCH_CHECK(k_now.sizes()==v_now.sizes(), "k_now/v_now shape mismatch");
    TORCH_CHECK(k_cache.sizes()==v_cache.sizes(), "k_cache/v_cache shape mismatch");

    const int B=(int)k_now.size(0);
    const int H=(int)k_now.size(1);
    const int D=(int)k_now.size(2);
    const int capacity=(int)k_cache.size(2);
    const int position=(int)position_i64;
    const int BH=B*H;

    TORCH_CHECK(D>0 && D<=64, "native append currently requires D in [1,64]");
    TORCH_CHECK(k_cache.size(0)==B && k_cache.size(1)==H && k_cache.size(3)==D, "cache shape mismatch");
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    check_same_device(k_now,v_now,"k_now","v_now");
    check_same_device(k_now,k_cache,"k_now","k_cache");
    check_same_device(k_now,v_cache,"k_now","v_cache");

    c10::cuda::CUDAGuard guard(k_now.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(k_now.get_device()).stream();
    baseline_append_cache_kernel<<<BH,64,0,stream>>>(
        reinterpret_cast<const half*>(k_now.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_now.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        BH,capacity,D,position
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gauss_append_cache_cuda(
    torch::Tensor c_now,
    torch::Tensor rho_now,
    torch::Tensor c_cache,
    torch::Tensor rho_cache,
    int64_t position_i64
) {
    check_half(c_now, "c_now");
    check_half(rho_now, "rho_now");
    check_half(c_cache, "c_cache");
    check_half(rho_cache, "rho_cache");
    TORCH_CHECK(c_now.dim()==3, "c_now must be [B,H,R]");
    TORCH_CHECK(rho_now.dim()==2, "rho_now must be [B,H]");
    TORCH_CHECK(c_cache.dim()==4, "c_cache must be [B,H,capacity,R]");
    TORCH_CHECK(rho_cache.dim()==3, "rho_cache must be [B,H,capacity]");

    const int B=(int)c_now.size(0);
    const int H=(int)c_now.size(1);
    const int R=(int)c_now.size(2);
    const int capacity=(int)c_cache.size(2);
    const int position=(int)position_i64;
    const int BH=B*H;

    TORCH_CHECK(R>0 && R<=64, "R must be in [1,64]");
    TORCH_CHECK(rho_now.size(0)==B && rho_now.size(1)==H, "rho_now shape mismatch");
    TORCH_CHECK(c_cache.size(0)==B && c_cache.size(1)==H && c_cache.size(3)==R, "c_cache shape mismatch");
    TORCH_CHECK(rho_cache.size(0)==B && rho_cache.size(1)==H && rho_cache.size(2)==capacity, "rho_cache shape mismatch");
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    check_same_device(c_now,rho_now,"c_now","rho_now");
    check_same_device(c_now,c_cache,"c_now","c_cache");
    check_same_device(c_now,rho_cache,"c_now","rho_cache");

    c10::cuda::CUDAGuard guard(c_now.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(c_now.get_device()).stream();
    gauss_append_cache_kernel<<<BH,64,0,stream>>>(
        reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
        BH,capacity,R,position
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// ============================================================
// PUBLIC: APPEND + DECODE IN ONE NATIVE CALL
// The current-token cache write and decode kernels are enqueued on the same
// CUDA stream.  This removes Python cache.append()/copy_ from the hot loop.
// ============================================================

void baseline_decode_append_out_cuda(
    torch::Tensor q,
    torch::Tensor k_now,
    torch::Tensor v_now,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t position_i64
) {
    check_half(q,"q");
    check_half(k_now,"k_now");
    check_half(v_now,"v_now");
    check_half(k_cache,"k_cache");
    check_half(v_cache,"v_cache");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    TORCH_CHECK(q.dim()==3, "q must be [B,H,D]");
    TORCH_CHECK(k_now.dim()==3 && v_now.dim()==3, "k_now/v_now must be [B,H,D]");
    TORCH_CHECK(k_cache.dim()==4 && v_cache.dim()==4, "cache must be [B,H,capacity,D]");

    const int B=(int)q.size(0);
    const int H=(int)q.size(1);
    const int D=(int)q.size(2);
    const int capacity=(int)k_cache.size(2);
    const int position=(int)position_i64;
    const int T=position+1;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(B>0 && H>0 && D>0 && D<=64, "invalid baseline dimensions");
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    TORCH_CHECK(k_now.sizes()==q.sizes() && v_now.sizes()==q.sizes(), "current K/V shape mismatch");
    TORCH_CHECK(k_cache.size(0)==B && k_cache.size(1)==H && k_cache.size(3)==D, "K cache shape mismatch");
    TORCH_CHECK(v_cache.sizes()==k_cache.sizes(), "V cache shape mismatch");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==D, "po shape mismatch");
    check_same_device(q,k_now,"q","k_now");
    check_same_device(q,v_now,"q","v_now");
    check_same_device(q,k_cache,"q","k_cache");
    check_same_device(q,v_cache,"q","v_cache");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    if (mode==0) {
        baseline_stream_append_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,D,splits,position,(float)scale
        );
    } else if (mode==1) {
        baseline_tiled_append_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,D,splits,position,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }

    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,D,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gauss_decode_append_out_cuda(
    torch::Tensor q,
    torch::Tensor c_now,
    torch::Tensor rho_now,
    torch::Tensor c_cache,
    torch::Tensor rho_cache,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t position_i64
) {
    check_half(q,"q");
    check_half(c_now,"c_now");
    check_half(rho_now,"rho_now");
    check_half(c_cache,"c_cache");
    check_half(rho_cache,"rho_cache");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c_now.dim()==3, "c_now must be [B,H,R]");
    TORCH_CHECK(rho_now.dim()==2, "rho_now must be [B,H]");
    TORCH_CHECK(c_cache.dim()==4, "c_cache must be [B,H,capacity,R]");
    TORCH_CHECK(rho_cache.dim()==3, "rho_cache must be [B,H,capacity]");

    const int B=(int)q.size(0);
    const int H=(int)q.size(1);
    const int R=(int)q.size(2);
    const int capacity=(int)c_cache.size(2);
    const int position=(int)position_i64;
    const int T=position+1;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);

    TORCH_CHECK(B>0 && H>0 && R>0 && R<=64, "invalid Gauss dimensions");
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    TORCH_CHECK(c_now.sizes()==q.sizes(), "current C shape mismatch");
    TORCH_CHECK(rho_now.size(0)==B && rho_now.size(1)==H, "current rho shape mismatch");
    TORCH_CHECK(c_cache.size(0)==B && c_cache.size(1)==H && c_cache.size(3)==R, "C cache shape mismatch");
    TORCH_CHECK(rho_cache.size(0)==B && rho_cache.size(1)==H && rho_cache.size(2)==capacity, "rho cache shape mismatch");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_same_device(q,c_now,"q","c_now");
    check_same_device(q,rho_now,"q","rho_now");
    check_same_device(q,c_cache,"q","c_cache");
    check_same_device(q,rho_cache,"q","rho_cache");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

    if (mode==0) {
        gauss_stream_append_partial<<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale
        );
    } else if (mode==1) {
        gauss_tiled_append_partial<<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }

    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,R,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}




// ============================================================
// GAUSS + RoPE ZERO-ALLOCATION DECODE
// q is already rotated at the query position. Keys are rotated on-the-fly;
// raw C remains both the compact cache and the value tensor.
// ============================================================

void gauss_rope_decode_out_used_cuda(
    torch::Tensor q,
    torch::Tensor c,
    torch::Tensor rho,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t used_i64,
    double rope_base,
    int64_t rope_dim_i64
) {
    check_half(q,"q");
    check_half(c,"c");
    check_half(rho,"rho");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c.dim()==4 && rho.dim()==3, "c/rho cache rank mismatch");
    const int B=(int)q.size(0);
    const int H=(int)q.size(1);
    const int R=(int)q.size(2);
    const int capacity=(int)c.size(2);
    const int T=(int)used_i64;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);
    const int rope_dim=(int)rope_dim_i64;

    TORCH_CHECK(B>0 && H>0 && R>0 && R<=64, "invalid Gauss dimensions");
    TORCH_CHECK(T>=1 && T<=capacity, "used length must be in [1, cache capacity]");
    TORCH_CHECK(rope_base>0.0, "rope_base must be positive");
    TORCH_CHECK(rope_dim>=2 && rope_dim<=R && (rope_dim%2)==0,
                "rope_dim must be an even value in [2,R]");
    TORCH_CHECK(c.size(0)==B && c.size(1)==H && c.size(3)==R, "C cache shape mismatch");
    TORCH_CHECK(rho.size(0)==B && rho.size(1)==H && rho.size(2)==capacity, "rho cache shape mismatch");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_same_device(q,c,"q","c");
    check_same_device(q,rho,"q","rho");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    const float rope_log_base = logf((float)rope_base);

    if (mode==0) {
        gauss_rope_stream_partial<false><<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho.data_ptr<at::Half>()),
            nullptr,nullptr,
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,-1,(float)scale,rope_log_base,rope_dim
        );
    } else if (mode==1) {
        gauss_rope_tiled_partial<false><<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho.data_ptr<at::Half>()),
            nullptr,nullptr,
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,-1,(float)scale,rope_log_base,rope_dim
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }
    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,R,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void gauss_rope_decode_append_out_cuda(
    torch::Tensor q,
    torch::Tensor c_now,
    torch::Tensor rho_now,
    torch::Tensor c_cache,
    torch::Tensor rho_cache,
    torch::Tensor pm,
    torch::Tensor pl,
    torch::Tensor po,
    torch::Tensor out,
    double scale,
    int64_t mode,
    int64_t splits_i64,
    int64_t position_i64,
    double rope_base,
    int64_t rope_dim_i64
) {
    check_half(q,"q");
    check_half(c_now,"c_now");
    check_half(rho_now,"rho_now");
    check_half(c_cache,"c_cache");
    check_half(rho_cache,"rho_cache");
    check_half(out,"out");
    check_float(pm,"pm");
    check_float(pl,"pl");
    check_float(po,"po");

    TORCH_CHECK(q.dim()==3, "q must be [B,H,R]");
    TORCH_CHECK(c_now.dim()==3, "c_now must be [B,H,R]");
    TORCH_CHECK(rho_now.dim()==2, "rho_now must be [B,H]");
    TORCH_CHECK(c_cache.dim()==4, "c_cache must be [B,H,capacity,R]");
    TORCH_CHECK(rho_cache.dim()==3, "rho_cache must be [B,H,capacity]");
    const int B=(int)q.size(0);
    const int H=(int)q.size(1);
    const int R=(int)q.size(2);
    const int capacity=(int)c_cache.size(2);
    const int position=(int)position_i64;
    const int T=position+1;
    const int BH=B*H;
    const int splits=valid_splits(splits_i64);
    const int rope_dim=(int)rope_dim_i64;

    TORCH_CHECK(B>0 && H>0 && R>0 && R<=64, "invalid Gauss dimensions");
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    TORCH_CHECK(rope_base>0.0, "rope_base must be positive");
    TORCH_CHECK(rope_dim>=2 && rope_dim<=R && (rope_dim%2)==0,
                "rope_dim must be an even value in [2,R]");
    TORCH_CHECK(c_now.sizes()==q.sizes(), "current C shape mismatch");
    TORCH_CHECK(rho_now.size(0)==B && rho_now.size(1)==H, "current rho shape mismatch");
    TORCH_CHECK(c_cache.size(0)==B && c_cache.size(1)==H && c_cache.size(3)==R, "C cache shape mismatch");
    TORCH_CHECK(rho_cache.size(0)==B && rho_cache.size(1)==H && rho_cache.size(2)==capacity, "rho cache shape mismatch");
    TORCH_CHECK(out.sizes()==q.sizes(), "out shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_same_device(q,c_now,"q","c_now");
    check_same_device(q,rho_now,"q","rho_now");
    check_same_device(q,c_cache,"q","c_cache");
    check_same_device(q,rho_cache,"q","rho_cache");
    check_same_device(q,out,"q","out");
    check_same_device(q,pm,"q","pm");
    check_same_device(q,pl,"q","pl");
    check_same_device(q,po,"q","po");

    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    const float rope_log_base = logf((float)rope_base);

    if (mode==0) {
        gauss_rope_stream_partial<true><<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale,rope_log_base,rope_dim
        );
    } else if (mode==1) {
        gauss_rope_tiled_partial<true><<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale,rope_log_base,rope_dim
        );
    } else {
        TORCH_CHECK(false,"unknown mode");
    }
    merge_partial<<<BH,32,0,stream>>>(
        pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),BH,R,splits
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================
// RoPE variants of direct C+rho decode + W_O (NO GLOBAL O)
// ============================================================

torch::Tensor gauss_rope_decode_project_out_used_cuda(
    torch::Tensor q, torch::Tensor c, torch::Tensor rho,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits_i64, int64_t used_i64,
    double rope_base, int64_t rope_dim_i64
) {
    check_half(q,"q"); check_half(c,"c"); check_half(rho,"rho");
    check_float(pm,"pm"); check_float(pl,"pl"); check_float(po,"po");
    TORCH_CHECK(q.dim()==3 && c.dim()==4 && rho.dim()==3, "Gauss+RoPE rank mismatch");
    const int B=(int)q.size(0), H=(int)q.size(1), R=(int)q.size(2);
    const int capacity=(int)c.size(2), T=(int)used_i64, BH=B*H;
    const int splits=valid_splits(splits_i64), rope_dim=(int)rope_dim_i64;
    TORCH_CHECK(T>=1 && T<=capacity, "used length must be in [1, cache capacity]");
    TORCH_CHECK(rope_base>0.0 && rope_dim>=2 && rope_dim<=R && (rope_dim%2)==0, "invalid RoPE config");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_project_weight(q,weight,H,R);
    auto y=make_project_output(q,weight); const int D=(int)weight.size(0);
    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    const float rope_log_base=logf((float)rope_base);
    if(mode==0){
        gauss_rope_stream_partial<false><<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho.data_ptr<at::Half>()),nullptr,nullptr,
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,-1,(float)scale,rope_log_base,rope_dim);
    } else if(mode==1){
        gauss_rope_tiled_partial<false><<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho.data_ptr<at::Half>()),nullptr,nullptr,
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,-1,(float)scale,rope_log_base,rope_dim);
    } else TORCH_CHECK(false,"unknown mode");
    launch_merge_project(
        pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        B, H, R, D, splits, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

torch::Tensor gauss_rope_decode_append_project_out_cuda(
    torch::Tensor q, torch::Tensor c_now, torch::Tensor rho_now,
    torch::Tensor c_cache, torch::Tensor rho_cache,
    torch::Tensor pm, torch::Tensor pl, torch::Tensor po,
    torch::Tensor weight, double scale, int64_t mode, int64_t splits_i64, int64_t position_i64,
    double rope_base, int64_t rope_dim_i64
) {
    check_half(q,"q"); check_half(c_now,"c_now"); check_half(rho_now,"rho_now");
    check_half(c_cache,"c_cache"); check_half(rho_cache,"rho_cache");
    check_float(pm,"pm"); check_float(pl,"pl"); check_float(po,"po");
    TORCH_CHECK(q.dim()==3 && c_now.dim()==3 && rho_now.dim()==2, "current projected state rank mismatch");
    const int B=(int)q.size(0), H=(int)q.size(1), R=(int)q.size(2);
    const int capacity=(int)c_cache.size(2), position=(int)position_i64, T=position+1, BH=B*H;
    const int splits=valid_splits(splits_i64), rope_dim=(int)rope_dim_i64;
    TORCH_CHECK(position>=0 && position<capacity, "position exceeds cache capacity");
    TORCH_CHECK(rope_base>0.0 && rope_dim>=2 && rope_dim<=R && (rope_dim%2)==0, "invalid RoPE config");
    TORCH_CHECK(c_now.sizes()==q.sizes(), "current C shape mismatch");
    TORCH_CHECK(rho_now.size(0)==B && rho_now.size(1)==H, "current rho shape mismatch");
    TORCH_CHECK(pm.size(0)==BH && pm.size(1)==splits, "pm shape mismatch");
    TORCH_CHECK(pl.sizes()==pm.sizes(), "pl shape mismatch");
    TORCH_CHECK(po.size(0)==BH && po.size(1)==splits && po.size(2)==R, "po shape mismatch");
    check_project_weight(q,weight,H,R);
    auto y=make_project_output(q,weight); const int D=(int)weight.size(0);
    c10::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    const float rope_log_base=logf((float)rope_base);
    if(mode==0){
        gauss_rope_stream_partial<true><<<BH*splits,32,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale,rope_log_base,rope_dim);
    } else if(mode==1){
        gauss_rope_tiled_partial<true><<<BH*splits,TILE_THREADS,0,stream>>>(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c_cache.data_ptr<at::Half>()),
            reinterpret_cast<half*>(rho_cache.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(c_now.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(rho_now.data_ptr<at::Half>()),
            pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),
            BH,T,capacity,R,splits,position,(float)scale,rope_log_base,rope_dim);
    } else TORCH_CHECK(false,"unknown mode");
    launch_merge_project(
        pm.data_ptr<float>(), pl.data_ptr<float>(), po.data_ptr<float>(),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        B, H, R, D, splits, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}

