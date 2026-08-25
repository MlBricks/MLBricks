#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/AccumulateType.h>
#include <cmath>
#include <algorithm>

namespace mlbricks_vision {

static inline int64_t base_order_index(int64_t t, int64_t h, int64_t w, bool vertical) {
    if (!vertical) {
        const int64_t row = t / w;
        const int64_t off = t - row * w;
        const int64_t col = (row & 1) ? (w - 1 - off) : off;
        return row * w + col;
    }
    const int64_t col = t / h;
    const int64_t off = t - col * h;
    const int64_t row = (col & 1) ? (h - 1 - off) : off;
    return row * w + col;
}

static inline int64_t order_index(int64_t t, int64_t h, int64_t w, int64_t scan_kind, int64_t phase) {
    const int64_t n = h * w;
    bool reverse = false;
    bool vertical = false;
    bool raster = false;
    if (scan_kind == 3) { raster = true; reverse = (phase & 1); }
    else if (scan_kind == 1) { vertical = false; reverse = (phase & 1); }
    else if (scan_kind == 2) { vertical = true; reverse = (phase & 1); }
    else {
        const int64_t p = ((phase % 4) + 4) % 4;
        vertical = p >= 2;
        reverse = (p == 1 || p == 3);
    }
    const int64_t k = reverse ? (n - 1 - t) : t;
    return raster ? k : base_order_index(k, h, w, vertical);
}

torch::Tensor scan_reorder_cpu(const torch::Tensor& x_in, int64_t h, int64_t w, int64_t scan_kind, int64_t phase, bool inverse) {
    auto x = x_in.contiguous();
    auto out = torch::empty_like(x);
    const int64_t B = x.size(0), N = x.size(1), D = x.size(2);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "vision_scan_reorder_cpu", [&] {
        const scalar_t* src = x.data_ptr<scalar_t>();
        scalar_t* dst = out.data_ptr<scalar_t>();
        at::parallel_for(0, B * N, 0, [&](int64_t begin, int64_t end) {
            for (int64_t bt = begin; bt < end; ++bt) {
                const int64_t b = bt / N;
                const int64_t t = bt - b * N;
                const int64_t canonical = order_index(t, h, w, scan_kind, phase);
                const int64_t src_t = inverse ? t : canonical;
                const int64_t dst_t = inverse ? canonical : t;
                const scalar_t* s = src + (b * N + src_t) * D;
                scalar_t* d = dst + (b * N + dst_t) * D;
                std::copy(s, s + D, d);
            }
        });
    });
    return out;
}

static inline float coord_value(int64_t pos, int64_t length, int64_t channel, int64_t channels) {
    if (channels <= 0) return 0.0f;
    const int64_t pairs = channels / 2;
    if (pairs == 0) {
        return channel == 0 ? static_cast<float>(pos) / static_cast<float>(std::max<int64_t>(length - 1, 1)) : 0.0f;
    }
    if (channel >= 2 * pairs) return 0.0f;
    const bool cosine = channel >= pairs;
    const int64_t idx = cosine ? (channel - pairs) : channel;
    const float inv = std::exp(-std::log(10000.0f) * static_cast<float>(idx) / static_cast<float>(std::max<int64_t>(pairs, 1)));
    const float angle = static_cast<float>(pos) * inv;
    return cosine ? std::cos(angle) : std::sin(angle);
}

torch::Tensor sincos2d_cpu(const torch::Tensor& reference, int64_t h, int64_t w) {
    const int64_t D = reference.size(-1), N = h * w;
    auto out = torch::empty({N, D}, reference.options());
    const int64_t ydim = D / 2, xdim = D - ydim;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, reference.scalar_type(), "vision_sincos2d_cpu", [&] {
        scalar_t* dst = out.data_ptr<scalar_t>();
        at::parallel_for(0, N * D, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                const int64_t t = i / D, d = i - t * D;
                const int64_t row = t / w, col = t - row * w;
                const float v = d < ydim ? coord_value(row, h, d, ydim) : coord_value(col, w, d - ydim, xdim);
                dst[i] = static_cast<scalar_t>(v);
            }
        });
    });
    return out;
}

torch::Tensor add_sincos2d_cpu(const torch::Tensor& x, int64_t h, int64_t w) {
    auto pos = sincos2d_cpu(x, h, w);
    return x + pos.unsqueeze(0);
}

torch::Tensor unpatchify_cpu(const torch::Tensor& patches_in, int64_t gh, int64_t gw, int64_t p, int64_t c) {
    auto patches = patches_in.contiguous();
    const int64_t B = patches.size(0), H = gh * p, W = gw * p;
    auto out = torch::empty({B, c, H, W}, patches.options());
    const int64_t N = gh * gw, F = p * p * c;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, patches.scalar_type(), "vision_unpatchify_cpu", [&] {
        const scalar_t* src = patches.data_ptr<scalar_t>();
        scalar_t* dst = out.data_ptr<scalar_t>();
        at::parallel_for(0, B * c * H * W, 0, [&](int64_t begin, int64_t end) {
            for (int64_t idx = begin; idx < end; ++idx) {
                int64_t r = idx;
                const int64_t x = r % W; r /= W;
                const int64_t y = r % H; r /= H;
                const int64_t ch = r % c; const int64_t b = r / c;
                const int64_t pr = y / p, pc = x / p, iy = y % p, ix = x % p;
                const int64_t token = pr * gw + pc;
                const int64_t feat = ((iy * p + ix) * c) + ch;
                dst[idx] = src[(b * N + token) * F + feat];
            }
        });
    });
    return out;
}

torch::Tensor patchify_layout_cpu(const torch::Tensor& image_in, int64_t p) {
    auto image = image_in.contiguous();
    const int64_t B=image.size(0), C=image.size(1), H=image.size(2), W=image.size(3);
    const int64_t gh=H/p, gw=W/p, N=gh*gw, F=p*p*C;
    auto out = torch::empty({B,N,F}, image.options());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, image.scalar_type(), "vision_patchify_layout_cpu", [&] {
        const scalar_t* src=image.data_ptr<scalar_t>(); scalar_t* dst=out.data_ptr<scalar_t>();
        at::parallel_for(0, B*N*F, 0, [&](int64_t begin,int64_t end){
            for(int64_t idx=begin;idx<end;++idx){
                int64_t r=idx; const int64_t f=r%F; r/=F; const int64_t t=r%N; const int64_t b=r/N;
                const int64_t ch=f%C; const int64_t pix=f/C; const int64_t iy=pix/p, ix=pix%p;
                const int64_t pr=t/gw, pc=t%gw; const int64_t y=pr*p+iy, x=pc*p+ix;
                dst[idx]=src[((b*C+ch)*H+y)*W+x];
            }
        });
    });
    return out;
}

} // namespace mlbricks_vision
