#include <torch/extension.h>
#include <torch/library.h>
#include <cmath>
#include <limits>
#include <vector>

namespace mlbricks_vision {

// CPU implementations.
torch::Tensor scan_reorder_cpu(const torch::Tensor& x, int64_t height, int64_t width, int64_t scan_kind, int64_t phase, bool inverse);
torch::Tensor add_sincos2d_cpu(const torch::Tensor& x, int64_t height, int64_t width);
torch::Tensor sincos2d_cpu(const torch::Tensor& reference, int64_t height, int64_t width);
torch::Tensor unpatchify_cpu(const torch::Tensor& patches, int64_t gh, int64_t gw, int64_t patch, int64_t channels);
torch::Tensor patchify_layout_cpu(const torch::Tensor& image, int64_t patch);

#ifdef WITH_CUDA
torch::Tensor bolt_full_fused_cuda(
    const torch::Tensor& q, const torch::Tensor& u, const torch::Tensor& g,
    int64_t heads, int64_t latent_dim, int64_t head_dim, double eps, bool causal);
torch::Tensor scan_reorder_cuda(const torch::Tensor& x, int64_t height, int64_t width, int64_t scan_kind, int64_t phase, bool inverse);
torch::Tensor add_sincos2d_cuda(const torch::Tensor& x, int64_t height, int64_t width);
torch::Tensor sincos2d_cuda(const torch::Tensor& reference, int64_t height, int64_t width);
torch::Tensor unpatchify_cuda(const torch::Tensor& patches, int64_t gh, int64_t gw, int64_t patch, int64_t channels);
torch::Tensor patchify_layout_cuda(const torch::Tensor& image, int64_t patch);
#endif

static void check_tokens(const torch::Tensor& x, int64_t height, int64_t width) {
    TORCH_CHECK(x.dim() == 3, "x must have shape [B,N,D]");
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(x.size(1) == height * width, "token count must equal height*width");
    TORCH_CHECK(x.is_floating_point(), "vision native token ops require floating-point tensors");
}

torch::Tensor scan_reorder(
    const torch::Tensor& x,
    int64_t height,
    int64_t width,
    int64_t scan_kind,
    int64_t phase,
    bool inverse) {
    check_tokens(x, height, width);
    TORCH_CHECK(scan_kind >= 0 && scan_kind <= 3, "scan_kind must be 0..3");
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        return scan_reorder_cuda(x, height, width, scan_kind, phase, inverse);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return scan_reorder_cpu(x, height, width, scan_kind, phase, inverse);
}

torch::Tensor add_sincos2d(const torch::Tensor& x, int64_t height, int64_t width) {
    check_tokens(x, height, width);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        return add_sincos2d_cuda(x, height, width);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return add_sincos2d_cpu(x, height, width);
}

torch::Tensor sincos2d(const torch::Tensor& reference, int64_t height, int64_t width) {
    TORCH_CHECK(reference.dim() >= 1, "reference must have at least one dimension");
    TORCH_CHECK(reference.is_floating_point(), "reference must be floating point");
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    const int64_t dim = reference.size(-1);
    TORCH_CHECK(dim > 0, "embedding dimension must be positive");
    if (reference.is_cuda()) {
#ifdef WITH_CUDA
        return sincos2d_cuda(reference, height, width);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return sincos2d_cpu(reference, height, width);
}

torch::Tensor unpatchify(
    const torch::Tensor& patches,
    int64_t gh,
    int64_t gw,
    int64_t patch,
    int64_t channels) {
    TORCH_CHECK(patches.dim() == 3, "patches must have shape [B,N,P*P*C]");
    TORCH_CHECK(gh > 0 && gw > 0 && patch > 0 && channels > 0, "grid/patch/channels must be positive");
    TORCH_CHECK(patches.size(1) == gh * gw, "patch count must equal grid size");
    TORCH_CHECK(patches.size(2) == patch * patch * channels, "patch feature size mismatch");
    TORCH_CHECK(patches.is_floating_point(), "patches must be floating point");
    if (patches.is_cuda()) {
#ifdef WITH_CUDA
        return unpatchify_cuda(patches, gh, gw, patch, channels);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return unpatchify_cpu(patches, gh, gw, patch, channels);
}

torch::Tensor patchify_layout(const torch::Tensor& image, int64_t patch) {
    TORCH_CHECK(image.dim() == 4, "image must have shape [B,C,H,W]");
    TORCH_CHECK(patch > 0, "patch must be positive");
    TORCH_CHECK(image.size(2) % patch == 0 && image.size(3) % patch == 0, "image size must be divisible by patch");
    TORCH_CHECK(image.is_floating_point(), "image must be floating point");
    if (image.is_cuda()) {
#ifdef WITH_CUDA
        return patchify_layout_cuda(image, patch);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return patchify_layout_cpu(image, patch);
}

// Full-sequence Bolt core. This is intentionally a CompositeImplicitAutograd
// operator: projections remain ordinary Linear/cuBLAS operations and this C++
// body owns the Bolt equation while ATen dispatches matmul/softmax to the
// current CPU/CUDA backend with autograd support.
torch::Tensor bolt_full(
    const torch::Tensor& q_in,
    const torch::Tensor& u_in,
    const torch::Tensor& g_in,
    int64_t heads,
    int64_t latent_dim,
    int64_t head_dim,
    double eps,
    bool causal) {
    TORCH_CHECK(q_in.dim() == 3 && u_in.dim() == 3 && g_in.dim() == 3,
                "q/u/g must have shape [B,T,H*R]");
    TORCH_CHECK(q_in.sizes() == u_in.sizes() && q_in.sizes() == g_in.sizes(),
                "q/u/g shapes must match");
    TORCH_CHECK(heads > 0 && latent_dim > 0 && head_dim > 0, "invalid Bolt dimensions");
    TORCH_CHECK(q_in.size(2) == heads * latent_dim, "last dimension must equal heads*latent_dim");
    TORCH_CHECK(q_in.device() == u_in.device() && q_in.device() == g_in.device(), "q/u/g devices must match");
    TORCH_CHECK(q_in.scalar_type() == u_in.scalar_type() && q_in.scalar_type() == g_in.scalar_type(), "q/u/g dtypes must match");

    const int64_t B = q_in.size(0);
    const int64_t T = q_in.size(1);
    auto c_flat = u_in * (1.0 + torch::tanh(g_in));
    auto q = q_in.view({B, T, heads, latent_dim}).transpose(1, 2);
    auto c = c_flat.view({B, T, heads, latent_dim}).transpose(1, 2);
    auto rho = torch::rsqrt(c.to(torch::kFloat).pow(2).mean(-1) + eps);
    auto scores = torch::matmul(q, c.transpose(-2, -1));
    scores = scores.to(torch::kFloat) * rho.unsqueeze(-2);
    scores = scores * (1.0 / std::sqrt(static_cast<double>(head_dim)));
    if (causal && T > 0) {
        auto mask = torch::ones({T, T}, torch::TensorOptions().device(q_in.device()).dtype(torch::kBool)).tril();
        scores = scores.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
    }
    auto p = torch::softmax(scores, -1).to(c.scalar_type());
    auto y = torch::matmul(p, c);
    return y.transpose(1, 2).contiguous().view({B, T, heads * latent_dim});
}


torch::Tensor bolt_full_fused(
    const torch::Tensor& q,
    const torch::Tensor& u,
    const torch::Tensor& g,
    int64_t heads,
    int64_t latent_dim,
    int64_t head_dim,
    double eps,
    bool causal) {
    if (q.is_cuda()) {
#ifdef WITH_CUDA
        TORCH_CHECK(latent_dim <= 64, "fused CUDA Bolt currently supports latent_dim <= 64");
        return bolt_full_fused_cuda(q, u, g, heads, latent_dim, head_dim, eps, causal);
#else
        TORCH_CHECK(false, "MLBricks vision native extension was built without CUDA support");
#endif
    }
    return bolt_full(q, u, g, heads, latent_dim, head_dim, eps, causal);
}

torch::Tensor perspective_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t groups,
    double eps) {
    TORCH_CHECK(x.dim() >= 2, "perspective_norm x must have rank >=2");
    TORCH_CHECK(groups > 0 && x.size(-1) % groups == 0, "invalid perspective groups");
    const int64_t gd = x.size(-1) / groups;
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == groups && weight.size(1) == gd, "weight must have shape [groups, group_dim]");
    TORCH_CHECK(bias.sizes() == weight.sizes(), "bias shape must match weight");
    TORCH_CHECK(weight.device() == x.device() && bias.device() == x.device(), "norm parameters must share x device");
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end());
    shape.back() = groups;
    shape.push_back(gd);
    auto y = x.view(shape);
    auto yf = y.to(torch::kFloat);
    auto mean = yf.mean(-1, true);
    auto var = (yf - mean).pow(2).mean(-1, true);
    auto norm = (yf - mean) * torch::rsqrt(var + eps);
    auto out = norm.to(x.scalar_type()) * weight + bias;
    return out.reshape(x.sizes());
}

bool has_cuda() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

} // namespace mlbricks_vision

TORCH_LIBRARY(mlbricks_vision_native, m) {
    m.def("scan_reorder(Tensor x, int height, int width, int scan_kind, int phase, bool inverse=False) -> Tensor");
    m.def("add_sincos2d(Tensor x, int height, int width) -> Tensor");
    m.def("sincos2d(Tensor reference, int height, int width) -> Tensor");
    m.def("unpatchify(Tensor patches, int gh, int gw, int patch, int channels) -> Tensor");
    m.def("patchify_layout(Tensor image, int patch) -> Tensor");
    m.def("bolt_full(Tensor q, Tensor u, Tensor g, int heads, int latent_dim, int head_dim, float eps, bool causal) -> Tensor");
    m.def("bolt_full_fused(Tensor q, Tensor u, Tensor g, int heads, int latent_dim, int head_dim, float eps, bool causal) -> Tensor");
    m.def("perspective_norm(Tensor x, Tensor weight, Tensor bias, int groups, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlbricks_vision_native, CompositeExplicitAutograd, m) {
    m.impl("scan_reorder", TORCH_FN(mlbricks_vision::scan_reorder));
    m.impl("add_sincos2d", TORCH_FN(mlbricks_vision::add_sincos2d));
    m.impl("sincos2d", TORCH_FN(mlbricks_vision::sincos2d));
    m.impl("unpatchify", TORCH_FN(mlbricks_vision::unpatchify));
    m.impl("patchify_layout", TORCH_FN(mlbricks_vision::patchify_layout));
    m.impl("bolt_full_fused", TORCH_FN(mlbricks_vision::bolt_full_fused));
}

TORCH_LIBRARY_IMPL(mlbricks_vision_native, CompositeImplicitAutograd, m) {
    m.impl("bolt_full", TORCH_FN(mlbricks_vision::bolt_full));
    m.impl("perspective_norm", TORCH_FN(mlbricks_vision::perspective_norm));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MLBricks shared vision C++/CUDA operators";
    m.def("has_cuda", &mlbricks_vision::has_cuda, "Whether CUDA kernels were compiled");
}
