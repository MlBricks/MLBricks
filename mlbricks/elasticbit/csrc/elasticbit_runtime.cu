#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/ops/mm.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>

namespace py = pybind11;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                           \
        if (error__ != cudaSuccess) {                                           \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(error__) +   \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)           \
            );                                                                  \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// ElasticBit vNext invariants
// -----------------------------------------------------------------------------
// Storage is adaptive and threshold-driven. Users never select a bit width.
// Search: 3..15 bits, with FP16 (16 bit) as the threshold-preserving fallback.
// Activations remain FP16 in every execution policy.
// hardwareNative execution is phase- and hardware-aware. Tesla T4 keeps the
// validated mapping 3..4 -> W4A16, 5..8 -> W8A16, 9..16 -> W16A16. Other
// CUDA GPUs benchmark the legal W4A16/W8A16/W16A16 candidates for each matrix
// shape and select the fastest built-in ElasticBit decode kernel.
// fullPrecision execution: all storage widths -> W16A16.
// Prefill always materializes FP16 weights on-device and delegates GEMM to
// PyTorch/vendor libraries; activations are never quantized.
// Bit selection validates BOTH low-bit and full-precision arithmetic so policy
// switching and hardware widening never invalidate the compression threshold.
// -----------------------------------------------------------------------------

static constexpr int kMinCompressedBits = 3;
static constexpr int kMaxCompressedBits = 15;
static constexpr int kFallbackBits = 16;

static void validate_storage_bits(int bits) {
    if (bits < kMinCompressedBits || bits > kFallbackBits) {
        throw std::invalid_argument("ElasticBit storage bits must be between 3 and 16");
    }
}

static inline uint32_t qmax_for_bits(int bits) {
    if (bits >= 32) return 0x7fffffffu;
    return (1u << (bits - 1)) - 1u;
}

static inline size_t packed_bytes_for_values(size_t value_count, int bits) {
    return (value_count * static_cast<size_t>(bits) + 7u) / 8u;
}

static inline float round_to_fp16_host(float value) {
    return __half2float(__float2half_rn(value));
}

static inline int32_t quantize_scalar(float value, float scale, int bits) {
    const int32_t qmax = static_cast<int32_t>(qmax_for_bits(bits));
    int32_t q = static_cast<int32_t>(std::nearbyint(value / scale));
    return std::max(-qmax, std::min(qmax, q));
}

static std::vector<float> make_row_scales(
    const float* weights,
    int rows,
    int cols,
    int bits
) {
    if (bits == kFallbackBits) return {};
    const float qmax = static_cast<float>(qmax_for_bits(bits));
    std::vector<float> scales(rows, 1.0f);
    for (int row = 0; row < rows; ++row) {
        const float* row_ptr = weights + static_cast<size_t>(row) * cols;
        float max_abs = 0.0f;
        for (int col = 0; col < cols; ++col) {
            max_abs = std::max(max_abs, std::fabs(row_ptr[col]));
        }
        scales[row] = max_abs > 0.0f ? max_abs / qmax : 1.0f;
    }
    return scales;
}

static std::vector<uint8_t> pack_exact_weights(
    const float* weights,
    int rows,
    int cols,
    int bits,
    const std::vector<float>& scales
) {
    validate_storage_bits(bits);
    const size_t count = static_cast<size_t>(rows) * cols;

    if (bits == kFallbackBits) {
        std::vector<uint8_t> payload(count * sizeof(__half));
        __half* half_values = reinterpret_cast<__half*>(payload.data());
        for (size_t index = 0; index < count; ++index) {
            half_values[index] = __float2half_rn(weights[index]);
        }
        return payload;
    }

    const size_t payload_bytes = packed_bytes_for_values(count, bits);
    std::vector<uint8_t> payload(payload_bytes, 0u);
    const int32_t qmax = static_cast<int32_t>(qmax_for_bits(bits));

    for (int row = 0; row < rows; ++row) {
        const float scale = scales[row];
        for (int col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row) * cols + col;
            const int32_t q = quantize_scalar(weights[index], scale, bits);
            const uint32_t code = static_cast<uint32_t>(q + qmax);
            const size_t bit_offset = index * static_cast<size_t>(bits);
            const size_t byte_offset = bit_offset >> 3;
            const int shift = static_cast<int>(bit_offset & 7u);
            const uint64_t shifted = static_cast<uint64_t>(code) << shift;
            // At most 15 bits plus a 7-bit bit offset => at most 22 bits.
            for (int byte_index = 0; byte_index < 3; ++byte_index) {
                const size_t target = byte_offset + static_cast<size_t>(byte_index);
                if (target < payload.size()) {
                    payload[target] |= static_cast<uint8_t>(
                        (shifted >> (8 * byte_index)) & 0xffull
                    );
                }
            }
        }
    }
    return payload;
}

static inline uint32_t extract_code_host(
    const uint8_t* payload,
    size_t payload_bytes,
    size_t index,
    int bits
) {
    const size_t bit_offset = index * static_cast<size_t>(bits);
    const size_t byte_offset = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7u);
    uint32_t word = 0u;
    for (int byte_index = 0; byte_index < 3; ++byte_index) {
        const size_t source = byte_offset + static_cast<size_t>(byte_index);
        if (source < payload_bytes) {
            word |= static_cast<uint32_t>(payload[source]) << (8 * byte_index);
        }
    }
    const uint32_t mask = (1u << bits) - 1u;
    return (word >> shift) & mask;
}

static std::vector<float> dequantize_payload_host(
    const std::vector<uint8_t>& payload,
    const std::vector<float>& scales,
    int rows,
    int cols,
    int bits
) {
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<float> output(count);
    if (bits == kFallbackBits) {
        const __half* half_values = reinterpret_cast<const __half*>(payload.data());
        for (size_t index = 0; index < count; ++index) {
            output[index] = __half2float(half_values[index]);
        }
        return output;
    }

    const int32_t qmax = static_cast<int32_t>(qmax_for_bits(bits));
    for (int row = 0; row < rows; ++row) {
        const float scale = scales[row];
        for (int col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row) * cols + col;
            const uint32_t code = extract_code_host(
                payload.data(), payload.size(), index, bits
            );
            const int32_t q = static_cast<int32_t>(code) - qmax;
            output[index] = round_to_fp16_host(static_cast<float>(q) * scale);
        }
    }
    return output;
}

static int execution_width_for_bits(int storage_bits) {
    if (storage_bits <= 4) return 4;
    if (storage_bits <= 8) return 8;
    return 16;
}

enum class DecodePolicy : uint8_t {
    HardwareNative = 1,
    FullPrecision = 2,
};

static DecodePolicy parse_decode_policy(const std::string& value) {
    if (value == "hardwareNative") return DecodePolicy::HardwareNative;
    if (value == "fullPrecision") return DecodePolicy::FullPrecision;
    throw std::invalid_argument(
        "decodePolicy must be 'hardwareNative' or 'fullPrecision'"
    );
}

static const char* decode_policy_name(DecodePolicy value) {
    return value == DecodePolicy::HardwareNative
        ? "hardwareNative"
        : "fullPrecision";
}

struct BitAnalysis {
    int bits = 16;
    double hardware_native_error = 0.0;
    double full_precision_error = 0.0;
    double error = 0.0;
    size_t storage_bytes = 0;
};

static std::vector<float> fp16_reference_outputs(
    const float* weights,
    int rows,
    int cols,
    const float* calibration,
    int samples
) {
    std::vector<float> refs(static_cast<size_t>(samples) * rows, 0.0f);
    std::vector<float> half_weights(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < half_weights.size(); ++i) {
        half_weights[i] = round_to_fp16_host(weights[i]);
    }

    for (int sample = 0; sample < samples; ++sample) {
        const float* input = calibration + static_cast<size_t>(sample) * cols;
        std::vector<float> input_half(cols);
        for (int col = 0; col < cols; ++col) {
            input_half[col] = round_to_fp16_host(input[col]);
        }
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            const float* row_ptr = half_weights.data() + static_cast<size_t>(row) * cols;
            for (int col = 0; col < cols; ++col) {
                sum += row_ptr[col] * input_half[col];
            }
            refs[static_cast<size_t>(sample) * rows + row] = sum;
        }
    }
    return refs;
}

static BitAnalysis analyze_one_bit_width(
    const float* weights,
    int rows,
    int cols,
    const float* calibration,
    int samples,
    const std::vector<float>& references,
    int bits
) {
    validate_storage_bits(bits);
    const std::vector<float> scales = make_row_scales(weights, rows, cols, bits);

    long double hw_num = 0.0L;
    long double full_num = 0.0L;
    long double denom = 0.0L;

    for (int sample = 0; sample < samples; ++sample) {
        const float* input = calibration + static_cast<size_t>(sample) * cols;
        std::vector<float> input_half(cols);
        for (int col = 0; col < cols; ++col) {
            input_half[col] = round_to_fp16_host(input[col]);
        }

        for (int row = 0; row < rows; ++row) {
            const float* row_ptr = weights + static_cast<size_t>(row) * cols;
            float hw_sum = 0.0f;
            float full_sum = 0.0f;

            if (bits == kFallbackBits) {
                for (int col = 0; col < cols; ++col) {
                    const float w = round_to_fp16_host(row_ptr[col]);
                    hw_sum += w * input_half[col];
                }
                full_sum = hw_sum;
            } else if (bits <= 8) {
                const float scale = scales[row];
                float integer_dot = 0.0f;
                for (int col = 0; col < cols; ++col) {
                    const int32_t q = quantize_scalar(row_ptr[col], scale, bits);
                    integer_dot += static_cast<float>(q) * input_half[col];
                    const float deq_half = round_to_fp16_host(
                        static_cast<float>(q) * scale
                    );
                    full_sum += deq_half * input_half[col];
                }
                hw_sum = integer_dot * scale;
            } else {
                const float scale = scales[row];
                for (int col = 0; col < cols; ++col) {
                    const int32_t q = quantize_scalar(row_ptr[col], scale, bits);
                    const float deq_half = round_to_fp16_host(
                        static_cast<float>(q) * scale
                    );
                    hw_sum += deq_half * input_half[col];
                }
                full_sum = hw_sum;
            }

            const double ref = references[static_cast<size_t>(sample) * rows + row];
            const double hw_diff = static_cast<double>(hw_sum) - ref;
            const double full_diff = static_cast<double>(full_sum) - ref;
            hw_num += static_cast<long double>(hw_diff) * hw_diff;
            full_num += static_cast<long double>(full_diff) * full_diff;
            denom += static_cast<long double>(ref) * ref;
        }
    }

    const long double safe_denom = std::max(denom, 1.0e-30L);
    const double hw_error = std::sqrt(static_cast<double>(hw_num / safe_denom));
    const double full_error = std::sqrt(static_cast<double>(full_num / safe_denom));
    const double error = std::max(hw_error, full_error);

    const size_t count = static_cast<size_t>(rows) * cols;
    const size_t payload_bytes = bits == kFallbackBits
        ? count * sizeof(__half)
        : packed_bytes_for_values(count, bits);
    const size_t scale_bytes = bits == kFallbackBits
        ? 0u
        : static_cast<size_t>(rows) * sizeof(float);

    return BitAnalysis{bits, hw_error, full_error, error, payload_bytes + scale_bytes};
}

static std::vector<BitAnalysis> analyze_all_bits(
    const float* weights,
    int rows,
    int cols,
    const float* calibration,
    int samples
) {
    if (rows <= 0 || cols <= 0 || samples <= 0) {
        throw std::invalid_argument("weights and calibration must be non-empty");
    }
    const std::vector<float> refs = fp16_reference_outputs(
        weights, rows, cols, calibration, samples
    );
    std::vector<BitAnalysis> analyses;
    analyses.reserve(kFallbackBits - kMinCompressedBits + 1);
    for (int bits = kMinCompressedBits; bits <= kFallbackBits; ++bits) {
        analyses.push_back(analyze_one_bit_width(
            weights, rows, cols, calibration, samples, refs, bits
        ));
    }
    return analyses;
}

static BitAnalysis select_analysis(
    const std::vector<BitAnalysis>& analyses,
    double threshold
) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::invalid_argument("threshold must be a finite non-negative value");
    }
    for (const auto& item : analyses) {
        if (item.bits <= kMaxCompressedBits && item.error <= threshold) {
            return item;
        }
    }
    // FP16 is the threshold-preserving fallback. Its error is zero relative to
    // the FP16 reference by construction.
    const auto& fallback = analyses.back();
    if (fallback.bits != kFallbackBits) {
        throw std::runtime_error("ElasticBit analyzer internal fallback error");
    }
    return fallback;
}

static py::dict analyze_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> weights,
    py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
    double threshold
) {
    auto w = weights.request();
    auto c = calibration.request();
    if (w.ndim != 2 || c.ndim != 2) {
        throw std::invalid_argument("weights and calibrationData must both be 2D");
    }
    const int rows = static_cast<int>(w.shape[0]);
    const int cols = static_cast<int>(w.shape[1]);
    const int samples = static_cast<int>(c.shape[0]);
    if (c.shape[1] != cols) {
        throw std::invalid_argument("calibrationData width must equal weight cols");
    }

    const auto analyses = analyze_all_bits(
        static_cast<const float*>(w.ptr), rows, cols,
        static_cast<const float*>(c.ptr), samples
    );
    const auto selected = select_analysis(analyses, threshold);

    py::list candidates;
    const double original_bytes = static_cast<double>(
        static_cast<size_t>(rows) * cols * sizeof(__half)
    );
    for (const auto& item : analyses) {
        py::dict row;
        row["bits"] = item.bits;
        row["hardwareNativeError"] = item.hardware_native_error;
        row["fullPrecisionError"] = item.full_precision_error;
        row["error"] = item.error;
        row["storageBytes"] = item.storage_bytes;
        row["memoryReduction"] = 1.0 - static_cast<double>(item.storage_bytes) / original_bytes;
        row["passes"] = item.error <= threshold;
        candidates.append(row);
    }

    py::dict result;
    result["threshold"] = threshold;
    result["selectedBits"] = selected.bits;
    result["selectedError"] = selected.error;
    result["candidates"] = candidates;
    return result;
}

// -----------------------------------------------------------------------------
// CUDA execution kernels. Activations are always FP16.
// -----------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

__device__ __forceinline__ int sign4(uint32_t nibble) {
    const int value = static_cast<int>(nibble & 0x0fu);
    return (value ^ 8) - 8;
}

// Validated T4-style W4A16 decode kernel.  The hot Granite shapes are all
// multiples of 8, so each thread consumes eight packed 4-bit weights from one
// 32-bit load and four FP16 half2 activation loads.  A scalar tail path keeps
// the RuntimeMatrix API correct for arbitrary matrix widths.
__global__ void w4a16_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ scales,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int cols,
    int row_bytes
) {
    constexpr int kWarps = 8;  // launch_width uses 256 threads.
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;

    const uint8_t* row_bytes_ptr = weights + static_cast<size_t>(row) * row_bytes;
    float partial = 0.0f;

    if ((cols & 7) == 0) {
        const uint32_t* row4 = reinterpret_cast<const uint32_t*>(row_bytes_ptr);
        const int groups8 = cols >> 3;
        for (int group = tid; group < groups8; group += blockDim.x) {
            const uint32_t packed = row4[group];
            const int col = group << 3;
            const __half2* x2 = reinterpret_cast<const __half2*>(input + col);
            const float2 x0 = __half22float2(x2[0]);
            const float2 x1 = __half22float2(x2[1]);
            const float2 x2v = __half22float2(x2[2]);
            const float2 x3 = __half22float2(x2[3]);

            partial = fmaf(x0.x, static_cast<float>(sign4(packed >> 0)), partial);
            partial = fmaf(x0.y, static_cast<float>(sign4(packed >> 4)), partial);
            partial = fmaf(x1.x, static_cast<float>(sign4(packed >> 8)), partial);
            partial = fmaf(x1.y, static_cast<float>(sign4(packed >> 12)), partial);
            partial = fmaf(x2v.x, static_cast<float>(sign4(packed >> 16)), partial);
            partial = fmaf(x2v.y, static_cast<float>(sign4(packed >> 20)), partial);
            partial = fmaf(x3.x, static_cast<float>(sign4(packed >> 24)), partial);
            partial = fmaf(x3.y, static_cast<float>(sign4(packed >> 28)), partial);
        }
    } else {
        for (int col = tid; col < cols; col += blockDim.x) {
            const uint8_t packed = row_bytes_ptr[col >> 1];
            const uint8_t nibble = (col & 1) ? (packed >> 4) : (packed & 0x0f);
            partial = fmaf(
                __half2float(input[col]), static_cast<float>(sign4(nibble)), partial
            );
        }
    }

    partial = warp_sum(partial);
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();

    if (warp == 0) {
        float total = lane < kWarps ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) output[row] = __float2half_rn(total * scales[row]);
    }
}

// Validated T4-style W8A16 decode kernel.  Granite's hot shapes are multiples
// of 4: one 32-bit weight load supplies four INT8 values and activations are
// consumed as two half2 values.  The scalar fallback preserves generic shapes.
__global__ void w8a16_gemv_kernel(
    const int8_t* __restrict__ weights,
    const float* __restrict__ scales,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int cols
) {
    constexpr int kWarps = 8;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;

    const int8_t* row_ptr = weights + static_cast<size_t>(row) * cols;
    float partial = 0.0f;

    if ((cols & 3) == 0) {
        const uint32_t* row4 = reinterpret_cast<const uint32_t*>(row_ptr);
        const int groups4 = cols >> 2;
        for (int group = tid; group < groups4; group += blockDim.x) {
            const uint32_t packed = row4[group];
            const int col = group << 2;
            const int q0 = static_cast<int>(static_cast<int8_t>((packed >> 0) & 0xffu));
            const int q1 = static_cast<int>(static_cast<int8_t>((packed >> 8) & 0xffu));
            const int q2 = static_cast<int>(static_cast<int8_t>((packed >> 16) & 0xffu));
            const int q3 = static_cast<int>(static_cast<int8_t>((packed >> 24) & 0xffu));
            const __half2* x2 = reinterpret_cast<const __half2*>(input + col);
            const float2 a = __half22float2(x2[0]);
            const float2 b = __half22float2(x2[1]);
            partial = fmaf(a.x, static_cast<float>(q0), partial);
            partial = fmaf(a.y, static_cast<float>(q1), partial);
            partial = fmaf(b.x, static_cast<float>(q2), partial);
            partial = fmaf(b.y, static_cast<float>(q3), partial);
        }
    } else {
        for (int col = tid; col < cols; col += blockDim.x) {
            partial = fmaf(
                __half2float(input[col]), static_cast<float>(row_ptr[col]), partial
            );
        }
    }

    partial = warp_sum(partial);
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();

    if (warp == 0) {
        float total = lane < kWarps ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) output[row] = __float2half_rn(total * scales[row]);
    }
}

// Kept for the low-level RuntimeMatrix host/benchmark API.  Model-level W16A16
// execution bypasses this kernel and uses the framework/vendor GEMM path.
__global__ void w16a16_gemv_kernel(
    const __half* __restrict__ weights,
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int cols
) {
    constexpr int kWarps = 8;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (row >= rows) return;

    const __half* row_ptr = weights + static_cast<size_t>(row) * cols;
    float partial = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        partial = fmaf(__half2float(row_ptr[col]), __half2float(input[col]), partial);
    }
    partial = warp_sum(partial);
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float total = lane < kWarps ? warp_sums[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) output[row] = __float2half_rn(total);
    }
}

__global__ void w4_to_fp16_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ scales,
    __half* __restrict__ output,
    int rows,
    int cols,
    int row_bytes
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * cols;
    if (index >= count) return;
    const int row = static_cast<int>(index / cols);
    const int col = static_cast<int>(index - static_cast<size_t>(row) * cols);
    const uint8_t packed = weights[static_cast<size_t>(row) * row_bytes + (col >> 1)];
    const uint8_t nibble = (col & 1) ? (packed >> 4) : (packed & 0x0f);
    output[index] = __float2half_rn(static_cast<float>(sign4(nibble)) * scales[row]);
}

__global__ void w8_to_fp16_kernel(
    const int8_t* __restrict__ weights,
    const float* __restrict__ scales,
    __half* __restrict__ output,
    int rows,
    int cols
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * cols;
    if (index >= count) return;
    const int row = static_cast<int>(index / cols);
    output[index] = __float2half_rn(static_cast<float>(weights[index]) * scales[row]);
}

static uint64_t fnv1a64(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < size; ++index) {
        hash ^= static_cast<uint64_t>(bytes[index]);
        hash *= 1099511628211ull;
    }
    return hash;
}

#pragma pack(push, 1)
struct MLB4Header {
    char magic[4];
    uint16_t version;
    uint16_t header_bytes;
    uint32_t rows;
    uint32_t cols;
    uint8_t storage_bits;
    uint8_t reserved[3];
    uint32_t scale_count;
    uint64_t scale_bytes;
    uint64_t payload_bytes;
    uint64_t original_fp16_bytes;
    uint64_t scale_checksum;
    uint64_t payload_checksum;
    double threshold;
    double selected_error;
};
#pragma pack(pop)

static_assert(sizeof(MLB4Header) == 80, "Unexpected MLB4Header size");

class RuntimeMatrix {
public:
    ~RuntimeMatrix() { release_device_memory(); }
    RuntimeMatrix(const RuntimeMatrix&) = delete;
    RuntimeMatrix& operator=(const RuntimeMatrix&) = delete;

    static std::unique_ptr<RuntimeMatrix> compress(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
        double threshold,
        const std::string& decode_policy = "hardwareNative"
    ) {
        auto w = weights.request();
        auto c = calibration.request();
        if (w.ndim != 2 || c.ndim != 2) {
            throw std::invalid_argument("weights and calibrationData must both be 2D");
        }
        const int rows = static_cast<int>(w.shape[0]);
        const int cols = static_cast<int>(w.shape[1]);
        const int samples = static_cast<int>(c.shape[0]);
        if (c.shape[1] != cols) {
            throw std::invalid_argument("calibrationData width must equal weight cols");
        }
        const auto analyses = analyze_all_bits(
            static_cast<const float*>(w.ptr), rows, cols,
            static_cast<const float*>(c.ptr), samples
        );
        const auto selected = select_analysis(analyses, threshold);
        return std::unique_ptr<RuntimeMatrix>(new RuntimeMatrix(
            weights, selected.bits, threshold, selected.error,
            parse_decode_policy(decode_policy)
        ));
    }

    static std::unique_ptr<RuntimeMatrix> compress_selected(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int selected_bits,
        double threshold,
        double selected_error,
        const std::string& decode_policy = "hardwareNative"
    ) {
        auto w = weights.request();
        if (w.ndim != 2) {
            throw std::invalid_argument("weights must be 2D");
        }
        validate_storage_bits(selected_bits);
        if (!std::isfinite(threshold) || threshold < 0.0) {
            throw std::invalid_argument("threshold must be a finite non-negative value");
        }
        if (!std::isfinite(selected_error) || selected_error < 0.0) {
            throw std::invalid_argument("selectedError must be a finite non-negative value");
        }
        if (selected_bits <= kMaxCompressedBits && selected_error > threshold) {
            throw std::invalid_argument("selected compressed width does not satisfy threshold");
        }
        if (selected_bits == kFallbackBits) {
            selected_error = 0.0;
        }
        return std::unique_ptr<RuntimeMatrix>(new RuntimeMatrix(
            weights, selected_bits, threshold, selected_error,
            parse_decode_policy(decode_policy)
        ));
    }

    static std::unique_ptr<RuntimeMatrix> load(
        const std::string& path,
        const std::string& decode_policy = "hardwareNative"
    ) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("failed to open ElasticBit file: " + path);
        const std::streamsize file_size = stream.tellg();
        stream.seekg(0, std::ios::beg);

        MLB4Header header{};
        stream.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!stream || std::memcmp(header.magic, "MLB4", 4) != 0) {
            throw std::runtime_error("invalid ElasticBit MLB4 file");
        }
        if (header.version != 4 || header.header_bytes != sizeof(MLB4Header)) {
            throw std::runtime_error("unsupported ElasticBit MLB4 version");
        }
        validate_storage_bits(header.storage_bits);
        if (header.rows == 0 || header.cols == 0) {
            throw std::runtime_error("invalid ElasticBit matrix dimensions");
        }

        const size_t count = static_cast<size_t>(header.rows) * header.cols;
        const size_t expected_payload = header.storage_bits == kFallbackBits
            ? count * sizeof(__half)
            : packed_bytes_for_values(count, header.storage_bits);
        const size_t expected_scales = header.storage_bits == kFallbackBits
            ? 0u : static_cast<size_t>(header.rows);
        const size_t expected_scale_bytes = expected_scales * sizeof(float);
        const size_t expected_file_size = sizeof(MLB4Header) + expected_scale_bytes + expected_payload;

        if (header.scale_count != expected_scales ||
            header.scale_bytes != expected_scale_bytes ||
            header.payload_bytes != expected_payload ||
            header.original_fp16_bytes != count * sizeof(__half) ||
            file_size != static_cast<std::streamsize>(expected_file_size)) {
            throw std::runtime_error("invalid ElasticBit MLB4 metadata");
        }

        std::vector<float> scales(expected_scales);
        if (!scales.empty()) {
            stream.read(reinterpret_cast<char*>(scales.data()), expected_scale_bytes);
        }
        std::vector<uint8_t> payload(expected_payload);
        stream.read(reinterpret_cast<char*>(payload.data()), expected_payload);
        if (!stream) throw std::runtime_error("truncated ElasticBit MLB4 file");

        const uint64_t scale_checksum = scales.empty()
            ? fnv1a64(nullptr, 0)
            : fnv1a64(scales.data(), expected_scale_bytes);
        const uint64_t payload_checksum = fnv1a64(payload.data(), payload.size());
        if (scale_checksum != header.scale_checksum || payload_checksum != header.payload_checksum) {
            throw std::runtime_error("ElasticBit MLB4 checksum mismatch");
        }

        return std::unique_ptr<RuntimeMatrix>(new RuntimeMatrix(
            static_cast<int>(header.rows), static_cast<int>(header.cols),
            static_cast<int>(header.storage_bits), header.threshold,
            header.selected_error, parse_decode_policy(decode_policy),
            std::move(scales), std::move(payload)
        ));
    }

    py::array_t<float> forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }
        CUDA_CHECK(cudaSetDevice(device_id_));
        ensure_host_io();
        std::vector<__half> converted(cols_);
        const float* source = static_cast<const float*>(info.ptr);
        for (int col = 0; col < cols_; ++col) converted[col] = __float2half_rn(source[col]);
        CUDA_CHECK(cudaMemcpy(
            d_host_input_, converted.data(), static_cast<size_t>(cols_) * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
        launch(d_host_input_, d_host_output_, nullptr);
        std::vector<__half> host_output(rows_);
        CUDA_CHECK(cudaMemcpy(
            host_output.data(), d_host_output_, static_cast<size_t>(rows_) * sizeof(__half),
            cudaMemcpyDeviceToHost
        ));
        py::array_t<float> output(rows_);
        auto out = output.request();
        float* out_ptr = static_cast<float*>(out.ptr);
        for (int row = 0; row < rows_; ++row) out_ptr[row] = __half2float(host_output[row]);
        return output;
    }

    torch::Tensor forward_torch(torch::Tensor input) {
        if (!input.is_cuda()) throw std::invalid_argument("forwardTorch requires a CUDA tensor");
        if (input.scalar_type() != torch::kFloat16) {
            throw std::invalid_argument("forwardTorch requires FP16 activations");
        }
        if (!input.is_contiguous()) input = input.contiguous();
        if (input.dim() < 1 || input.size(-1) != cols_ || input.numel() != cols_) {
            throw std::invalid_argument("forwardTorch supports exactly one M=1 activation row");
        }
        if (input.get_device() != device_id_) {
            throw std::invalid_argument("input CUDA device does not match RuntimeMatrix device");
        }

        std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
        shape.back() = rows_;

        if (planned_execution_width_ == 16) {
            // Match the validated direct experiment: W16A16 stays on the
            // framework/vendor FP16 path rather than a hand-written scalar
            // GEMV. materialize_torch() is zero-copy when d_w16_ exists.
            auto weight = materialize_torch();
            auto flat = input.reshape({1, cols_});
            auto output = at::mm(flat, weight.transpose(0, 1));
            return output.reshape(shape);
        }

        auto output = torch::empty(shape, input.options());
        auto stream = c10::cuda::getCurrentCUDAStream(device_id_);
        launch(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            stream.stream()
        );
        return output;
    }

    double benchmark(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        int iterations = 500
    ) {
        if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }
        CUDA_CHECK(cudaSetDevice(device_id_));
        ensure_host_io();
        std::vector<__half> converted(cols_);
        const float* source = static_cast<const float*>(info.ptr);
        for (int col = 0; col < cols_; ++col) converted[col] = __float2half_rn(source[col]);
        CUDA_CHECK(cudaMemcpy(
            d_host_input_, converted.data(), static_cast<size_t>(cols_) * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
        for (int warmup = 0; warmup < 20; ++warmup) launch(d_host_input_, d_host_output_, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) launch(d_host_input_, d_host_output_, nullptr);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return static_cast<double>(ms) / iterations;
    }

    py::array_t<float> dequantize() const {
        const auto values = dequantize_payload_host(payload_, scales_, rows_, cols_, storage_bits_);
        py::array_t<float> output({rows_, cols_});
        auto info = output.request();
        std::memcpy(info.ptr, values.data(), values.size() * sizeof(float));
        return output;
    }

    void set_decode_policy(const std::string& value) {
        const DecodePolicy next = parse_decode_policy(value);
        if (next == decode_policy_) return;
        decode_policy_ = next;
        configure_execution_weights();
    }

    torch::Tensor materialize_torch() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::Device(torch::kCUDA, device_id_));
        const size_t count = static_cast<size_t>(rows_) * cols_;

        if (d_w16_) {
            // Zero-copy view over RuntimeMatrix-owned execution memory.  The
            // no-op deleter is intentional: RuntimeMatrix releases d_w16_.
            // This lets PyTorch dispatch W16A16 through cuBLAS without making
            // a second full-size FP16 execution copy.
            return torch::from_blob(
                static_cast<void*>(d_w16_),
                {static_cast<int64_t>(rows_), static_cast<int64_t>(cols_)},
                [](void*) {},
                options
            );
        }

        auto output = torch::empty({rows_, cols_}, options);
        auto stream = c10::cuda::getCurrentCUDAStream(device_id_);
        __half* out = reinterpret_cast<__half*>(output.data_ptr<at::Half>());

        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        if (d_w4_) {
            const int row_bytes = (cols_ + 1) / 2;
            w4_to_fp16_kernel<<<blocks, threads, 0, stream.stream()>>>(
                d_w4_, d_scales_, out, rows_, cols_, row_bytes
            );
        } else if (d_w8_) {
            w8_to_fp16_kernel<<<blocks, threads, 0, stream.stream()>>>(
                d_w8_, d_scales_, out, rows_, cols_
            );
        } else {
            throw std::runtime_error("ElasticBit has no materialized execution weights");
        }
        CUDA_CHECK(cudaGetLastError());
        return output;
    }

    void save(const std::string& path) const {
        MLB4Header header{};
        std::memcpy(header.magic, "MLB4", 4);
        header.version = 4;
        header.header_bytes = sizeof(MLB4Header);
        header.rows = static_cast<uint32_t>(rows_);
        header.cols = static_cast<uint32_t>(cols_);
        header.storage_bits = static_cast<uint8_t>(storage_bits_);
        header.scale_count = static_cast<uint32_t>(scales_.size());
        header.scale_bytes = scales_.size() * sizeof(float);
        header.payload_bytes = payload_.size();
        header.original_fp16_bytes = original_bytes();
        header.scale_checksum = scales_.empty() ? fnv1a64(nullptr, 0) : fnv1a64(scales_.data(), header.scale_bytes);
        header.payload_checksum = fnv1a64(payload_.data(), payload_.size());
        header.threshold = threshold_;
        header.selected_error = selected_error_;

        std::ofstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("failed to open ElasticBit output: " + path);
        stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (!scales_.empty()) {
            stream.write(reinterpret_cast<const char*>(scales_.data()), header.scale_bytes);
        }
        stream.write(reinterpret_cast<const char*>(payload_.data()), header.payload_bytes);
        if (!stream) throw std::runtime_error("failed while writing ElasticBit output");
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int storage_bits() const { return storage_bits_; }
    int execution_width() const { return planned_execution_width_; }
    std::string execution_planner() const { return execution_planner_; }
    std::string decode_policy() const { return decode_policy_name(decode_policy_); }
    std::string activate_dtype() const { return "float16"; }
    size_t storage_bytes() const { return payload_.size() + scales_.size() * sizeof(float); }
    size_t execution_bytes() const { return execution_bytes_; }
    size_t original_bytes() const { return static_cast<size_t>(rows_) * cols_ * sizeof(__half); }
    double memory_reduction() const {
        return 1.0 - static_cast<double>(storage_bytes()) / static_cast<double>(original_bytes());
    }
    double threshold() const { return threshold_; }
    double error() const { return selected_error_; }
    py::tuple shape() const { return py::make_tuple(rows_, cols_); }

private:
    int rows_ = 0;
    int cols_ = 0;
    int storage_bits_ = kFallbackBits;
    double threshold_ = 0.0;
    double selected_error_ = 0.0;
    DecodePolicy decode_policy_ = DecodePolicy::HardwareNative;
    int device_id_ = 0;
    int planned_execution_width_ = 16;
    std::string execution_planner_ = "fullPrecision";

    std::vector<float> scales_;
    std::vector<uint8_t> payload_;

    uint8_t* d_w4_ = nullptr;
    int8_t* d_w8_ = nullptr;
    __half* d_w16_ = nullptr;
    float* d_scales_ = nullptr;
    __half* d_host_input_ = nullptr;
    __half* d_host_output_ = nullptr;
    size_t execution_bytes_ = 0;

    RuntimeMatrix(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int bits,
        double threshold,
        double selected_error,
        DecodePolicy policy
    ) : storage_bits_(bits), threshold_(threshold), selected_error_(selected_error), decode_policy_(policy) {
        validate_storage_bits(bits);
        auto info = weights.request();
        if (info.ndim != 2) throw std::invalid_argument("weights must be 2D");
        rows_ = static_cast<int>(info.shape[0]);
        cols_ = static_cast<int>(info.shape[1]);
        CUDA_CHECK(cudaGetDevice(&device_id_));
        const float* ptr = static_cast<const float*>(info.ptr);
        scales_ = make_row_scales(ptr, rows_, cols_, storage_bits_);
        payload_ = pack_exact_weights(ptr, rows_, cols_, storage_bits_, scales_);
        configure_execution_weights();
    }

    RuntimeMatrix(
        int rows,
        int cols,
        int bits,
        double threshold,
        double selected_error,
        DecodePolicy policy,
        std::vector<float>&& scales,
        std::vector<uint8_t>&& payload
    ) : rows_(rows), cols_(cols), storage_bits_(bits), threshold_(threshold),
        selected_error_(selected_error), decode_policy_(policy), scales_(std::move(scales)),
        payload_(std::move(payload)) {
        CUDA_CHECK(cudaGetDevice(&device_id_));
        configure_execution_weights();
    }

    void upload_scales() {
        if (scales_.empty()) return;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scales_), scales_.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            d_scales_, scales_.data(), scales_.size() * sizeof(float), cudaMemcpyHostToDevice
        ));
    }

    bool is_validated_t4() const {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
        const std::string name(prop.name);
        return prop.major == 7 && prop.minor == 5 && name.find("T4") != std::string::npos;
    }

    std::vector<int> legal_execution_widths() const {
        if (storage_bits_ <= 4) return {4, 8, 16};
        if (storage_bits_ <= 8) return {8, 16};
        return {16};
    }

    void initialize_execution_weights_for_width(int width) {
        release_weight_memory();
        CUDA_CHECK(cudaSetDevice(device_id_));
        const size_t count = static_cast<size_t>(rows_) * cols_;

        if (width == 16) {
            const auto values = dequantize_payload_host(payload_, scales_, rows_, cols_, storage_bits_);
            std::vector<__half> half_values(count);
            for (size_t i = 0; i < count; ++i) half_values[i] = __float2half_rn(values[i]);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w16_), count * sizeof(__half)));
            CUDA_CHECK(cudaMemcpy(
                d_w16_, half_values.data(), count * sizeof(__half), cudaMemcpyHostToDevice
            ));
            execution_bytes_ = count * sizeof(__half);
            return;
        }

        if (width != 4 && width != 8) {
            throw std::invalid_argument("ElasticBit execution width must be 4, 8, or 16");
        }
        if (width == 4 && storage_bits_ > 4) {
            throw std::invalid_argument("W4A16 cannot represent this ElasticBit storage width");
        }
        if (width == 8 && storage_bits_ > 8) {
            throw std::invalid_argument("W8A16 cannot represent this ElasticBit storage width");
        }

        upload_scales();
        execution_bytes_ = scales_.size() * sizeof(float);
        const int32_t qmax = static_cast<int32_t>(qmax_for_bits(storage_bits_));

        if (width == 4) {
            const int row_bytes = (cols_ + 1) / 2;
            std::vector<uint8_t> packed(static_cast<size_t>(rows_) * row_bytes, 0u);
            for (int row = 0; row < rows_; ++row) {
                for (int col = 0; col < cols_; ++col) {
                    const size_t index = static_cast<size_t>(row) * cols_ + col;
                    const uint32_t code = extract_code_host(payload_.data(), payload_.size(), index, storage_bits_);
                    const int32_t q = static_cast<int32_t>(code) - qmax;
                    const uint8_t nibble = static_cast<uint8_t>(q) & 0x0fu;
                    uint8_t& target = packed[static_cast<size_t>(row) * row_bytes + (col >> 1)];
                    if (col & 1) target |= static_cast<uint8_t>(nibble << 4);
                    else target |= nibble;
                }
            }
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w4_), packed.size()));
            CUDA_CHECK(cudaMemcpy(d_w4_, packed.data(), packed.size(), cudaMemcpyHostToDevice));
            execution_bytes_ += packed.size();
            return;
        }

        std::vector<int8_t> widened(count);
        for (size_t index = 0; index < count; ++index) {
            const uint32_t code = extract_code_host(payload_.data(), payload_.size(), index, storage_bits_);
            widened[index] = static_cast<int8_t>(static_cast<int32_t>(code) - qmax);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w8_), count * sizeof(int8_t)));
        CUDA_CHECK(cudaMemcpy(d_w8_, widened.data(), count * sizeof(int8_t), cudaMemcpyHostToDevice));
        execution_bytes_ += count * sizeof(int8_t);
    }

    double benchmark_current_width(int width) {
        ensure_host_io();
        std::vector<__half> host_input(cols_);
        for (int col = 0; col < cols_; ++col) {
            const float value = static_cast<float>((col % 29) - 14) / 14.0f;
            host_input[col] = __float2half_rn(value);
        }
        CUDA_CHECK(cudaMemcpy(
            d_host_input_, host_input.data(), static_cast<size_t>(cols_) * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
        auto current_stream = c10::cuda::getCurrentCUDAStream(device_id_);

        if (width == 16) {
            // Auto-tune W16 against the same ATen/cuBLAS path used by model
            // decode, not against ElasticBit's compatibility GEMV kernel.
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat16)
                .device(torch::Device(torch::kCUDA, device_id_));
            auto input_view = torch::from_blob(
                static_cast<void*>(d_host_input_), {1, static_cast<int64_t>(cols_)},
                [](void*) {}, options
            );
            auto output_view = torch::from_blob(
                static_cast<void*>(d_host_output_), {1, static_cast<int64_t>(rows_)},
                [](void*) {}, options
            );
            auto weight_view = torch::from_blob(
                static_cast<void*>(d_w16_),
                {static_cast<int64_t>(rows_), static_cast<int64_t>(cols_)},
                [](void*) {}, options
            );
            auto weight_t = weight_view.transpose(0, 1);

            constexpr int warmups = 6;
            constexpr int iterations = 24;
            for (int i = 0; i < warmups; ++i) {
                at::mm_out(output_view, input_view, weight_t);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start, current_stream.stream()));
            for (int i = 0; i < iterations; ++i) {
                at::mm_out(output_view, input_view, weight_t);
            }
            CUDA_CHECK(cudaEventRecord(stop, current_stream.stream()));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return static_cast<double>(milliseconds) / iterations;
        }

        constexpr int warmups = 6;
        constexpr int iterations = 24;
        for (int i = 0; i < warmups; ++i) {
            launch_width(width, d_host_input_, d_host_output_, current_stream.stream());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, current_stream.stream()));
        for (int i = 0; i < iterations; ++i) {
            launch_width(width, d_host_input_, d_host_output_, current_stream.stream());
        }
        CUDA_CHECK(cudaEventRecord(stop, current_stream.stream()));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return static_cast<double>(milliseconds) / iterations;
    }

    int autotune_execution_width() {
        const auto candidates = legal_execution_widths();
        if (candidates.size() == 1) return candidates.front();

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
        const int bucket = storage_bits_ <= 4 ? 4 : storage_bits_ <= 8 ? 8 : 16;
        std::ostringstream keyStream;
        keyStream << device_id_ << ':' << prop.major << '.' << prop.minor << ':'
                  << rows_ << 'x' << cols_ << ':' << bucket;
        const std::string key = keyStream.str();

        static std::mutex cacheMutex;
        static std::unordered_map<std::string, int> cache;
        {
            std::lock_guard<std::mutex> guard(cacheMutex);
            const auto found = cache.find(key);
            if (found != cache.end()) return found->second;
        }

        int best_width = candidates.front();
        double best_ms = std::numeric_limits<double>::infinity();
        for (int width : candidates) {
            initialize_execution_weights_for_width(width);
            const double elapsed = benchmark_current_width(width);
            if (elapsed < best_ms) {
                best_ms = elapsed;
                best_width = width;
            }
        }

        {
            std::lock_guard<std::mutex> guard(cacheMutex);
            cache[key] = best_width;
        }
        return best_width;
    }

    void configure_execution_weights() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        if (decode_policy_ == DecodePolicy::FullPrecision) {
            planned_execution_width_ = 16;
            execution_planner_ = "fullPrecision";
            initialize_execution_weights_for_width(16);
            return;
        }

        if (is_validated_t4()) {
            planned_execution_width_ = execution_width_for_bits(storage_bits_);
            execution_planner_ = "t4Validated";
            initialize_execution_weights_for_width(planned_execution_width_);
            return;
        }

        planned_execution_width_ = autotune_execution_width();
        execution_planner_ = "cudaAutoTune";
        initialize_execution_weights_for_width(planned_execution_width_);
    }

    void ensure_host_io() {
        if (!d_host_input_) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_host_input_), static_cast<size_t>(cols_) * sizeof(__half)));
        }
        if (!d_host_output_) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_host_output_), static_cast<size_t>(rows_) * sizeof(__half)));
        }
    }

    void launch_width(int width, const __half* input, __half* output, cudaStream_t stream) {
        constexpr int threads = 256;
        if (width == 4) {
            const int row_bytes = (cols_ + 1) / 2;
            w4a16_gemv_kernel<<<rows_, threads, 0, stream>>>(
                d_w4_, d_scales_, input, output, rows_, cols_, row_bytes
            );
        } else if (width == 8) {
            w8a16_gemv_kernel<<<rows_, threads, 0, stream>>>(
                d_w8_, d_scales_, input, output, rows_, cols_
            );
        } else {
            w16a16_gemv_kernel<<<rows_, threads, 0, stream>>>(
                d_w16_, input, output, rows_, cols_
            );
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void launch(const __half* input, __half* output, cudaStream_t stream) {
        launch_width(planned_execution_width_, input, output, stream);
    }

    void release_weight_memory() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        if (d_w4_) { cudaFree(d_w4_); d_w4_ = nullptr; }
        if (d_w8_) { cudaFree(d_w8_); d_w8_ = nullptr; }
        if (d_w16_) { cudaFree(d_w16_); d_w16_ = nullptr; }
        if (d_scales_) { cudaFree(d_scales_); d_scales_ = nullptr; }
        execution_bytes_ = 0;
    }

    void release_device_memory() {
        release_weight_memory();
        if (d_host_input_) { cudaFree(d_host_input_); d_host_input_ = nullptr; }
        if (d_host_output_) { cudaFree(d_host_output_); d_host_output_ = nullptr; }
    }
};

static py::dict backend_info_py() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    py::dict result;
    result["device"] = std::string(prop.name);
    result["deviceIndex"] = device;
    result["architecture"] = "sm" + std::to_string(prop.major) + std::to_string(prop.minor);
    result["weightWidths"] = py::make_tuple(4, 8, 16);
    result["activateDType"] = "float16";
    result["prefill"] = "native";
    const std::string name(prop.name);
    const bool validated_t4 = prop.major == 7 && prop.minor == 5 && name.find("T4") != std::string::npos;
    result["decodePlanner"] = validated_t4 ? "t4Validated" : "cudaAutoTune";
    result["validated"] = validated_t4;
    return result;
}

PYBIND11_MODULE(_C, module) {
    module.doc() = "ElasticBit adaptive threshold-driven weight-only CUDA runtime";

    module.def(
        "analyze",
        &analyze_py,
        py::arg("weights"),
        py::arg("calibrationData"),
        py::arg("threshold")
    );
    module.def("backendInfo", &backend_info_py);

    py::class_<RuntimeMatrix, std::unique_ptr<RuntimeMatrix>>(
        module, "RuntimeMatrix", py::module_local()
    )
        .def_static(
            "compress", &RuntimeMatrix::compress,
            py::arg("weights"), py::arg("calibrationData"), py::arg("threshold"),
            py::arg("decodePolicy") = "hardwareNative"
        )
        .def_static(
            "_compressSelected", &RuntimeMatrix::compress_selected,
            py::arg("weights"), py::arg("selectedBits"),
            py::arg("threshold"), py::arg("selectedError"),
            py::arg("decodePolicy") = "hardwareNative"
        )
        .def_static(
            "load", &RuntimeMatrix::load,
            py::arg("path"), py::arg("decodePolicy") = "hardwareNative"
        )
        .def("forward", &RuntimeMatrix::forward)
        .def("forwardTorch", &RuntimeMatrix::forward_torch)
        .def("benchmark", &RuntimeMatrix::benchmark,
            py::arg("input"), py::arg("iterations") = 500)
        .def("dequantize", &RuntimeMatrix::dequantize)
        .def("_materializeTorch", &RuntimeMatrix::materialize_torch)
        .def("setDecodePolicy", &RuntimeMatrix::set_decode_policy)
        .def("save", &RuntimeMatrix::save)
        .def_property_readonly("rows", &RuntimeMatrix::rows)
        .def_property_readonly("cols", &RuntimeMatrix::cols)
        .def_property_readonly("shape", &RuntimeMatrix::shape)
        .def_property_readonly("storageBits", &RuntimeMatrix::storage_bits)
        .def_property_readonly("executionWidth", &RuntimeMatrix::execution_width)
        .def_property_readonly("executionPlanner", &RuntimeMatrix::execution_planner)
        .def_property_readonly("decodePolicy", &RuntimeMatrix::decode_policy)
        .def_property_readonly("activateDType", &RuntimeMatrix::activate_dtype)
        .def_property_readonly("storageBytes", &RuntimeMatrix::storage_bytes)
        .def_property_readonly("executionBytes", &RuntimeMatrix::execution_bytes)
        .def_property_readonly("originalBytes", &RuntimeMatrix::original_bytes)
        .def_property_readonly("memoryReduction", &RuntimeMatrix::memory_reduction)
        .def_property_readonly("threshold", &RuntimeMatrix::threshold)
        .def_property_readonly("error", &RuntimeMatrix::error);
}
