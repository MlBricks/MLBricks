#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            throw std::runtime_error(                                            \
                std::string("CUDA error: ") + cudaGetErrorString(error__) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)             \
            );                                                                  \
        }                                                                       \
    } while (0)

enum class ComputeType : uint8_t {
    INT4 = 1,
    INT8 = 2,
    FP16 = 3,
    FP32 = 4,
};

static const char* compute_type_name(ComputeType type) {
    switch (type) {
        case ComputeType::INT4: return "int4";
        case ComputeType::INT8: return "int8";
        case ComputeType::FP16: return "fp16";
        case ComputeType::FP32: return "fp32";
        default: return "unknown";
    }
}

static ComputeType compute_type_for_bits(int storage_bits) {
    if (storage_bits == 4) {
        return ComputeType::INT4;
    }
    if (storage_bits <= 8) {
        return ComputeType::INT8;
    }
    if (storage_bits <= 16) {
        return ComputeType::FP16;
    }
    return ComputeType::FP32;
}

static void validate_storage_bits(int bits) {
    if (bits < 4 || bits > 32) {
        throw std::invalid_argument(
            "storage_bits must be between 4 and 32"
        );
    }
}

static inline float round_to_fp16_host(float value) {
    __half converted = __float2half_rn(value);
    return __half2float(converted);
}

static inline float fp16_multiply_host(float left, float right) {
    const float left_half = round_to_fp16_host(left);
    const float right_half = round_to_fp16_host(right);
    return round_to_fp16_host(left_half * right_half);
}

static inline uint32_t qmax_for_bits(int bits) {
    return (1u << (bits - 1)) - 1u;
}

static inline size_t packed_bytes_for_values(
    size_t value_count,
    int bits
) {
    return (value_count * static_cast<size_t>(bits) + 7u) / 8u;
}

static std::vector<float> make_row_scales(
    const float* weights,
    int rows,
    int cols,
    int bits
) {
    if (bits == 16 || bits == 32) {
        return {};
    }

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

static inline int32_t quantize_scalar(
    float value,
    float scale,
    int bits
) {
    const int32_t qmax = static_cast<int32_t>(qmax_for_bits(bits));
    int32_t q = static_cast<int32_t>(std::nearbyint(value / scale));
    q = std::max(-qmax, std::min(qmax, q));
    return q;
}

static std::vector<uint8_t> pack_exact_weights(
    const float* weights,
    int rows,
    int cols,
    int bits,
    const std::vector<float>& scales
) {
    const size_t count = static_cast<size_t>(rows) * cols;

    if (bits == 16) {
        std::vector<uint8_t> payload(count * sizeof(__half));
        __half* half_values = reinterpret_cast<__half*>(payload.data());

        for (size_t index = 0; index < count; ++index) {
            half_values[index] = __float2half_rn(weights[index]);
        }

        return payload;
    }

    if (bits == 32) {
        std::vector<uint8_t> payload(count * sizeof(float));
        std::memcpy(payload.data(), weights, payload.size());
        return payload;
    }

    const size_t payload_bytes = packed_bytes_for_values(count, bits);
    std::vector<uint8_t> payload(payload_bytes, 0u);
    const uint32_t qmax = qmax_for_bits(bits);

    for (int row = 0; row < rows; ++row) {
        const float scale = scales[row];

        for (int col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row) * cols + col;
            const int32_t q = quantize_scalar(weights[index], scale, bits);
            const uint32_t code = static_cast<uint32_t>(q + static_cast<int32_t>(qmax));

            const size_t bit_offset = index * static_cast<size_t>(bits);
            const size_t byte_offset = bit_offset >> 3;
            const int shift = static_cast<int>(bit_offset & 7u);
            const uint64_t shifted = static_cast<uint64_t>(code) << shift;

            for (int byte_index = 0; byte_index < 5; ++byte_index) {
                const size_t target = byte_offset + static_cast<size_t>(byte_index);
                if (target < payload.size()) {
                    payload[target] |= static_cast<uint8_t>(
                        (shifted >> (8 * byte_index)) & 0xFFull
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

    uint64_t word = 0u;
    for (int byte_index = 0; byte_index < 5; ++byte_index) {
        const size_t source = byte_offset + static_cast<size_t>(byte_index);
        if (source < payload_bytes) {
            word |= static_cast<uint64_t>(payload[source]) << (8 * byte_index);
        }
    }

    const uint64_t mask = bits == 32
        ? 0xFFFFFFFFull
        : ((1ull << bits) - 1ull);
    return static_cast<uint32_t>((word >> shift) & mask);
}

static std::vector<float> dequantize_payload_host(
    const std::vector<uint8_t>& payload,
    const std::vector<float>& scales,
    int rows,
    int cols,
    int bits,
    bool apply_compute_promotion
) {
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<float> output(count);

    if (bits == 16) {
        const __half* half_values = reinterpret_cast<const __half*>(payload.data());
        for (size_t index = 0; index < count; ++index) {
            output[index] = __half2float(half_values[index]);
        }
        return output;
    }

    if (bits == 32) {
        std::memcpy(output.data(), payload.data(), count * sizeof(float));
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
            float value = static_cast<float>(q) * scale;

            if (apply_compute_promotion && bits >= 9 && bits <= 16) {
                value = round_to_fp16_host(value);
            }

            output[index] = value;
        }
    }

    return output;
}

struct BitAnalysis {
    int bits;
    ComputeType compute_type;
    double error;
    size_t payload_bytes;
    size_t scale_bytes;
};

static std::vector<float> reference_outputs(
    const float* weights,
    int rows,
    int cols,
    const float* calibration,
    int samples
) {
    std::vector<float> outputs(static_cast<size_t>(samples) * rows, 0.0f);

    for (int sample = 0; sample < samples; ++sample) {
        const float* input = calibration + static_cast<size_t>(sample) * cols;
        float* output = outputs.data() + static_cast<size_t>(sample) * rows;

        for (int row = 0; row < rows; ++row) {
            const float* row_ptr = weights + static_cast<size_t>(row) * cols;
            double sum = 0.0;
            for (int col = 0; col < cols; ++col) {
                sum += static_cast<double>(row_ptr[col]) * input[col];
            }
            output[row] = static_cast<float>(sum);
        }
    }

    return outputs;
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

    std::vector<float> scales = make_row_scales(
        weights, rows, cols, bits
    );

    long double numerator = 0.0L;
    long double denominator = 0.0L;

    for (int sample = 0; sample < samples; ++sample) {
        const float* input = calibration + static_cast<size_t>(sample) * cols;

        if (bits <= 8) {
            // Compute promotion:
            // 4-bit storage -> INT4 activation/weight integer math.
            // 5..8-bit storage -> INT8 activation/weight integer math.
            const int32_t input_qmax = bits == 4 ? 7 : 127;
            float input_max_abs = 0.0f;
            for (int col = 0; col < cols; ++col) {
                input_max_abs = std::max(input_max_abs, std::fabs(input[col]));
            }
            const float input_scale = input_max_abs > 0.0f
                ? input_max_abs / static_cast<float>(input_qmax)
                : 1.0f;

            std::vector<int8_t> input_q(cols);
            for (int col = 0; col < cols; ++col) {
                int32_t q = static_cast<int32_t>(
                    std::nearbyint(input[col] / input_scale)
                );
                q = std::max(-input_qmax, std::min(input_qmax, q));
                input_q[col] = static_cast<int8_t>(q);
            }

            for (int row = 0; row < rows; ++row) {
                const float* row_ptr = weights + static_cast<size_t>(row) * cols;
                const float weight_scale = scales[row];
                int64_t integer_sum = 0;

                for (int col = 0; col < cols; ++col) {
                    const int32_t q_weight = quantize_scalar(
                        row_ptr[col], weight_scale, bits
                    );
                    integer_sum += static_cast<int64_t>(q_weight) *
                        static_cast<int32_t>(input_q[col]);
                }

                const double quantized_sum = static_cast<double>(integer_sum) *
                    weight_scale * input_scale;
                const double reference = references[
                    static_cast<size_t>(sample) * rows + row
                ];
                const double difference = quantized_sum - reference;

                numerator += static_cast<long double>(difference) * difference;
                denominator += static_cast<long double>(reference) * reference;
            }
        } else if (bits <= 16) {
            // 9..16-bit storage -> FP16 weight and activation compute.
            std::vector<float> input_half(cols);
            for (int col = 0; col < cols; ++col) {
                input_half[col] = round_to_fp16_host(input[col]);
            }

            for (int row = 0; row < rows; ++row) {
                const float* row_ptr = weights + static_cast<size_t>(row) * cols;
                double quantized_sum = 0.0;

                for (int col = 0; col < cols; ++col) {
                    float weight_value;
                    if (bits == 16) {
                        weight_value = round_to_fp16_host(row_ptr[col]);
                    } else {
                        const int32_t q_weight = quantize_scalar(
                            row_ptr[col], scales[row], bits
                        );
                        weight_value = round_to_fp16_host(
                            static_cast<float>(q_weight) * scales[row]
                        );
                    }

                    quantized_sum += fp16_multiply_host(
                        weight_value, input_half[col]
                    );
                }

                const double reference = references[
                    static_cast<size_t>(sample) * rows + row
                ];
                const double difference = quantized_sum - reference;

                numerator += static_cast<long double>(difference) * difference;
                denominator += static_cast<long double>(reference) * reference;
            }
        } else {
            // 17..32-bit storage -> FP32 weight and activation compute.
            for (int row = 0; row < rows; ++row) {
                const float* row_ptr = weights + static_cast<size_t>(row) * cols;
                float quantized_sum = 0.0f;

                for (int col = 0; col < cols; ++col) {
                    float weight_value;
                    if (bits == 32) {
                        weight_value = row_ptr[col];
                    } else {
                        const int32_t q_weight = quantize_scalar(
                            row_ptr[col], scales[row], bits
                        );
                        weight_value = static_cast<float>(q_weight) * scales[row];
                    }
                    quantized_sum += weight_value * input[col];
                }

                const double reference = references[
                    static_cast<size_t>(sample) * rows + row
                ];
                const double difference = static_cast<double>(quantized_sum) - reference;

                numerator += static_cast<long double>(difference) * difference;
                denominator += static_cast<long double>(reference) * reference;
            }
        }
    }

    const double error = std::sqrt(
        static_cast<double>(numerator / std::max(denominator, 1.0e-30L))
    );

    const size_t count = static_cast<size_t>(rows) * cols;
    const size_t payload_bytes = bits == 16
        ? count * sizeof(__half)
        : (bits == 32
            ? count * sizeof(float)
            : packed_bytes_for_values(count, bits));
    const size_t scale_bytes = (bits == 16 || bits == 32)
        ? 0u
        : static_cast<size_t>(rows) * sizeof(float);

    return BitAnalysis{
        bits,
        compute_type_for_bits(bits),
        error,
        payload_bytes,
        scale_bytes,
    };
}

static std::vector<BitAnalysis> analyze_all_bits(
    const float* weights,
    int rows,
    int cols,
    const float* calibration,
    int samples,
    int min_bits,
    int max_bits
) {
    validate_storage_bits(min_bits);
    validate_storage_bits(max_bits);

    if (min_bits > max_bits) {
        throw std::invalid_argument("min_bits must be <= max_bits");
    }

    const std::vector<float> references = reference_outputs(
        weights, rows, cols, calibration, samples
    );

    std::vector<BitAnalysis> analyses;
    analyses.reserve(static_cast<size_t>(max_bits - min_bits + 1));

    for (int bits = min_bits; bits <= max_bits; ++bits) {
        analyses.push_back(
            analyze_one_bit_width(
                weights,
                rows,
                cols,
                calibration,
                samples,
                references,
                bits
            )
        );
    }

    return analyses;
}

__device__ __forceinline__ uint32_t extract_code_device(
    const uint8_t* payload,
    size_t payload_bytes,
    size_t index,
    int bits
) {
    const size_t bit_offset = index * static_cast<size_t>(bits);
    const size_t byte_offset = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7u);

    uint64_t word = 0u;

    #pragma unroll
    for (int byte_index = 0; byte_index < 5; ++byte_index) {
        const size_t source = byte_offset + static_cast<size_t>(byte_index);
        if (source < payload_bytes) {
            word |= static_cast<uint64_t>(payload[source]) << (8 * byte_index);
        }
    }

    const uint64_t mask = bits == 32
        ? 0xFFFFFFFFull
        : ((1ull << bits) - 1ull);
    return static_cast<uint32_t>((word >> shift) & mask);
}

__global__ void exact_packed_integer_gemv_kernel(
    const uint8_t* __restrict__ payload,
    size_t payload_bytes,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ input_q,
    float input_scale,
    float* __restrict__ output,
    int rows,
    int cols,
    int bits
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t qmax = static_cast<int32_t>((1u << (bits - 1)) - 1u);
    int32_t partial = 0;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        const uint32_t code = extract_code_device(
            payload, payload_bytes, index, bits
        );
        const int32_t q_weight = static_cast<int32_t>(code) - qmax;
        partial += q_weight * static_cast<int32_t>(input_q[col]);
    }

    __shared__ int32_t reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = static_cast<float>(reduction[0]) *
            weight_scales[row] * input_scale;
    }
}

__global__ void direct_int4_gemv_kernel(
    const uint8_t* __restrict__ payload,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ input_q,
    float input_scale,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int32_t partial = 0;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        const uint8_t packed = payload[index >> 1];
        const uint8_t code = (index & 1u)
            ? static_cast<uint8_t>((packed >> 4) & 0x0Fu)
            : static_cast<uint8_t>(packed & 0x0Fu);
        const int32_t q_weight = static_cast<int32_t>(code) - 7;
        partial += q_weight * static_cast<int32_t>(input_q[col]);
    }

    __shared__ int32_t reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = static_cast<float>(reduction[0]) *
            weight_scales[row] * input_scale;
    }
}

__global__ void direct_int8_gemv_kernel(
    const uint8_t* __restrict__ payload,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ input_q,
    float input_scale,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int32_t partial = 0;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        const int32_t q_weight = static_cast<int32_t>(payload[index]) - 127;
        partial += q_weight * static_cast<int32_t>(input_q[col]);
    }

    __shared__ int32_t reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = static_cast<float>(reduction[0]) *
            weight_scales[row] * input_scale;
    }
}

__global__ void exact_packed_fp16_gemv_kernel(
    const uint8_t* __restrict__ payload,
    size_t payload_bytes,
    const float* __restrict__ weight_scales,
    const __half* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int bits
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t qmax = static_cast<int32_t>((1u << (bits - 1)) - 1u);
    float partial = 0.0f;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        const uint32_t code = extract_code_device(
            payload, payload_bytes, index, bits
        );
        const int32_t q_weight = static_cast<int32_t>(code) - qmax;
        const __half weight = __float2half_rn(
            static_cast<float>(q_weight) * weight_scales[row]
        );
        partial += __half2float(__hmul(weight, input[col]));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = reduction[0];
    }
}

__global__ void fp16_gemv_kernel(
    const __half* __restrict__ weights,
    const __half* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float partial = 0.0f;
    const __half* row_ptr = weights + static_cast<size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        partial += __half2float(__hmul(row_ptr[col], input[col]));
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = reduction[0];
    }
}

#pragma pack(push, 1)
struct MLB2Header {
    char magic[4];
    uint16_t version;
    uint16_t header_bytes;
    uint32_t rows;
    uint32_t cols;
    uint8_t storage_bits;
    uint8_t compute_type;
    uint16_t flags;
    uint32_t scale_count;
    uint64_t scale_bytes;
    uint64_t payload_bytes;
    uint64_t original_fp16_bytes;
};
#pragma pack(pop)

static_assert(sizeof(MLB2Header) == 48, "Unexpected MLB2Header size");

class ElasticBitExactMatrixV2 {
public:
    ElasticBitExactMatrixV2(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int storage_bits
    ) {
        initialize_from_weights(weights, storage_bits);
    }

    ~ElasticBitExactMatrixV2() {
        release_device_memory();
    }

    ElasticBitExactMatrixV2(const ElasticBitExactMatrixV2&) = delete;
    ElasticBitExactMatrixV2& operator=(const ElasticBitExactMatrixV2&) = delete;

    static std::unique_ptr<ElasticBitExactMatrixV2> from_auto(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
        double threshold,
        int min_bits = 4,
        int max_bits = 32
    ) {
        const int selected = select_storage_bits(
            weights, calibration, threshold, min_bits, max_bits
        );

        return std::unique_ptr<ElasticBitExactMatrixV2>(
            new ElasticBitExactMatrixV2(weights, selected)
        );
    }

    py::array_t<float> forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        auto input_info = input.request();
        if (input_info.ndim != 1 || input_info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        ensure_io_buffers();
        prepare_selected_input(static_cast<const float*>(input_info.ptr));
        launch_selected_kernel();

        py::array_t<float> output(rows_);
        auto output_info = output.request();

        CUDA_CHECK(cudaMemcpy(
            output_info.ptr,
            d_output_,
            static_cast<size_t>(rows_) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        return output;
    }

    py::array_t<float> forward_fp16(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        auto input_info = input.request();
        if (input_info.ndim != 1 || input_info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        ensure_io_buffers();
        prepare_fp16_input(static_cast<const float*>(input_info.ptr));
        launch_fp16_baseline_kernel();

        py::array_t<float> output(rows_);
        auto output_info = output.request();

        CUDA_CHECK(cudaMemcpy(
            output_info.ptr,
            d_output_,
            static_cast<size_t>(rows_) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        return output;
    }

    py::array_t<float> forward_reference(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) const {
        auto input_info = input.request();
        if (input_info.ndim != 1 || input_info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        const float* input_ptr = static_cast<const float*>(input_info.ptr);
        py::array_t<float> output(rows_);
        auto output_info = output.request();
        float* output_ptr = static_cast<float*>(output_info.ptr);

        if (storage_bits_ <= 8) {
            const int32_t input_qmax = storage_bits_ == 4 ? 7 : 127;
            float input_max_abs = 0.0f;
            for (int col = 0; col < cols_; ++col) {
                input_max_abs = std::max(input_max_abs, std::fabs(input_ptr[col]));
            }
            const float input_scale = input_max_abs > 0.0f
                ? input_max_abs / static_cast<float>(input_qmax)
                : 1.0f;

            std::vector<int8_t> input_q(cols_);
            for (int col = 0; col < cols_; ++col) {
                int32_t q = static_cast<int32_t>(
                    std::nearbyint(input_ptr[col] / input_scale)
                );
                q = std::max(-input_qmax, std::min(input_qmax, q));
                input_q[col] = static_cast<int8_t>(q);
            }

            const int32_t weight_qmax = static_cast<int32_t>(
                qmax_for_bits(storage_bits_)
            );

            for (int row = 0; row < rows_; ++row) {
                int64_t integer_sum = 0;
                for (int col = 0; col < cols_; ++col) {
                    const size_t index = static_cast<size_t>(row) * cols_ + col;
                    const uint32_t code = extract_code_host(
                        payload_.data(), payload_.size(), index, storage_bits_
                    );
                    const int32_t q_weight = static_cast<int32_t>(code) - weight_qmax;
                    integer_sum += static_cast<int64_t>(q_weight) *
                        static_cast<int32_t>(input_q[col]);
                }
                output_ptr[row] = static_cast<float>(integer_sum) *
                    scales_[row] * input_scale;
            }
        } else {
            std::vector<float> input_half(cols_);
            for (int col = 0; col < cols_; ++col) {
                input_half[col] = round_to_fp16_host(input_ptr[col]);
            }

            if (storage_bits_ == 16) {
                const __half* weights_half = reinterpret_cast<const __half*>(
                    payload_.data()
                );
                for (int row = 0; row < rows_; ++row) {
                    float sum = 0.0f;
                    for (int col = 0; col < cols_; ++col) {
                        const size_t index = static_cast<size_t>(row) * cols_ + col;
                        sum += fp16_multiply_host(
                            __half2float(weights_half[index]), input_half[col]
                        );
                    }
                    output_ptr[row] = sum;
                }
            } else {
                const int32_t weight_qmax = static_cast<int32_t>(
                    qmax_for_bits(storage_bits_)
                );
                for (int row = 0; row < rows_; ++row) {
                    float sum = 0.0f;
                    for (int col = 0; col < cols_; ++col) {
                        const size_t index = static_cast<size_t>(row) * cols_ + col;
                        const uint32_t code = extract_code_host(
                            payload_.data(), payload_.size(), index, storage_bits_
                        );
                        const int32_t q_weight = static_cast<int32_t>(code) - weight_qmax;
                        const float weight_half = round_to_fp16_host(
                            static_cast<float>(q_weight) * scales_[row]
                        );
                        sum += fp16_multiply_host(weight_half, input_half[col]);
                    }
                    output_ptr[row] = sum;
                }
            }
        }

        return output;
    }

    py::dict benchmark(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        int iterations = 500
    ) {
        if (iterations <= 0) {
            throw std::invalid_argument("iterations must be positive");
        }

        auto input_info = input.request();
        if (input_info.ndim != 1 || input_info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        ensure_io_buffers();
        prepare_selected_input(static_cast<const float*>(input_info.ptr));
        prepare_fp16_input(static_cast<const float*>(input_info.ptr));

        for (int warmup = 0; warmup < 20; ++warmup) {
            launch_selected_kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch_selected_kernel();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float selected_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&selected_total_ms, start, stop));

        for (int warmup = 0; warmup < 20; ++warmup) {
            launch_fp16_baseline_kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch_fp16_baseline_kernel();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float fp16_total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&fp16_total_ms, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        const double selected_ms = selected_total_ms / iterations;
        const double fp16_ms = fp16_total_ms / iterations;

        py::dict result;
        result["storage_bits"] = storage_bits_;
        result["compute_type"] = compute_type_name(compute_type_);
        result["selected_ms"] = selected_ms;
        result["fp16_ms"] = fp16_ms;
        result["speedup_vs_fp16"] = fp16_ms / selected_ms;
        result["payload_bytes"] = payload_.size();
        result["scale_bytes"] = scales_.size() * sizeof(float);
        result["runtime_weight_bytes"] = runtime_weight_bytes();
        result["fp16_weight_bytes"] = fp16_weight_bytes();
        result["storage_reduction_vs_fp16"] = storage_reduction_vs_fp16();
        return result;
    }

    py::array_t<float> dequantize() const {
        std::vector<float> values = dequantize_payload_host(
            payload_,
            scales_,
            rows_,
            cols_,
            storage_bits_,
            true
        );

        py::array_t<float> output({rows_, cols_});
        auto info = output.request();
        std::memcpy(
            info.ptr,
            values.data(),
            values.size() * sizeof(float)
        );
        return output;
    }

    void save(const std::string& path) const {
        MLB2Header header{};
        std::memcpy(header.magic, "MLB2", 4);
        header.version = 2;
        header.header_bytes = sizeof(MLB2Header);
        header.rows = static_cast<uint32_t>(rows_);
        header.cols = static_cast<uint32_t>(cols_);
        header.storage_bits = static_cast<uint8_t>(storage_bits_);
        header.compute_type = static_cast<uint8_t>(compute_type_);
        header.flags = storage_bits_ == 16 ? 1u : 0u;
        header.scale_count = static_cast<uint32_t>(scales_.size());
        header.scale_bytes = scales_.size() * sizeof(float);
        header.payload_bytes = payload_.size();
        header.original_fp16_bytes = fp16_weight_bytes();

        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("failed to open output file: " + path);
        }

        stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!scales_.empty()) {
            stream.write(
                reinterpret_cast<const char*>(scales_.data()),
                static_cast<std::streamsize>(header.scale_bytes)
            );
        }

        stream.write(
            reinterpret_cast<const char*>(payload_.data()),
            static_cast<std::streamsize>(header.payload_bytes)
        );

        if (!stream) {
            throw std::runtime_error("failed while writing: " + path);
        }
    }

    static std::unique_ptr<ElasticBitExactMatrixV2> load(
        const std::string& path
    ) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("failed to open model file: " + path);
        }

        MLB2Header header{};
        stream.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!stream || std::memcmp(header.magic, "MLB2", 4) != 0) {
            throw std::runtime_error("invalid MLB2 model file");
        }
        if (header.version != 2 || header.header_bytes != sizeof(MLB2Header)) {
            throw std::runtime_error("unsupported MLB2 version");
        }

        validate_storage_bits(header.storage_bits);

        std::vector<float> scales(header.scale_count);
        if (header.scale_bytes != scales.size() * sizeof(float)) {
            throw std::runtime_error("invalid MLB2 scale byte count");
        }

        if (!scales.empty()) {
            stream.read(
                reinterpret_cast<char*>(scales.data()),
                static_cast<std::streamsize>(header.scale_bytes)
            );
        }

        std::vector<uint8_t> payload(header.payload_bytes);
        stream.read(
            reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(header.payload_bytes)
        );

        if (!stream) {
            throw std::runtime_error("truncated MLB2 model file");
        }

        return std::unique_ptr<ElasticBitExactMatrixV2>(
            new ElasticBitExactMatrixV2(
                static_cast<int>(header.rows),
                static_cast<int>(header.cols),
                static_cast<int>(header.storage_bits),
                std::move(scales),
                std::move(payload)
            )
        );
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int storage_bits() const { return storage_bits_; }
    std::string compute_type() const { return compute_type_name(compute_type_); }
    size_t payload_bytes() const { return payload_.size(); }
    size_t scale_bytes() const { return scales_.size() * sizeof(float); }
    size_t runtime_weight_bytes() const {
        return payload_.size() + scales_.size() * sizeof(float);
    }
    size_t fp16_weight_bytes() const {
        return static_cast<size_t>(rows_) * cols_ * sizeof(__half);
    }
    double storage_reduction_vs_fp16() const {
        return 1.0 - static_cast<double>(runtime_weight_bytes()) /
            static_cast<double>(fp16_weight_bytes());
    }

private:
    int rows_ = 0;
    int cols_ = 0;
    int storage_bits_ = 16;
    ComputeType compute_type_ = ComputeType::FP16;

    std::vector<float> scales_;
    std::vector<uint8_t> payload_;
    std::vector<__half> fp16_baseline_;

    uint8_t* d_payload_ = nullptr;
    float* d_scales_ = nullptr;
    __half* d_fp16_baseline_ = nullptr;
    int8_t* d_input_q_ = nullptr;
    __half* d_input_half_ = nullptr;
    float* d_output_ = nullptr;
    float current_input_scale_ = 1.0f;

    ElasticBitExactMatrixV2(
        int rows,
        int cols,
        int storage_bits,
        std::vector<float>&& scales,
        std::vector<uint8_t>&& payload
    ) :
        rows_(rows),
        cols_(cols),
        storage_bits_(storage_bits),
        compute_type_(compute_type_for_bits(storage_bits)),
        scales_(std::move(scales)),
        payload_(std::move(payload))
    {
        const std::vector<float> reconstructed = dequantize_payload_host(
            payload_, scales_, rows_, cols_, storage_bits_, true
        );

        fp16_baseline_.resize(reconstructed.size());
        for (size_t index = 0; index < reconstructed.size(); ++index) {
            fp16_baseline_[index] = __float2half_rn(reconstructed[index]);
        }

        upload_weights();
    }

    void initialize_from_weights(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int storage_bits
    ) {
        validate_storage_bits(storage_bits);

        auto info = weights.request();
        if (info.ndim != 2) {
            throw std::invalid_argument("weights must be a 2D float32 array");
        }

        rows_ = static_cast<int>(info.shape[0]);
        cols_ = static_cast<int>(info.shape[1]);
        storage_bits_ = storage_bits;
        compute_type_ = compute_type_for_bits(storage_bits_);

        const float* weight_ptr = static_cast<const float*>(info.ptr);

        scales_ = make_row_scales(
            weight_ptr, rows_, cols_, storage_bits_
        );
        payload_ = pack_exact_weights(
            weight_ptr, rows_, cols_, storage_bits_, scales_
        );

        const size_t count = static_cast<size_t>(rows_) * cols_;
        fp16_baseline_.resize(count);
        for (size_t index = 0; index < count; ++index) {
            fp16_baseline_[index] = __float2half_rn(weight_ptr[index]);
        }

        upload_weights();
    }

    static int select_storage_bits(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
        double threshold,
        int min_bits,
        int max_bits
    ) {
        auto weight_info = weights.request();
        auto calibration_info = calibration.request();

        if (weight_info.ndim != 2) {
            throw std::invalid_argument("weights must be 2D");
        }
        if (calibration_info.ndim != 2) {
            throw std::invalid_argument("calibration must be 2D");
        }

        const int rows = static_cast<int>(weight_info.shape[0]);
        const int cols = static_cast<int>(weight_info.shape[1]);
        const int samples = static_cast<int>(calibration_info.shape[0]);

        if (calibration_info.shape[1] != cols) {
            throw std::invalid_argument("calibration width must equal weight cols");
        }

        const auto analyses = analyze_all_bits(
            static_cast<const float*>(weight_info.ptr),
            rows,
            cols,
            static_cast<const float*>(calibration_info.ptr),
            samples,
            min_bits,
            max_bits
        );

        for (const auto& analysis : analyses) {
            if (analysis.error <= threshold) {
                return analysis.bits;
            }
        }

        return max_bits;
    }

    void upload_weights() {
        release_device_memory();

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_payload_),
            std::max<size_t>(payload_.size(), 1u)
        ));
        CUDA_CHECK(cudaMemcpy(
            d_payload_, payload_.data(), payload_.size(), cudaMemcpyHostToDevice
        ));

        if (!scales_.empty()) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_scales_),
                scales_.size() * sizeof(float)
            ));
            CUDA_CHECK(cudaMemcpy(
                d_scales_,
                scales_.data(),
                scales_.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ));
        }

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_fp16_baseline_),
            fp16_baseline_.size() * sizeof(__half)
        ));
        CUDA_CHECK(cudaMemcpy(
            d_fp16_baseline_,
            fp16_baseline_.data(),
            fp16_baseline_.size() * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
    }

    void ensure_io_buffers() {
        if (!d_input_q_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_q_),
                static_cast<size_t>(cols_) * sizeof(int8_t)
            ));
        }
        if (!d_input_half_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_half_),
                static_cast<size_t>(cols_) * sizeof(__half)
            ));
        }
        if (!d_output_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_output_),
                static_cast<size_t>(rows_) * sizeof(float)
            ));
        }
    }

    void prepare_selected_input(const float* input) {
        if (storage_bits_ <= 8) {
            const int32_t input_qmax = storage_bits_ == 4 ? 7 : 127;
            float max_abs = 0.0f;
            for (int col = 0; col < cols_; ++col) {
                max_abs = std::max(max_abs, std::fabs(input[col]));
            }
            current_input_scale_ = max_abs > 0.0f
                ? max_abs / static_cast<float>(input_qmax)
                : 1.0f;

            std::vector<int8_t> quantized(cols_);
            for (int col = 0; col < cols_; ++col) {
                int32_t q = static_cast<int32_t>(
                    std::nearbyint(input[col] / current_input_scale_)
                );
                q = std::max(-input_qmax, std::min(input_qmax, q));
                quantized[col] = static_cast<int8_t>(q);
            }

            CUDA_CHECK(cudaMemcpy(
                d_input_q_,
                quantized.data(),
                static_cast<size_t>(cols_) * sizeof(int8_t),
                cudaMemcpyHostToDevice
            ));
        } else {
            prepare_fp16_input(input);
        }
    }

    void prepare_fp16_input(const float* input) {
        std::vector<__half> converted(cols_);
        for (int col = 0; col < cols_; ++col) {
            converted[col] = __float2half_rn(input[col]);
        }
        CUDA_CHECK(cudaMemcpy(
            d_input_half_,
            converted.data(),
            static_cast<size_t>(cols_) * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
    }

    void launch_selected_kernel() {
        constexpr int threads = 256;

        if (storage_bits_ == 16) {
            fp16_gemv_kernel<<<rows_, threads>>>(
                reinterpret_cast<const __half*>(d_payload_),
                d_input_half_,
                d_output_,
                rows_,
                cols_
            );
        } else if (storage_bits_ == 4) {
            direct_int4_gemv_kernel<<<rows_, threads>>>(
                d_payload_,
                d_scales_,
                d_input_q_,
                current_input_scale_,
                d_output_,
                rows_,
                cols_
            );
        } else if (storage_bits_ == 8) {
            direct_int8_gemv_kernel<<<rows_, threads>>>(
                d_payload_,
                d_scales_,
                d_input_q_,
                current_input_scale_,
                d_output_,
                rows_,
                cols_
            );
        } else if (storage_bits_ <= 8) {
            exact_packed_integer_gemv_kernel<<<rows_, threads>>>(
                d_payload_,
                payload_.size(),
                d_scales_,
                d_input_q_,
                current_input_scale_,
                d_output_,
                rows_,
                cols_,
                storage_bits_
            );
        } else {
            exact_packed_fp16_gemv_kernel<<<rows_, threads>>>(
                d_payload_,
                payload_.size(),
                d_scales_,
                d_input_half_,
                d_output_,
                rows_,
                cols_,
                storage_bits_
            );
        }

        CUDA_CHECK(cudaGetLastError());
    }

    void launch_fp16_baseline_kernel() {
        constexpr int threads = 256;
        fp16_gemv_kernel<<<rows_, threads>>>(
            d_fp16_baseline_,
            d_input_half_,
            d_output_,
            rows_,
            cols_
        );
        CUDA_CHECK(cudaGetLastError());
    }

    void release_device_memory() {
        if (d_payload_) {
            cudaFree(d_payload_);
            d_payload_ = nullptr;
        }
        if (d_scales_) {
            cudaFree(d_scales_);
            d_scales_ = nullptr;
        }
        if (d_fp16_baseline_) {
            cudaFree(d_fp16_baseline_);
            d_fp16_baseline_ = nullptr;
        }
        if (d_input_q_) {
            cudaFree(d_input_q_);
            d_input_q_ = nullptr;
        }
        if (d_input_half_) {
            cudaFree(d_input_half_);
            d_input_half_ = nullptr;
        }
        if (d_output_) {
            cudaFree(d_output_);
            d_output_ = nullptr;
        }
    }
};

static py::dict bitsAnaliser_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> weights,
    py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
    double threshold,
    int min_bits = 4,
    int max_bits = 32
) {
    auto weight_info = weights.request();
    auto calibration_info = calibration.request();

    if (weight_info.ndim != 2 || calibration_info.ndim != 2) {
        throw std::invalid_argument("weights and calibration must both be 2D");
    }

    const int rows = static_cast<int>(weight_info.shape[0]);
    const int cols = static_cast<int>(weight_info.shape[1]);
    const int samples = static_cast<int>(calibration_info.shape[0]);

    if (calibration_info.shape[1] != cols) {
        throw std::invalid_argument("calibration width must equal weight cols");
    }

    const auto analyses = analyze_all_bits(
        static_cast<const float*>(weight_info.ptr),
        rows,
        cols,
        static_cast<const float*>(calibration_info.ptr),
        samples,
        min_bits,
        max_bits
    );

    int selected_bits = max_bits;
    double selected_error = analyses.back().error;

    py::list entries;

    for (const auto& analysis : analyses) {
        py::dict entry;
        entry["bits"] = analysis.bits;
        entry["compute_type"] = compute_type_name(analysis.compute_type);
        entry["error"] = analysis.error;
        entry["payload_bytes"] = analysis.payload_bytes;
        entry["scale_bytes"] = analysis.scale_bytes;
        entry["runtime_weight_bytes"] = analysis.payload_bytes + analysis.scale_bytes;
        entry["storage_reduction_vs_fp16"] = 1.0 -
            static_cast<double>(analysis.payload_bytes + analysis.scale_bytes) /
            static_cast<double>(static_cast<size_t>(rows) * cols * sizeof(__half));
        entries.append(entry);

        if (selected_bits == max_bits && analysis.error <= threshold) {
            selected_bits = analysis.bits;
            selected_error = analysis.error;
        }
    }

    // The logic above cannot distinguish "selected max_bits" from
    // "not selected yet". Re-run the simple first-pass deterministically.
    selected_bits = max_bits;
    selected_error = analyses.back().error;
    for (const auto& analysis : analyses) {
        if (analysis.error <= threshold) {
            selected_bits = analysis.bits;
            selected_error = analysis.error;
            break;
        }
    }

    py::dict result;
    result["threshold"] = threshold;
    result["selected_bits"] = selected_bits;
    result["selected_error"] = selected_error;
    result["selected_compute_type"] = compute_type_name(
        compute_type_for_bits(selected_bits)
    );
    result["analyses"] = entries;
    return result;
}



// ================================================================
// ElasticBit production runtime
// - Exact MLB3 storage
// - Compact or fast runtime modes
// - Direct signed-code widening for 5..8 bit fast mode
// - No hidden FP16 benchmark copy in production matrices
// ================================================================


__global__ void fp32_gemv_kernel(
    const float* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float partial = 0.0f;
    const float* row_ptr = weights + static_cast<size_t>(row) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        partial += row_ptr[col] * input[col];
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = reduction[0];
    }
}

__global__ void exact_packed_fp32_gemv_kernel(
    const uint8_t* __restrict__ payload,
    size_t payload_bytes,
    const float* __restrict__ weight_scales,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int bits
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t qmax = static_cast<int32_t>((1u << (bits - 1)) - 1u);
    float partial = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        const uint32_t code = extract_code_device(payload, payload_bytes, index, bits);
        const int32_t q_weight = static_cast<int32_t>(code) - qmax;
        const float weight = static_cast<float>(q_weight) * weight_scales[row];
        partial += weight * input[col];
    }

    __shared__ float reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = reduction[0];
    }
}

__global__ void direct_signed_int8_gemv_kernel(
    const int8_t* __restrict__ weights,
    const float* __restrict__ weight_scales,
    const int8_t* __restrict__ input_q,
    float input_scale,
    float* __restrict__ output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    int32_t partial = 0;
    const size_t row_offset = static_cast<size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        partial += static_cast<int32_t>(weights[row_offset + col]) *
            static_cast<int32_t>(input_q[col]);
    }

    __shared__ int32_t reduction[256];
    reduction[threadIdx.x] = partial;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[row] = static_cast<float>(reduction[0]) *
            weight_scales[row] * input_scale;
    }
}

enum class RuntimeMode : uint8_t {
    Compact = 1,
    Fast = 2,
};

static RuntimeMode parse_runtime_mode(const std::string& mode) {
    if (mode == "compact") {
        return RuntimeMode::Compact;
    }
    if (mode == "fast") {
        return RuntimeMode::Fast;
    }
    throw std::invalid_argument("runtime mode must be 'compact' or 'fast'");
}

static const char* runtime_mode_name(RuntimeMode mode) {
    return mode == RuntimeMode::Compact ? "compact" : "fast";
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
struct MLB3Header {
    char magic[4];
    uint16_t version;
    uint16_t header_bytes;
    uint32_t rows;
    uint32_t cols;
    uint8_t storage_bits;
    uint8_t compute_type;
    uint16_t flags;
    uint32_t scale_count;
    uint64_t scale_bytes;
    uint64_t payload_bytes;
    uint64_t original_fp16_bytes;
    uint64_t scale_checksum;
    uint64_t payload_checksum;
};
#pragma pack(pop)

static_assert(sizeof(MLB3Header) == 64, "Unexpected MLB3Header size");

class RuntimeMatrix {
public:
    RuntimeMatrix(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int storage_bits,
        const std::string& runtime_mode = "compact"
    ) : mode_(parse_runtime_mode(runtime_mode)) {
        initialize_from_weights(weights, storage_bits);
    }

    ~RuntimeMatrix() {
        release_device_memory();
    }

    RuntimeMatrix(const RuntimeMatrix&) = delete;
    RuntimeMatrix& operator=(const RuntimeMatrix&) = delete;

    static std::unique_ptr<RuntimeMatrix> from_auto(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        py::array_t<float, py::array::c_style | py::array::forcecast> calibration,
        double threshold,
        const std::string& runtime_mode = "compact",
        int min_bits = 4,
        int max_bits = 32
    ) {
        auto weight_info = weights.request();
        auto calibration_info = calibration.request();
        if (weight_info.ndim != 2 || calibration_info.ndim != 2) {
            throw std::invalid_argument("weights and calibration must be 2D");
        }
        const int rows = static_cast<int>(weight_info.shape[0]);
        const int cols = static_cast<int>(weight_info.shape[1]);
        const int samples = static_cast<int>(calibration_info.shape[0]);
        if (calibration_info.shape[1] != cols) {
            throw std::invalid_argument("calibration width must equal weight cols");
        }

        const auto analyses = analyze_all_bits(
            static_cast<const float*>(weight_info.ptr),
            rows,
            cols,
            static_cast<const float*>(calibration_info.ptr),
            samples,
            min_bits,
            max_bits
        );

        int selected_bits = max_bits;
        for (const auto& analysis : analyses) {
            if (analysis.error <= threshold) {
                selected_bits = analysis.bits;
                break;
            }
        }

        return std::unique_ptr<RuntimeMatrix>(
            new RuntimeMatrix(weights, selected_bits, runtime_mode)
        );
    }

    static std::unique_ptr<RuntimeMatrix> load(
        const std::string& path,
        const std::string& runtime_mode = "compact"
    ) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("failed to open model file: " + path);
        }

        const std::streamsize file_size = stream.tellg();
        stream.seekg(0, std::ios::beg);

        MLB3Header header{};
        stream.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!stream || std::memcmp(header.magic, "MLB3", 4) != 0) {
            throw std::runtime_error("invalid MLB3 model file");
        }
        if (header.version != 3 || header.header_bytes != sizeof(MLB3Header)) {
            throw std::runtime_error("unsupported MLB3 version");
        }
        if (header.rows == 0 || header.cols == 0) {
            throw std::runtime_error("invalid MLB3 matrix dimensions");
        }

        validate_storage_bits(header.storage_bits);

        const size_t count = static_cast<size_t>(header.rows) * header.cols;
        const size_t expected_payload = header.storage_bits == 16
            ? count * sizeof(__half)
            : (header.storage_bits == 32
                ? count * sizeof(float)
                : packed_bytes_for_values(count, header.storage_bits));
        const size_t expected_scales = (header.storage_bits == 16 || header.storage_bits == 32)
            ? 0u
            : static_cast<size_t>(header.rows);
        const size_t expected_scale_bytes = expected_scales * sizeof(float);
        const size_t expected_file_size = sizeof(MLB3Header) +
            expected_scale_bytes + expected_payload;

        if (header.compute_type != static_cast<uint8_t>(
                compute_type_for_bits(header.storage_bits))) {
            throw std::runtime_error("MLB3 compute type does not match storage bits");
        }
        if (header.scale_count != expected_scales ||
            header.scale_bytes != expected_scale_bytes) {
            throw std::runtime_error("invalid MLB3 scale metadata");
        }
        if (header.payload_bytes != expected_payload) {
            throw std::runtime_error("invalid MLB3 payload size");
        }
        if (header.original_fp16_bytes != count * sizeof(__half)) {
            throw std::runtime_error("invalid MLB3 FP16 reference size");
        }
        if (file_size != static_cast<std::streamsize>(expected_file_size)) {
            throw std::runtime_error("MLB3 file size mismatch or trailing data");
        }

        std::vector<float> scales(expected_scales);
        if (!scales.empty()) {
            stream.read(
                reinterpret_cast<char*>(scales.data()),
                static_cast<std::streamsize>(expected_scale_bytes)
            );
        }

        std::vector<uint8_t> payload(expected_payload);
        stream.read(
            reinterpret_cast<char*>(payload.data()),
            static_cast<std::streamsize>(expected_payload)
        );

        if (!stream) {
            throw std::runtime_error("truncated MLB3 model file");
        }

        const uint64_t scale_checksum = scales.empty()
            ? fnv1a64(nullptr, 0)
            : fnv1a64(scales.data(), expected_scale_bytes);
        const uint64_t payload_checksum = fnv1a64(
            payload.data(), payload.size()
        );

        if (scale_checksum != header.scale_checksum ||
            payload_checksum != header.payload_checksum) {
            throw std::runtime_error("MLB3 checksum mismatch");
        }

        return std::unique_ptr<RuntimeMatrix>(
            new RuntimeMatrix(
                static_cast<int>(header.rows),
                static_cast<int>(header.cols),
                static_cast<int>(header.storage_bits),
                parse_runtime_mode(runtime_mode),
                std::move(scales),
                std::move(payload)
            )
        );
    }

    py::array_t<float> forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        ensure_io_buffers();
        prepare_input(static_cast<const float*>(info.ptr));
        launch_kernel();

        py::array_t<float> output(rows_);
        auto output_info = output.request();
        CUDA_CHECK(cudaMemcpy(
            output_info.ptr,
            d_output_,
            static_cast<size_t>(rows_) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        return output;
    }

    py::array_t<float> forward_reference(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) const {
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }
        const float* input_ptr = static_cast<const float*>(info.ptr);

        py::array_t<float> output(rows_);
        auto output_info = output.request();
        float* output_ptr = static_cast<float*>(output_info.ptr);

        if (storage_bits_ <= 8) {
            const int32_t input_qmax = storage_bits_ == 4 ? 7 : 127;
            float input_max_abs = 0.0f;
            for (int col = 0; col < cols_; ++col) {
                input_max_abs = std::max(input_max_abs, std::fabs(input_ptr[col]));
            }
            const float input_scale = input_max_abs > 0.0f
                ? input_max_abs / static_cast<float>(input_qmax)
                : 1.0f;
            std::vector<int8_t> input_q(cols_);
            for (int col = 0; col < cols_; ++col) {
                int32_t q = static_cast<int32_t>(
                    std::nearbyint(input_ptr[col] / input_scale)
                );
                q = std::max(-input_qmax, std::min(input_qmax, q));
                input_q[col] = static_cast<int8_t>(q);
            }

            const int32_t weight_qmax = static_cast<int32_t>(
                qmax_for_bits(storage_bits_)
            );
            for (int row = 0; row < rows_; ++row) {
                int64_t sum = 0;
                for (int col = 0; col < cols_; ++col) {
                    const size_t index = static_cast<size_t>(row) * cols_ + col;
                    const uint32_t code = extract_code_host(
                        payload_.data(), payload_.size(), index, storage_bits_
                    );
                    const int32_t q_weight = static_cast<int32_t>(code) - weight_qmax;
                    sum += static_cast<int64_t>(q_weight) * input_q[col];
                }
                output_ptr[row] = static_cast<float>(sum) *
                    scales_[row] * input_scale;
            }
        } else if (storage_bits_ <= 16) {
            std::vector<float> input_half(cols_);
            for (int col = 0; col < cols_; ++col) {
                input_half[col] = round_to_fp16_host(input_ptr[col]);
            }
            const std::vector<float> weights = dequantize_payload_host(
                payload_, scales_, rows_, cols_, storage_bits_, true
            );
            for (int row = 0; row < rows_; ++row) {
                float sum = 0.0f;
                for (int col = 0; col < cols_; ++col) {
                    const size_t index = static_cast<size_t>(row) * cols_ + col;
                    sum += fp16_multiply_host(weights[index], input_half[col]);
                }
                output_ptr[row] = sum;
            }
        } else {
            const std::vector<float> weights = dequantize_payload_host(
                payload_, scales_, rows_, cols_, storage_bits_, false
            );
            for (int row = 0; row < rows_; ++row) {
                float sum = 0.0f;
                for (int col = 0; col < cols_; ++col) {
                    const size_t index = static_cast<size_t>(row) * cols_ + col;
                    sum += weights[index] * input_ptr[col];
                }
                output_ptr[row] = sum;
            }
        }
        return output;
    }

    double benchmark(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        int iterations = 500
    ) {
        if (iterations <= 0) {
            throw std::invalid_argument("iterations must be positive");
        }
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }

        ensure_io_buffers();
        prepare_input(static_cast<const float*>(info.ptr));
        for (int warmup = 0; warmup < 20; ++warmup) {
            launch_kernel();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch_kernel();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return static_cast<double>(total_ms) / iterations;
    }

    py::array_t<float> dequantize() const {
        const std::vector<float> values = dequantize_payload_host(
            payload_, scales_, rows_, cols_, storage_bits_, true
        );
        py::array_t<float> output({rows_, cols_});
        auto info = output.request();
        std::memcpy(info.ptr, values.data(), values.size() * sizeof(float));
        return output;
    }

    void save(const std::string& path) const {
        MLB3Header header{};
        std::memcpy(header.magic, "MLB3", 4);
        header.version = 3;
        header.header_bytes = sizeof(MLB3Header);
        header.rows = static_cast<uint32_t>(rows_);
        header.cols = static_cast<uint32_t>(cols_);
        header.storage_bits = static_cast<uint8_t>(storage_bits_);
        header.compute_type = static_cast<uint8_t>(compute_type_);
        header.flags = 1u;
        header.scale_count = static_cast<uint32_t>(scales_.size());
        header.scale_bytes = scales_.size() * sizeof(float);
        header.payload_bytes = payload_.size();
        header.original_fp16_bytes = fp16_weight_bytes();
        header.scale_checksum = scales_.empty()
            ? fnv1a64(nullptr, 0)
            : fnv1a64(scales_.data(), header.scale_bytes);
        header.payload_checksum = fnv1a64(payload_.data(), payload_.size());

        std::ofstream stream(path, std::ios::binary);
        if (!stream) {
            throw std::runtime_error("failed to open output file: " + path);
        }
        stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
        if (!scales_.empty()) {
            stream.write(
                reinterpret_cast<const char*>(scales_.data()),
                static_cast<std::streamsize>(header.scale_bytes)
            );
        }
        stream.write(
            reinterpret_cast<const char*>(payload_.data()),
            static_cast<std::streamsize>(header.payload_bytes)
        );
        if (!stream) {
            throw std::runtime_error("failed while writing: " + path);
        }
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int storage_bits() const { return storage_bits_; }
    std::string compute_type() const { return compute_type_name(compute_type_); }
    std::string runtime_mode() const { return runtime_mode_name(mode_); }
    size_t file_payload_bytes() const { return payload_.size(); }
    size_t file_scale_bytes() const { return scales_.size() * sizeof(float); }
    size_t file_weight_bytes() const { return file_payload_bytes() + file_scale_bytes(); }
    size_t fp16_weight_bytes() const {
        return static_cast<size_t>(rows_) * cols_ * sizeof(__half);
    }
    size_t gpu_weight_bytes() const { return gpu_weight_bytes_; }
    double file_reduction_vs_fp16() const {
        return 1.0 - static_cast<double>(file_weight_bytes()) /
            static_cast<double>(fp16_weight_bytes());
    }
    double gpu_reduction_vs_fp16() const {
        return 1.0 - static_cast<double>(gpu_weight_bytes_) /
            static_cast<double>(fp16_weight_bytes());
    }

private:
    int rows_ = 0;
    int cols_ = 0;
    int storage_bits_ = 16;
    ComputeType compute_type_ = ComputeType::FP16;
    RuntimeMode mode_ = RuntimeMode::Compact;

    std::vector<float> scales_;
    std::vector<uint8_t> payload_;

    uint8_t* d_compact_payload_ = nullptr;
    float* d_scales_ = nullptr;
    int8_t* d_fast_int8_ = nullptr;
    __half* d_fast_half_ = nullptr;
    float* d_fast_float_ = nullptr;
    int8_t* d_input_q_ = nullptr;
    __half* d_input_half_ = nullptr;
    float* d_input_float_ = nullptr;
    float* d_output_ = nullptr;
    float current_input_scale_ = 1.0f;
    size_t gpu_weight_bytes_ = 0;

    RuntimeMatrix(
        int rows,
        int cols,
        int storage_bits,
        RuntimeMode mode,
        std::vector<float>&& scales,
        std::vector<uint8_t>&& payload
    ) :
        rows_(rows),
        cols_(cols),
        storage_bits_(storage_bits),
        compute_type_(compute_type_for_bits(storage_bits)),
        mode_(mode),
        scales_(std::move(scales)),
        payload_(std::move(payload))
    {
        initialize_runtime_weights();
    }

    void initialize_from_weights(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights,
        int storage_bits
    ) {
        validate_storage_bits(storage_bits);
        auto info = weights.request();
        if (info.ndim != 2) {
            throw std::invalid_argument("weights must be a 2D float32 array");
        }
        rows_ = static_cast<int>(info.shape[0]);
        cols_ = static_cast<int>(info.shape[1]);
        storage_bits_ = storage_bits;
        compute_type_ = compute_type_for_bits(storage_bits_);
        const float* weight_ptr = static_cast<const float*>(info.ptr);
        scales_ = make_row_scales(weight_ptr, rows_, cols_, storage_bits_);
        payload_ = pack_exact_weights(
            weight_ptr, rows_, cols_, storage_bits_, scales_
        );
        initialize_runtime_weights();
    }

    void initialize_runtime_weights() {
        release_weight_memory();
        const size_t count = static_cast<size_t>(rows_) * cols_;

        if (mode_ == RuntimeMode::Compact) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_compact_payload_),
                std::max<size_t>(payload_.size(), 1u)
            ));
            CUDA_CHECK(cudaMemcpy(
                d_compact_payload_, payload_.data(), payload_.size(),
                cudaMemcpyHostToDevice
            ));
            gpu_weight_bytes_ = payload_.size();
            if (!scales_.empty()) {
                upload_scales();
                gpu_weight_bytes_ += scales_.size() * sizeof(float);
            }
            return;
        }

        if (storage_bits_ == 4) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_compact_payload_),
                std::max<size_t>(payload_.size(), 1u)
            ));
            CUDA_CHECK(cudaMemcpy(
                d_compact_payload_, payload_.data(), payload_.size(),
                cudaMemcpyHostToDevice
            ));
            upload_scales();
            gpu_weight_bytes_ = payload_.size() + scales_.size() * sizeof(float);
            return;
        }

        if (storage_bits_ <= 8) {
            std::vector<int8_t> widened(count);
            const int32_t qmax = static_cast<int32_t>(qmax_for_bits(storage_bits_));
            for (size_t index = 0; index < count; ++index) {
                const uint32_t code = extract_code_host(
                    payload_.data(), payload_.size(), index, storage_bits_
                );
                widened[index] = static_cast<int8_t>(
                    static_cast<int32_t>(code) - qmax
                );
            }
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_fast_int8_),
                count * sizeof(int8_t)
            ));
            CUDA_CHECK(cudaMemcpy(
                d_fast_int8_, widened.data(), count * sizeof(int8_t),
                cudaMemcpyHostToDevice
            ));
            upload_scales();
            gpu_weight_bytes_ = count * sizeof(int8_t) +
                scales_.size() * sizeof(float);
            return;
        }

        if (storage_bits_ <= 16) {
            std::vector<__half> widened_half(count);
            if (storage_bits_ == 16) {
                std::memcpy(
                    widened_half.data(), payload_.data(), count * sizeof(__half)
                );
            } else {
                const std::vector<float> values = dequantize_payload_host(
                    payload_, scales_, rows_, cols_, storage_bits_, true
                );
                for (size_t index = 0; index < count; ++index) {
                    widened_half[index] = __float2half_rn(values[index]);
                }
            }
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_fast_half_),
                count * sizeof(__half)
            ));
            CUDA_CHECK(cudaMemcpy(
                d_fast_half_, widened_half.data(), count * sizeof(__half),
                cudaMemcpyHostToDevice
            ));
            gpu_weight_bytes_ = count * sizeof(__half);
            return;
        }

        std::vector<float> widened_float(count);
        if (storage_bits_ == 32) {
            std::memcpy(widened_float.data(), payload_.data(), count * sizeof(float));
        } else {
            widened_float = dequantize_payload_host(
                payload_, scales_, rows_, cols_, storage_bits_, false
            );
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_fast_float_),
            count * sizeof(float)
        ));
        CUDA_CHECK(cudaMemcpy(
            d_fast_float_, widened_float.data(), count * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        gpu_weight_bytes_ = count * sizeof(float);
    }

    void upload_scales() {
        if (scales_.empty()) {
            return;
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_scales_),
            scales_.size() * sizeof(float)
        ));
        CUDA_CHECK(cudaMemcpy(
            d_scales_, scales_.data(), scales_.size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }

    void ensure_io_buffers() {
        if (!d_input_q_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_q_),
                static_cast<size_t>(cols_) * sizeof(int8_t)
            ));
        }
        if (!d_input_half_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_half_),
                static_cast<size_t>(cols_) * sizeof(__half)
            ));
        }
        if (!d_input_float_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_float_),
                static_cast<size_t>(cols_) * sizeof(float)
            ));
        }
        if (!d_output_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_output_),
                static_cast<size_t>(rows_) * sizeof(float)
            ));
        }
    }

    void prepare_input(const float* input) {
        if (storage_bits_ <= 8) {
            const int32_t input_qmax = storage_bits_ == 4 ? 7 : 127;
            float max_abs = 0.0f;
            for (int col = 0; col < cols_; ++col) {
                max_abs = std::max(max_abs, std::fabs(input[col]));
            }
            current_input_scale_ = max_abs > 0.0f
                ? max_abs / static_cast<float>(input_qmax)
                : 1.0f;
            std::vector<int8_t> quantized(cols_);
            for (int col = 0; col < cols_; ++col) {
                int32_t q = static_cast<int32_t>(
                    std::nearbyint(input[col] / current_input_scale_)
                );
                q = std::max(-input_qmax, std::min(input_qmax, q));
                quantized[col] = static_cast<int8_t>(q);
            }
            CUDA_CHECK(cudaMemcpy(
                d_input_q_, quantized.data(), static_cast<size_t>(cols_) * sizeof(int8_t),
                cudaMemcpyHostToDevice
            ));
        } else if (storage_bits_ <= 16) {
            std::vector<__half> converted(cols_);
            for (int col = 0; col < cols_; ++col) {
                converted[col] = __float2half_rn(input[col]);
            }
            CUDA_CHECK(cudaMemcpy(
                d_input_half_, converted.data(), static_cast<size_t>(cols_) * sizeof(__half),
                cudaMemcpyHostToDevice
            ));
        } else {
            CUDA_CHECK(cudaMemcpy(
                d_input_float_, input, static_cast<size_t>(cols_) * sizeof(float),
                cudaMemcpyHostToDevice
            ));
        }
    }

    void launch_kernel() {
        constexpr int threads = 256;
        if (mode_ == RuntimeMode::Fast) {
            if (storage_bits_ == 4) {
                direct_int4_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, d_scales_, d_input_q_, current_input_scale_,
                    d_output_, rows_, cols_
                );
            } else if (storage_bits_ <= 8) {
                direct_signed_int8_gemv_kernel<<<rows_, threads>>>(
                    d_fast_int8_, d_scales_, d_input_q_, current_input_scale_,
                    d_output_, rows_, cols_
                );
            } else if (storage_bits_ <= 16) {
                fp16_gemv_kernel<<<rows_, threads>>>(
                    d_fast_half_, d_input_half_, d_output_, rows_, cols_
                );
            } else {
                fp32_gemv_kernel<<<rows_, threads>>>(
                    d_fast_float_, d_input_float_, d_output_, rows_, cols_
                );
            }
        } else {
            if (storage_bits_ == 16) {
                fp16_gemv_kernel<<<rows_, threads>>>(
                    reinterpret_cast<const __half*>(d_compact_payload_),
                    d_input_half_, d_output_, rows_, cols_
                );
            } else if (storage_bits_ == 32) {
                fp32_gemv_kernel<<<rows_, threads>>>(
                    reinterpret_cast<const float*>(d_compact_payload_),
                    d_input_float_, d_output_, rows_, cols_
                );
            } else if (storage_bits_ == 4) {
                direct_int4_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, d_scales_, d_input_q_, current_input_scale_,
                    d_output_, rows_, cols_
                );
            } else if (storage_bits_ == 8) {
                direct_int8_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, d_scales_, d_input_q_, current_input_scale_,
                    d_output_, rows_, cols_
                );
            } else if (storage_bits_ <= 8) {
                exact_packed_integer_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, payload_.size(), d_scales_, d_input_q_,
                    current_input_scale_, d_output_, rows_, cols_, storage_bits_
                );
            } else if (storage_bits_ <= 16) {
                exact_packed_fp16_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, payload_.size(), d_scales_, d_input_half_,
                    d_output_, rows_, cols_, storage_bits_
                );
            } else {
                exact_packed_fp32_gemv_kernel<<<rows_, threads>>>(
                    d_compact_payload_, payload_.size(), d_scales_, d_input_float_,
                    d_output_, rows_, cols_, storage_bits_
                );
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void release_weight_memory() {
        if (d_compact_payload_) {
            cudaFree(d_compact_payload_);
            d_compact_payload_ = nullptr;
        }
        if (d_scales_) {
            cudaFree(d_scales_);
            d_scales_ = nullptr;
        }
        if (d_fast_int8_) {
            cudaFree(d_fast_int8_);
            d_fast_int8_ = nullptr;
        }
        if (d_fast_half_) {
            cudaFree(d_fast_half_);
            d_fast_half_ = nullptr;
        }
        if (d_fast_float_) {
            cudaFree(d_fast_float_);
            d_fast_float_ = nullptr;
        }
        gpu_weight_bytes_ = 0;
    }

    void release_device_memory() {
        release_weight_memory();
        if (d_input_q_) {
            cudaFree(d_input_q_);
            d_input_q_ = nullptr;
        }
        if (d_input_half_) {
            cudaFree(d_input_half_);
            d_input_half_ = nullptr;
        }
        if (d_input_float_) {
            cudaFree(d_input_float_);
            d_input_float_ = nullptr;
        }
        if (d_output_) {
            cudaFree(d_output_);
            d_output_ = nullptr;
        }
    }
};

class NativeFP16Matrix {
public:
    explicit NativeFP16Matrix(
        py::array_t<float, py::array::c_style | py::array::forcecast> weights
    ) {
        auto info = weights.request();
        if (info.ndim != 2) {
            throw std::invalid_argument("weights must be a 2D float32 array");
        }
        rows_ = static_cast<int>(info.shape[0]);
        cols_ = static_cast<int>(info.shape[1]);
        const size_t count = static_cast<size_t>(rows_) * cols_;
        const float* source = static_cast<const float*>(info.ptr);
        std::vector<__half> converted(count);
        for (size_t index = 0; index < count; ++index) {
            converted[index] = __float2half_rn(source[index]);
        }
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_weights_), count * sizeof(__half)
        ));
        CUDA_CHECK(cudaMemcpy(
            d_weights_, converted.data(), count * sizeof(__half), cudaMemcpyHostToDevice
        ));
    }

    ~NativeFP16Matrix() {
        if (d_weights_) cudaFree(d_weights_);
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
    }

    py::array_t<float> forward(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        prepare(input);
        launch();
        py::array_t<float> output(rows_);
        auto info = output.request();
        CUDA_CHECK(cudaMemcpy(
            info.ptr, d_output_, static_cast<size_t>(rows_) * sizeof(float),
            cudaMemcpyDeviceToHost
        ));
        return output;
    }

    double benchmark(
        py::array_t<float, py::array::c_style | py::array::forcecast> input,
        int iterations = 500
    ) {
        if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
        prepare(input);
        for (int warmup = 0; warmup < 20; ++warmup) launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start;
        cudaEvent_t stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int iteration = 0; iteration < iterations; ++iteration) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return static_cast<double>(total_ms) / iterations;
    }

    size_t gpu_weight_bytes() const {
        return static_cast<size_t>(rows_) * cols_ * sizeof(__half);
    }

private:
    int rows_ = 0;
    int cols_ = 0;
    __half* d_weights_ = nullptr;
    __half* d_input_ = nullptr;
    float* d_output_ = nullptr;

    void prepare(
        py::array_t<float, py::array::c_style | py::array::forcecast> input
    ) {
        auto info = input.request();
        if (info.ndim != 1 || info.shape[0] != cols_) {
            throw std::invalid_argument("input must have shape [cols]");
        }
        if (!d_input_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_input_),
                static_cast<size_t>(cols_) * sizeof(__half)
            ));
        }
        if (!d_output_) {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_output_),
                static_cast<size_t>(rows_) * sizeof(float)
            ));
        }
        const float* source = static_cast<const float*>(info.ptr);
        std::vector<__half> converted(cols_);
        for (int col = 0; col < cols_; ++col) {
            converted[col] = __float2half_rn(source[col]);
        }
        CUDA_CHECK(cudaMemcpy(
            d_input_, converted.data(), static_cast<size_t>(cols_) * sizeof(__half),
            cudaMemcpyHostToDevice
        ));
    }

    void launch() {
        constexpr int threads = 256;
        fp16_gemv_kernel<<<rows_, threads>>>(
            d_weights_, d_input_, d_output_, rows_, cols_
        );
        CUDA_CHECK(cudaGetLastError());
    }
};

PYBIND11_MODULE(_C, module) {
    module.doc() = "ElasticBit exact-storage CUDA runtime";

    module.def(
        "bitsAnaliser",
        &bitsAnaliser_py,
        py::arg("weights"),
        py::arg("calibration"),
        py::arg("threshold"),
        py::arg("min_bits") = 4,
        py::arg("max_bits") = 32
    );

    py::class_<
        RuntimeMatrix,
        std::unique_ptr<RuntimeMatrix>
    >(
        module,
        "RuntimeMatrix",
        py::module_local()
    )
        .def(
            py::init<
                py::array_t<float, py::array::c_style | py::array::forcecast>,
                int,
                const std::string&
            >(),
            py::arg("weights"),
            py::arg("storage_bits"),
            py::arg("runtime_mode") = "compact"
        )
        .def_static(
            "from_auto",
            &RuntimeMatrix::from_auto,
            py::arg("weights"),
            py::arg("calibration"),
            py::arg("threshold"),
            py::arg("runtime_mode") = "compact",
            py::arg("min_bits") = 4,
            py::arg("max_bits") = 32
        )
        .def_static(
            "load",
            &RuntimeMatrix::load,
            py::arg("path"),
            py::arg("runtime_mode") = "compact"
        )
        .def("forward", &RuntimeMatrix::forward)
        .def("forward_reference", &RuntimeMatrix::forward_reference)
        .def("benchmark", &RuntimeMatrix::benchmark,
            py::arg("input"), py::arg("iterations") = 500)
        .def("dequantize", &RuntimeMatrix::dequantize)
        .def("save", &RuntimeMatrix::save)
        .def_property_readonly("rows", &RuntimeMatrix::rows)
        .def_property_readonly("cols", &RuntimeMatrix::cols)
        .def_property_readonly("storage_bits", &RuntimeMatrix::storage_bits)
        .def_property_readonly("compute_type", &RuntimeMatrix::compute_type)
        .def_property_readonly("runtime_mode", &RuntimeMatrix::runtime_mode)
        .def_property_readonly("file_payload_bytes", &RuntimeMatrix::file_payload_bytes)
        .def_property_readonly("file_scale_bytes", &RuntimeMatrix::file_scale_bytes)
        .def_property_readonly("file_weight_bytes", &RuntimeMatrix::file_weight_bytes)
        .def_property_readonly("gpu_weight_bytes", &RuntimeMatrix::gpu_weight_bytes)
        .def_property_readonly("fp16_weight_bytes", &RuntimeMatrix::fp16_weight_bytes)
        .def_property_readonly("file_reduction_vs_fp16", &RuntimeMatrix::file_reduction_vs_fp16)
        .def_property_readonly("gpu_reduction_vs_fp16", &RuntimeMatrix::gpu_reduction_vs_fp16);

    py::class_<NativeFP16Matrix, std::unique_ptr<NativeFP16Matrix>>(
        module,
        "NativeFP16Matrix",
        py::module_local()
    )
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>
        >(), py::arg("weights"))
        .def("forward", &NativeFP16Matrix::forward)
        .def("benchmark", &NativeFP16Matrix::benchmark,
            py::arg("input"), py::arg("iterations") = 500)
        .def_property_readonly("gpu_weight_bytes", &NativeFP16Matrix::gpu_weight_bytes);


}
