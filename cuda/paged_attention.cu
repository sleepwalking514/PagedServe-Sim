#include "serving/cuda_bench.hpp"

#include <cuda_fp16.h>
#include <math_constants.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace serving {
namespace {

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void paged_attention_decode_kernel(const __half* query, const __half* key_cache,
                                              const __half* value_cache, const int* block_table,
                                              const int* sequence_lengths, __half* output, int query_heads,
                                              int kv_heads, int head_dim, int block_tokens, int max_blocks,
                                              float scale) {
    extern __shared__ float query_tile[];
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int batch = row / query_heads;
    const int query_head = row % query_heads;
    const int sequence_length = sequence_lengths[batch];
    const int kv_head = query_head / (query_heads / kv_heads);
    const __half* query_row = query + (static_cast<std::size_t>(batch) * query_heads + query_head) * head_dim;

    for (int dimension = lane; dimension < head_dim; dimension += 32) {
        query_tile[dimension] = __half2float(query_row[dimension]);
    }
    __syncthreads();

    float running_max = -CUDART_INF_F;
    float running_denom = 0.0F;
    float output_tile[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int values_per_lane = (head_dim + 31) / 32;

    for (int token = 0; token < sequence_length; ++token) {
        const int logical_block = token / block_tokens;
        const int offset = token % block_tokens;
        const int physical_block = block_table[batch * max_blocks + logical_block];
        const std::size_t cache_index =
            (static_cast<std::size_t>(physical_block) * kv_heads + kv_head) * block_tokens * head_dim +
            static_cast<std::size_t>(offset) * head_dim;
        const __half* key_row = key_cache + cache_index;
        const __half* value_row = value_cache + cache_index;

        float dot = 0.0F;
        for (int dimension = lane; dimension < head_dim; dimension += 32) {
            dot += query_tile[dimension] * __half2float(key_row[dimension]);
        }
        dot = warp_sum(dot) * scale;
        dot = __shfl_sync(0xffffffffU, dot, 0);

        const float next_max = fmaxf(running_max, dot);
        const float old_scale = isfinite(running_max) ? expf(running_max - next_max) : 0.0F;
        const float token_weight = expf(dot - next_max);
        running_denom = running_denom * old_scale + token_weight;
        running_max = next_max;

        for (int value_index = 0; value_index < values_per_lane; ++value_index) {
            const int dimension = lane + value_index * 32;
            if (dimension < head_dim) {
                output_tile[value_index] = output_tile[value_index] * old_scale +
                                           token_weight * __half2float(value_row[dimension]);
            }
        }
    }

    __half* output_row = output + (static_cast<std::size_t>(batch) * query_heads + query_head) * head_dim;
    for (int value_index = 0; value_index < values_per_lane; ++value_index) {
        const int dimension = lane + value_index * 32;
        if (dimension < head_dim) {
            output_row[dimension] = __float2half(output_tile[value_index] / running_denom);
        }
    }
}

void check_cuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

}  // namespace

bool cuda_benchmark_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void write_cuda_decode_profile(const std::filesystem::path& output_path, const CudaBenchmarkConfig& config) {
    if (!cuda_benchmark_available()) {
        throw std::runtime_error("No CUDA device is available");
    }
    if (config.query_heads == 0 || config.kv_heads == 0 || config.query_heads % config.kv_heads != 0 ||
        config.head_dim == 0 || config.head_dim > 128 || config.block_tokens == 0) {
        throw std::invalid_argument("Invalid CUDA benchmark shape");
    }

    constexpr std::array<int, 5> batches = {1, 4, 8, 16, 32};
    constexpr std::array<int, 3> sequence_lengths = {128, 512, 1'024};
    const int max_sequence_length = sequence_lengths.back();
    const int max_blocks = (max_sequence_length + static_cast<int>(config.block_tokens) - 1) /
                           static_cast<int>(config.block_tokens);
    const std::size_t cache_elements = static_cast<std::size_t>(max_blocks) * config.kv_heads * config.block_tokens *
                                       config.head_dim;

    std::mt19937 generator(7);
    std::uniform_real_distribution<float> random_value(-0.05F, 0.05F);
    std::vector<__half> host_key(cache_elements);
    std::vector<__half> host_value(cache_elements);
    for (std::size_t index = 0; index < cache_elements; ++index) {
        host_key[index] = __float2half(random_value(generator));
        host_value[index] = __float2half(random_value(generator));
    }

    __half* device_key = nullptr;
    __half* device_value = nullptr;
    __half* device_query = nullptr;
    __half* device_output = nullptr;
    int* device_block_table = nullptr;
    int* device_sequence_lengths = nullptr;
    check_cuda(cudaMalloc(&device_key, host_key.size() * sizeof(__half)), "cudaMalloc(key)");
    check_cuda(cudaMalloc(&device_value, host_value.size() * sizeof(__half)), "cudaMalloc(value)");
    check_cuda(cudaMemcpy(device_key, host_key.data(), host_key.size() * sizeof(__half), cudaMemcpyHostToDevice),
               "cudaMemcpy(key)");
    check_cuda(cudaMemcpy(device_value, host_value.data(), host_value.size() * sizeof(__half), cudaMemcpyHostToDevice),
               "cudaMemcpy(value)");

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Could not write CUDA profile: " + output_path.string());
    }
    output << "batch_size,sequence_length,latency_ms\n";

    for (const int batch : batches) {
        const std::size_t query_elements = static_cast<std::size_t>(batch) * config.query_heads * config.head_dim;
        const std::size_t table_elements = static_cast<std::size_t>(batch) * max_blocks;
        std::vector<__half> host_query(query_elements, __float2half(0.01F));
        std::vector<int> host_table(table_elements);
        std::vector<int> host_lengths(static_cast<std::size_t>(batch));
        for (int batch_index = 0; batch_index < batch; ++batch_index) {
            host_lengths[static_cast<std::size_t>(batch_index)] = sequence_lengths[0];
            for (int block = 0; block < max_blocks; ++block) {
                host_table[static_cast<std::size_t>(batch_index) * max_blocks + block] = block;
            }
        }
        check_cuda(cudaMalloc(&device_query, query_elements * sizeof(__half)), "cudaMalloc(query)");
        check_cuda(cudaMalloc(&device_output, query_elements * sizeof(__half)), "cudaMalloc(output)");
        check_cuda(cudaMalloc(&device_block_table, table_elements * sizeof(int)), "cudaMalloc(block_table)");
        check_cuda(cudaMalloc(&device_sequence_lengths, static_cast<std::size_t>(batch) * sizeof(int)),
                   "cudaMalloc(sequence_lengths)");
        check_cuda(cudaMemcpy(device_query, host_query.data(), query_elements * sizeof(__half), cudaMemcpyHostToDevice),
                   "cudaMemcpy(query)");
        check_cuda(cudaMemcpy(device_block_table, host_table.data(), table_elements * sizeof(int), cudaMemcpyHostToDevice),
                   "cudaMemcpy(block_table)");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        const dim3 grid(static_cast<unsigned>(batch * static_cast<int>(config.query_heads)));
        const dim3 block(32);
        const std::size_t shared_bytes = config.head_dim * sizeof(float);
        for (const int sequence_length : sequence_lengths) {
            std::fill(host_lengths.begin(), host_lengths.end(), sequence_length);
            check_cuda(cudaMemcpy(device_sequence_lengths, host_lengths.data(),
                                  static_cast<std::size_t>(batch) * sizeof(int), cudaMemcpyHostToDevice),
                       "cudaMemcpy(sequence_lengths)");
            for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
                paged_attention_decode_kernel<<<grid, block, shared_bytes>>>(
                    device_query, device_key, device_value, device_block_table, device_sequence_lengths,
                    device_output, static_cast<int>(config.query_heads), static_cast<int>(config.kv_heads),
                    static_cast<int>(config.head_dim), static_cast<int>(config.block_tokens), max_blocks,
                    1.0F / std::sqrt(static_cast<float>(config.head_dim)));
            }
            check_cuda(cudaGetLastError(), "kernel launch");
            check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
            check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            for (std::size_t iteration = 0; iteration < config.measured_iterations; ++iteration) {
                paged_attention_decode_kernel<<<grid, block, shared_bytes>>>(
                    device_query, device_key, device_value, device_block_table, device_sequence_lengths,
                    device_output, static_cast<int>(config.query_heads), static_cast<int>(config.kv_heads),
                    static_cast<int>(config.head_dim), static_cast<int>(config.block_tokens), max_blocks,
                    1.0F / std::sqrt(static_cast<float>(config.head_dim)));
            }
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
            float elapsed_ms = 0.0F;
            check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
            output << batch << ',' << sequence_length << ','
                   << static_cast<double>(elapsed_ms) / static_cast<double>(config.measured_iterations) << '\n';
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_query);
        cudaFree(device_output);
        cudaFree(device_block_table);
        cudaFree(device_sequence_lengths);
        device_query = nullptr;
        device_output = nullptr;
        device_block_table = nullptr;
        device_sequence_lengths = nullptr;
    }
    cudaFree(device_key);
    cudaFree(device_value);
}

CudaValidationResult validate_cuda_paged_attention() {
    if (!cuda_benchmark_available()) {
        throw std::runtime_error("No CUDA device is available");
    }

    constexpr int batch = 2;
    constexpr int query_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 32;
    constexpr int block_tokens = 4;
    constexpr int max_blocks = 3;
    constexpr int physical_blocks = 6;
    const std::array<int, batch> sequence_lengths = {7, 10};
    const std::array<int, batch * max_blocks> block_table = {3, 1, 5, 0, 4, 2};
    const std::size_t cache_elements =
        static_cast<std::size_t>(physical_blocks) * kv_heads * block_tokens * head_dim;
    const std::size_t query_elements = static_cast<std::size_t>(batch) * query_heads * head_dim;

    std::mt19937 generator(123);
    std::uniform_real_distribution<float> random_value(-0.25F, 0.25F);
    std::vector<__half> host_query(query_elements);
    std::vector<__half> host_key(cache_elements);
    std::vector<__half> host_value(cache_elements);
    for (auto& value : host_query) {
        value = __float2half(random_value(generator));
    }
    for (auto& value : host_key) {
        value = __float2half(random_value(generator));
    }
    for (auto& value : host_value) {
        value = __float2half(random_value(generator));
    }

    __half* device_query = nullptr;
    __half* device_key = nullptr;
    __half* device_value = nullptr;
    __half* device_output = nullptr;
    int* device_table = nullptr;
    int* device_lengths = nullptr;
    check_cuda(cudaMalloc(&device_query, query_elements * sizeof(__half)), "cudaMalloc(query)");
    check_cuda(cudaMalloc(&device_key, cache_elements * sizeof(__half)), "cudaMalloc(key)");
    check_cuda(cudaMalloc(&device_value, cache_elements * sizeof(__half)), "cudaMalloc(value)");
    check_cuda(cudaMalloc(&device_output, query_elements * sizeof(__half)), "cudaMalloc(output)");
    check_cuda(cudaMalloc(&device_table, block_table.size() * sizeof(int)), "cudaMalloc(block_table)");
    check_cuda(cudaMalloc(&device_lengths, sequence_lengths.size() * sizeof(int)), "cudaMalloc(sequence_lengths)");
    check_cuda(cudaMemcpy(device_query, host_query.data(), query_elements * sizeof(__half), cudaMemcpyHostToDevice),
               "cudaMemcpy(query)");
    check_cuda(cudaMemcpy(device_key, host_key.data(), cache_elements * sizeof(__half), cudaMemcpyHostToDevice),
               "cudaMemcpy(key)");
    check_cuda(cudaMemcpy(device_value, host_value.data(), cache_elements * sizeof(__half), cudaMemcpyHostToDevice),
               "cudaMemcpy(value)");
    check_cuda(cudaMemcpy(device_table, block_table.data(), block_table.size() * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy(block_table)");
    check_cuda(cudaMemcpy(device_lengths, sequence_lengths.data(), sequence_lengths.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(sequence_lengths)");

    paged_attention_decode_kernel<<<batch * query_heads, 32, head_dim * sizeof(float)>>>(
        device_query, device_key, device_value, device_table, device_lengths, device_output, query_heads, kv_heads,
        head_dim, block_tokens, max_blocks, 1.0F / std::sqrt(static_cast<float>(head_dim)));
    check_cuda(cudaGetLastError(), "validation kernel launch");
    check_cuda(cudaDeviceSynchronize(), "validation synchronize");
    std::vector<__half> host_output(query_elements);
    check_cuda(cudaMemcpy(host_output.data(), device_output, query_elements * sizeof(__half), cudaMemcpyDeviceToHost),
               "cudaMemcpy(output)");

    CudaValidationResult result;
    double total_error = 0.0;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int query_head = 0; query_head < query_heads; ++query_head) {
            const int kv_head = query_head / (query_heads / kv_heads);
            std::vector<float> scores(static_cast<std::size_t>(sequence_lengths[batch_index]));
            float max_score = -std::numeric_limits<float>::infinity();
            for (int token = 0; token < sequence_lengths[batch_index]; ++token) {
                const int physical_block = block_table[batch_index * max_blocks + token / block_tokens];
                const std::size_t key_base =
                    (static_cast<std::size_t>(physical_block) * kv_heads + kv_head) * block_tokens * head_dim +
                    static_cast<std::size_t>(token % block_tokens) * head_dim;
                const std::size_t query_base =
                    (static_cast<std::size_t>(batch_index) * query_heads + query_head) * head_dim;
                float dot = 0.0F;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    dot += __half2float(host_query[query_base + dimension]) *
                           __half2float(host_key[key_base + dimension]);
                }
                scores[static_cast<std::size_t>(token)] = dot * scale;
                max_score = std::max(max_score, scores[static_cast<std::size_t>(token)]);
            }
            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denominator += score;
            }
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                float expected = 0.0F;
                for (int token = 0; token < sequence_lengths[batch_index]; ++token) {
                    const int physical_block = block_table[batch_index * max_blocks + token / block_tokens];
                    const std::size_t value_base =
                        (static_cast<std::size_t>(physical_block) * kv_heads + kv_head) * block_tokens * head_dim +
                        static_cast<std::size_t>(token % block_tokens) * head_dim;
                    expected += scores[static_cast<std::size_t>(token)] * __half2float(host_value[value_base + dimension]);
                }
                expected /= denominator;
                const std::size_t output_index =
                    (static_cast<std::size_t>(batch_index) * query_heads + query_head) * head_dim + dimension;
                const double error = std::abs(static_cast<double>(expected) -
                                              static_cast<double>(__half2float(host_output[output_index])));
                result.max_absolute_error = std::max(result.max_absolute_error, error);
                total_error += error;
            }
        }
    }
    result.mean_absolute_error = total_error / static_cast<double>(query_elements);

    cudaFree(device_query);
    cudaFree(device_key);
    cudaFree(device_value);
    cudaFree(device_output);
    cudaFree(device_table);
    cudaFree(device_lengths);
    if (result.max_absolute_error > 5e-3) {
        throw std::runtime_error("PagedAttention validation failed: max error exceeds FP16 tolerance");
    }
    return result;
}

}  // namespace serving
