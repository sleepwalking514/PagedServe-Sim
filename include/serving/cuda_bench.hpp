#pragma once

#include <cstddef>
#include <filesystem>

namespace serving {

struct CudaBenchmarkConfig {
    std::size_t kv_heads = 8;
    std::size_t query_heads = 32;
    std::size_t head_dim = 128;
    std::size_t block_tokens = 16;
    std::size_t warmup_iterations = 20;
    std::size_t measured_iterations = 100;
};

struct CudaValidationResult {
    double max_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
};

[[nodiscard]] bool cuda_benchmark_available();
void write_cuda_decode_profile(const std::filesystem::path& output_path, const CudaBenchmarkConfig& config);
[[nodiscard]] CudaValidationResult validate_cuda_paged_attention();

}  // namespace serving
