#include "serving/cuda_bench.hpp"

#include <stdexcept>

namespace serving {

bool cuda_benchmark_available() {
    return false;
}

void write_cuda_decode_profile(const std::filesystem::path&, const CudaBenchmarkConfig&) {
    throw std::runtime_error("This build does not include CUDA. Configure with a CUDA compiler first.");
}

CudaValidationResult validate_cuda_paged_attention() {
    throw std::runtime_error("This build does not include CUDA. Configure with a CUDA compiler first.");
}

}  // namespace serving
