#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Keep the extension and the standalone benchmark on the same kernel implementation.
#include "paged_attention.cu"

namespace serving {

torch::Tensor torch_paged_attention_decode(torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
                                           torch::Tensor block_table, torch::Tensor sequence_lengths) {
    TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == torch::kFloat16 && key_cache.scalar_type() == torch::kFloat16 &&
                    value_cache.scalar_type() == torch::kFloat16,
                "query, key_cache, and value_cache must be float16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt && sequence_lengths.scalar_type() == torch::kInt,
                "block_table and sequence_lengths must be int32");
    TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() &&
                    block_table.is_contiguous() && sequence_lengths.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(),
                "expected query [B, Hq, D] and KV caches [num_blocks, Hkv, block_tokens, D]");
    TORCH_CHECK(block_table.dim() == 2 && sequence_lengths.dim() == 1 &&
                    block_table.size(0) == query.size(0) && sequence_lengths.size(0) == query.size(0),
                "batch dimensions must match");
    TORCH_CHECK(query.size(2) == key_cache.size(3) && query.size(1) % key_cache.size(1) == 0,
                "head dimensions must match and query heads must be divisible by KV heads");
    TORCH_CHECK(query.size(2) <= 128, "the readable baseline kernel supports head_dim <= 128");

    const c10::cuda::CUDAGuard device_guard(query.device());
    auto output = torch::empty_like(query);
    const dim3 grid(static_cast<unsigned>(query.size(0) * query.size(1)));
    const dim3 block(32);
    const std::size_t shared_bytes = static_cast<std::size_t>(query.size(2)) * sizeof(float);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(query.get_device()).stream();
    paged_attention_decode_kernel<<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()), block_table.data_ptr<int>(),
        sequence_lengths.data_ptr<int>(), reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        static_cast<int>(query.size(1)), static_cast<int>(key_cache.size(1)), static_cast<int>(query.size(2)),
        static_cast<int>(key_cache.size(2)), static_cast<int>(block_table.size(1)),
        1.0F / std::sqrt(static_cast<float>(query.size(2))));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace serving

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("paged_attention_decode", &serving::torch_paged_attention_decode,
               "FP16 GQA paged attention decode (CUDA)");
}
