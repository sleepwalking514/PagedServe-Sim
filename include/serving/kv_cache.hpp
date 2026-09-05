#pragma once

#include "serving/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace serving {

enum class AdmissionResult {
    Admitted,
    TemporarilyUnavailable,
    Impossible,
};

class KvCache {
public:
    virtual ~KvCache() = default;

    virtual AdmissionResult admit(RequestRuntime& request) = 0;
    virtual bool append_tokens(RequestRuntime& request, std::size_t token_count) = 0;
    virtual void cache_completed_prompt_block(RequestRuntime& request, std::size_t block_index,
                                              std::uint64_t block_hash) = 0;
    virtual void release(RequestRuntime& request) = 0;
    [[nodiscard]] virtual KvCacheStats stats() const = 0;
};

[[nodiscard]] std::unique_ptr<KvCache> make_kv_cache(const ModelConfig& model, const ServiceConfig& service);
[[nodiscard]] std::uint64_t hash_prompt_block(std::span<const std::uint32_t> tokens);

}  // namespace serving
