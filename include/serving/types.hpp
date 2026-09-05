#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace serving {

enum class RequestPhase {
    Queued,
    Prefill,
    Decode,
    Finished,
    Rejected,
};

enum class SchedulerPolicy {
    PrefillFirst,
    DecodeFirst,
};

enum class KvLayout {
    Contiguous,
    Paged,
};

struct ModelConfig {
    std::size_t num_layers = 32;
    std::size_t kv_heads = 8;
    std::size_t head_dim = 128;
    std::size_t block_tokens = 16;
    std::size_t total_blocks = 6'144;
    std::size_t dtype_bytes = 2;

    [[nodiscard]] std::size_t kv_block_bytes() const {
        return 2U * num_layers * kv_heads * head_dim * block_tokens * dtype_bytes;
    }
};

struct ServiceConfig {
    KvLayout kv_layout = KvLayout::Paged;
    SchedulerPolicy scheduler_policy = SchedulerPolicy::DecodeFirst;
    bool prefix_cache_enabled = true;
    std::size_t token_budget = 512;
    std::size_t prefill_chunk_tokens = 128;
    std::size_t max_active_requests = 64;
    double ttft_slo_ms = 1'000.0;
    double tpot_slo_ms = 80.0;
};

struct RequestSpec {
    std::uint64_t id = 0;
    double arrival_ms = 0.0;
    std::vector<std::uint32_t> prompt_tokens;
    std::size_t output_tokens = 0;
    std::string prefix_group;
};

struct RequestRuntime {
    RequestSpec spec;
    RequestPhase phase = RequestPhase::Queued;
    std::size_t prefetched_tokens = 0;
    std::size_t generated_tokens = 0;
    std::size_t stored_tokens = 0;
    std::size_t prefix_hit_tokens = 0;
    std::size_t reserved_blocks = 0;
    std::vector<int> block_table;
    double admitted_ms = std::numeric_limits<double>::quiet_NaN();
    double first_token_ms = std::numeric_limits<double>::quiet_NaN();
    double last_token_ms = std::numeric_limits<double>::quiet_NaN();
    double finished_ms = std::numeric_limits<double>::quiet_NaN();
    std::string rejection_reason;
    std::vector<double> inter_token_latencies_ms;

    [[nodiscard]] std::size_t prompt_tokens() const {
        return spec.prompt_tokens.size();
    }

    [[nodiscard]] std::size_t total_tokens() const {
        return prompt_tokens() + spec.output_tokens;
    }

    [[nodiscard]] bool is_active() const {
        return phase == RequestPhase::Prefill || phase == RequestPhase::Decode;
    }
};

struct KvCacheStats {
    std::size_t total_blocks = 0;
    std::size_t allocated_blocks = 0;
    std::size_t active_blocks = 0;
    std::size_t cached_blocks = 0;
    std::size_t available_blocks = 0;
    std::size_t cache_evictions = 0;
    double physical_block_utilization = 0.0;
    double fragmentation_ratio = 0.0;
};

struct SimulationConfig {
    ModelConfig model;
    ServiceConfig service;
};

[[nodiscard]] inline std::string to_string(const SchedulerPolicy policy) {
    return policy == SchedulerPolicy::DecodeFirst ? "decode_first" : "prefill_first";
}

[[nodiscard]] inline std::string to_string(const KvLayout layout) {
    return layout == KvLayout::Paged ? "paged" : "contiguous";
}

}  // namespace serving
