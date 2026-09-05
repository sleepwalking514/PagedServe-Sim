#pragma once

#include "serving/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace serving {

enum class WorkloadKind {
    Short,
    MixedBurst,
    PrefixHeavy,
};

struct WorkloadConfig {
    WorkloadKind kind = WorkloadKind::MixedBurst;
    std::size_t request_count = 500;
    std::uint32_t seed = 42;
    double arrival_rate_per_second = 24.0;
    std::size_t shared_prefix_tokens = 128;
};

[[nodiscard]] std::vector<RequestSpec> generate_workload(const WorkloadConfig& config);
void write_trace_csv(const std::filesystem::path& path, const std::vector<RequestSpec>& requests);
[[nodiscard]] std::string to_string(WorkloadKind kind);
[[nodiscard]] WorkloadKind workload_kind_from_string(const std::string& value);

}  // namespace serving
