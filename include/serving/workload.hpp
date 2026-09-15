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
  MixedPoisson,
  MixedBurst,
  PrefixHeavy,
};

struct WorkloadStats {
  double first_arrival_ms = 0.0;
  double last_arrival_ms = 0.0;
  double arrival_span_ms = 0.0;
  double realized_arrival_rate_per_second = 0.0;
  bool arrivals_monotonic = true;
};

struct WorkloadConfig {
  WorkloadKind kind = WorkloadKind::MixedBurst;
  std::size_t request_count = 500;
  std::uint32_t seed = 42;
  double arrival_rate_per_second = 24.0;
  std::size_t shared_prefix_tokens = 128;
};

[[nodiscard]] std::vector<RequestSpec>
generate_workload(const WorkloadConfig &config);
[[nodiscard]] WorkloadStats
summarize_workload(const std::vector<RequestSpec> &requests);
void write_trace_csv(const std::filesystem::path &path,
                     const std::vector<RequestSpec> &requests);
[[nodiscard]] std::string to_string(WorkloadKind kind);
[[nodiscard]] WorkloadKind workload_kind_from_string(const std::string &value);

} // namespace serving
