#pragma once

#include "serving/types.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace serving {

struct SimulationResult {
  std::string scenario_name;
  std::string latency_source;
  std::string tpot_slo_mode;
  double measurement_start_ms = 0.0;
  double simulated_time_ms = 0.0;
  double busy_time_ms = 0.0;
  double busy_fraction = 0.0;
  double workload_arrival_span_ms = 0.0;
  double realized_arrival_rate_per_second = 0.0;
  std::size_t completed_requests = 0;
  std::size_t admitted_requests = 0;
  std::size_t rejected_requests = 0;
  std::size_t capacity_rejected_requests = 0;
  std::size_t kv_growth_rejected_requests = 0;
  std::size_t ttft_expired_requests = 0;
  std::size_t generated_tokens = 0;
  std::size_t all_generated_tokens = 0;
  std::size_t processed_prefill_tokens = 0;
  std::size_t total_prompt_tokens = 0;
  std::size_t total_prefix_hit_tokens = 0;
  std::size_t slo_compliant_requests = 0;
  double prefix_hit_token_ratio = 0.0;
  double throughput_tokens_per_second = 0.0;
  double slo_goodput_requests_per_second = 0.0;
  double ttft_p50_ms = 0.0;
  double ttft_p95_ms = 0.0;
  double ttft_p99_ms = 0.0;
  double itl_p50_ms = 0.0;
  double itl_p95_ms = 0.0;
  double itl_p99_ms = 0.0;
  double mean_tpot_p50_ms = 0.0;
  double mean_tpot_p95_ms = 0.0;
  double mean_tpot_p99_ms = 0.0;
  double max_itl_p50_ms = 0.0;
  double max_itl_p95_ms = 0.0;
  double max_itl_p99_ms = 0.0;
  double max_queue_wait_ms = 0.0;
  double average_block_utilization = 0.0;
  double peak_internal_waste_ratio = 0.0;
  double peak_external_fragmentation_ratio = 0.0;
  std::size_t peak_allocated_blocks = 0;
  std::size_t peak_active_requests = 0;
  std::size_t cache_evictions = 0;
};

[[nodiscard]] SimulationResult summarize_result(
    const std::string &scenario_name, const std::string &latency_source,
    const std::vector<RequestRuntime> &requests, double simulated_time_ms,
    double busy_time_ms, std::size_t processed_prefill_tokens,
    double average_block_utilization, double peak_internal_waste_ratio,
    double peak_external_fragmentation_ratio, std::size_t peak_allocated_blocks,
    std::size_t peak_active_requests, std::size_t cache_evictions,
    const ServiceConfig &config);

void write_summary_csv(const std::filesystem::path &path,
                       const SimulationResult &result);
void write_requests_csv(const std::filesystem::path &path,
                        const std::vector<RequestRuntime> &requests,
                        const ServiceConfig &config);

} // namespace serving
