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
    double simulated_time_ms = 0.0;
    std::size_t completed_requests = 0;
    std::size_t rejected_requests = 0;
    std::size_t generated_tokens = 0;
    std::size_t slo_compliant_requests = 0;
    double throughput_tokens_per_second = 0.0;
    double slo_goodput_requests_per_second = 0.0;
    double ttft_p50_ms = 0.0;
    double ttft_p95_ms = 0.0;
    double ttft_p99_ms = 0.0;
    double tpot_p50_ms = 0.0;
    double tpot_p95_ms = 0.0;
    double tpot_p99_ms = 0.0;
    double max_queue_wait_ms = 0.0;
    double average_block_utilization = 0.0;
    double peak_fragmentation_ratio = 0.0;
    std::size_t cache_evictions = 0;
};

[[nodiscard]] SimulationResult summarize_result(const std::string& scenario_name,
                                                 const std::string& latency_source,
                                                 const std::vector<RequestRuntime>& requests,
                                                 double simulated_time_ms,
                                                 double average_block_utilization,
                                                 double peak_fragmentation_ratio,
                                                 std::size_t cache_evictions,
                                                 const ServiceConfig& config);

void write_summary_csv(const std::filesystem::path& path, const SimulationResult& result);
void write_requests_csv(const std::filesystem::path& path, const std::vector<RequestRuntime>& requests);

}  // namespace serving
