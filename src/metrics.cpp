#include "serving/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace serving {
namespace {

double percentile(std::vector<double> values, const double percentile_value) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = percentile_value * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

}  // namespace

SimulationResult summarize_result(const std::string& scenario_name, const std::string& latency_source,
                                  const std::vector<RequestRuntime>& requests, const double simulated_time_ms,
                                  const double average_block_utilization,
                                  const double peak_fragmentation_ratio, const std::size_t cache_evictions,
                                  const ServiceConfig& config) {
    SimulationResult result;
    result.scenario_name = scenario_name;
    result.latency_source = latency_source;
    result.simulated_time_ms = simulated_time_ms;
    result.average_block_utilization = average_block_utilization;
    result.peak_fragmentation_ratio = peak_fragmentation_ratio;
    result.cache_evictions = cache_evictions;

    std::vector<double> ttft;
    std::vector<double> tpot;
    ttft.reserve(requests.size());
    for (const RequestRuntime& request : requests) {
        if (request.phase == RequestPhase::Finished) {
            ++result.completed_requests;
            result.generated_tokens += request.generated_tokens;
            const double request_ttft = request.first_token_ms - request.spec.arrival_ms;
            ttft.push_back(request_ttft);
            tpot.insert(tpot.end(), request.inter_token_latencies_ms.begin(), request.inter_token_latencies_ms.end());
            bool meets_slo = request_ttft <= config.ttft_slo_ms;
            meets_slo = meets_slo && std::all_of(
                                      request.inter_token_latencies_ms.begin(),
                                      request.inter_token_latencies_ms.end(),
                                      [&config](const double value) { return value <= config.tpot_slo_ms; });
            if (meets_slo) {
                ++result.slo_compliant_requests;
            }
        } else if (request.phase == RequestPhase::Rejected) {
            ++result.rejected_requests;
        }
        if (std::isfinite(request.admitted_ms)) {
            result.max_queue_wait_ms = std::max(result.max_queue_wait_ms,
                                                request.admitted_ms - request.spec.arrival_ms);
        }
    }

    result.ttft_p50_ms = percentile(ttft, 0.50);
    result.ttft_p95_ms = percentile(ttft, 0.95);
    result.ttft_p99_ms = percentile(ttft, 0.99);
    result.tpot_p50_ms = percentile(tpot, 0.50);
    result.tpot_p95_ms = percentile(tpot, 0.95);
    result.tpot_p99_ms = percentile(tpot, 0.99);
    const double seconds = simulated_time_ms / 1'000.0;
    if (seconds > 0.0) {
        result.throughput_tokens_per_second = static_cast<double>(result.generated_tokens) / seconds;
        result.slo_goodput_requests_per_second = static_cast<double>(result.slo_compliant_requests) / seconds;
    }
    return result;
}

void write_summary_csv(const std::filesystem::path& path, const SimulationResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write summary: " + path.string());
    }
    output << "scenario,latency_source,simulated_time_ms,completed_requests,rejected_requests,generated_tokens,"
              "slo_compliant_requests,throughput_tokens_per_second,slo_goodput_requests_per_second,"
              "ttft_p50_ms,ttft_p95_ms,ttft_p99_ms,tpot_p50_ms,tpot_p95_ms,tpot_p99_ms,max_queue_wait_ms,"
              "average_block_utilization,peak_fragmentation_ratio,cache_evictions\n";
    output << std::fixed << std::setprecision(4) << result.scenario_name << ',' << result.latency_source << ','
           << result.simulated_time_ms << ',' << result.completed_requests << ',' << result.rejected_requests << ','
           << result.generated_tokens << ',' << result.slo_compliant_requests << ','
           << result.throughput_tokens_per_second << ',' << result.slo_goodput_requests_per_second << ','
           << result.ttft_p50_ms << ',' << result.ttft_p95_ms << ',' << result.ttft_p99_ms << ',' << result.tpot_p50_ms
           << ',' << result.tpot_p95_ms << ',' << result.tpot_p99_ms << ',' << result.max_queue_wait_ms << ','
           << result.average_block_utilization << ',' << result.peak_fragmentation_ratio << ','
           << result.cache_evictions << '\n';
}

void write_requests_csv(const std::filesystem::path& path, const std::vector<RequestRuntime>& requests) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write request metrics: " + path.string());
    }
    output << "request_id,arrival_ms,prompt_tokens,output_tokens,prefix_group,phase,prefix_hit_tokens,"
              "admitted_ms,first_token_ms,finished_ms,queue_wait_ms,ttft_ms,generated_tokens,rejection_reason\n";
    output << std::fixed << std::setprecision(4);
    for (const RequestRuntime& request : requests) {
        const auto finite_or_empty = [](const double value) {
            return std::isfinite(value) ? std::to_string(value) : std::string{};
        };
        const double queue_wait = std::isfinite(request.admitted_ms)
                                      ? request.admitted_ms - request.spec.arrival_ms
                                      : std::numeric_limits<double>::quiet_NaN();
        const double ttft = std::isfinite(request.first_token_ms)
                                ? request.first_token_ms - request.spec.arrival_ms
                                : std::numeric_limits<double>::quiet_NaN();
        output << request.spec.id << ',' << request.spec.arrival_ms << ',' << request.prompt_tokens() << ','
               << request.spec.output_tokens << ',' << request.spec.prefix_group << ','
               << static_cast<int>(request.phase) << ',' << request.prefix_hit_tokens << ','
               << finite_or_empty(request.admitted_ms) << ',' << finite_or_empty(request.first_token_ms) << ','
               << finite_or_empty(request.finished_ms) << ',' << finite_or_empty(queue_wait) << ','
               << finite_or_empty(ttft) << ',' << request.generated_tokens << ',' << request.rejection_reason << '\n';
    }
}

}  // namespace serving
