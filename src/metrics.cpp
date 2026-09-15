#include "serving/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace serving {
namespace {

double percentile(std::vector<double> values, const double percentile_value) {
  values.erase(
      std::remove_if(values.begin(), values.end(),
                     [](const double value) { return !std::isfinite(value); }),
      values.end());
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position =
      percentile_value * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  if (lower == upper) {
    return values[lower];
  }
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double mean(const std::vector<double> &values) {
  return values.empty() ? 0.0
                        : std::accumulate(values.begin(), values.end(), 0.0) /
                              static_cast<double>(values.size());
}

struct RequestTpotStats {
  double mean_ms = 0.0;
  double p95_ms = 0.0;
  double max_ms = 0.0;
};

RequestTpotStats request_tpot_stats(const RequestRuntime &request) {
  RequestTpotStats stats;
  if (request.inter_token_latencies_ms.empty()) {
    return stats;
  }
  stats.mean_ms = mean(request.inter_token_latencies_ms);
  stats.p95_ms = percentile(request.inter_token_latencies_ms, 0.95);
  stats.max_ms = *std::max_element(request.inter_token_latencies_ms.begin(),
                                   request.inter_token_latencies_ms.end());
  return stats;
}

double selected_tpot_value(const RequestTpotStats &stats,
                           const TpotSloMode mode) {
  switch (mode) {
  case TpotSloMode::Mean:
    return stats.mean_ms;
  case TpotSloMode::P95:
    return stats.p95_ms;
  case TpotSloMode::Max:
    return stats.max_ms;
  }
  return stats.mean_ms;
}

bool request_meets_slo(const RequestRuntime &request,
                       const ServiceConfig &config,
                       const RequestTpotStats &tpot_stats) {
  if (request.phase != RequestPhase::Finished ||
      !std::isfinite(request.first_token_ms)) {
    return false;
  }
  const double ttft_ms = request.first_token_ms - request.spec.arrival_ms;
  return ttft_ms <= config.ttft_slo_ms &&
         selected_tpot_value(tpot_stats, config.tpot_slo_mode) <=
             config.tpot_slo_ms;
}

std::string csv_cell(const std::string &value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '\"') {
      escaped += "\"\"";
    } else {
      escaped += character;
    }
  }
  escaped += '\"';
  return escaped;
}

} // namespace

SimulationResult summarize_result(
    const std::string &scenario_name, const std::string &latency_source,
    const std::vector<RequestRuntime> &requests, const double simulated_time_ms,
    const double busy_time_ms, const std::size_t processed_prefill_tokens,
    const double average_block_utilization,
    const double peak_internal_waste_ratio,
    const double peak_external_fragmentation_ratio,
    const std::size_t peak_allocated_blocks,
    const std::size_t peak_active_requests, const std::size_t cache_evictions,
    const ServiceConfig &config) {
  SimulationResult result;
  result.scenario_name = scenario_name;
  result.latency_source = latency_source;
  result.tpot_slo_mode = to_string(config.tpot_slo_mode);
  result.simulated_time_ms = simulated_time_ms;
  result.busy_time_ms = busy_time_ms;
  result.processed_prefill_tokens = processed_prefill_tokens;
  if (simulated_time_ms > 0.0) {
    result.busy_fraction = busy_time_ms / simulated_time_ms;
  }
  result.average_block_utilization = average_block_utilization;
  result.peak_internal_waste_ratio = peak_internal_waste_ratio;
  result.peak_external_fragmentation_ratio = peak_external_fragmentation_ratio;
  result.peak_allocated_blocks = peak_allocated_blocks;
  result.peak_active_requests = peak_active_requests;
  result.cache_evictions = cache_evictions;

  if (!requests.empty()) {
    result.measurement_start_ms = requests.front().spec.arrival_ms;
    double last_arrival_ms = requests.front().spec.arrival_ms;
    for (const RequestRuntime &request : requests) {
      result.measurement_start_ms =
          std::min(result.measurement_start_ms, request.spec.arrival_ms);
      last_arrival_ms = std::max(last_arrival_ms, request.spec.arrival_ms);
    }
    result.workload_arrival_span_ms =
        last_arrival_ms - result.measurement_start_ms;
    if (requests.size() > 1 && result.workload_arrival_span_ms > 0.0) {
      result.realized_arrival_rate_per_second =
          static_cast<double>(requests.size() - 1) * 1'000.0 /
          result.workload_arrival_span_ms;
    }
  }

  std::vector<double> ttft;
  std::vector<double> all_itl;
  std::vector<double> request_mean_tpot;
  std::vector<double> request_max_itl;
  ttft.reserve(requests.size());
  request_mean_tpot.reserve(requests.size());
  request_max_itl.reserve(requests.size());
  for (const RequestRuntime &request : requests) {
    result.all_generated_tokens += request.generated_tokens;
    result.total_prompt_tokens += request.prompt_tokens();
    result.total_prefix_hit_tokens += request.prefix_hit_tokens;
    if (std::isfinite(request.admitted_ms)) {
      ++result.admitted_requests;
      result.max_queue_wait_ms =
          std::max(result.max_queue_wait_ms,
                   request.admitted_ms - request.spec.arrival_ms);
    }
    if (request.phase == RequestPhase::Finished) {
      ++result.completed_requests;
      result.generated_tokens += request.generated_tokens;
      if (std::isfinite(request.first_token_ms)) {
        const double request_ttft =
            request.first_token_ms - request.spec.arrival_ms;
        const RequestTpotStats tpot_stats = request_tpot_stats(request);
        ttft.push_back(request_ttft);
        all_itl.insert(all_itl.end(), request.inter_token_latencies_ms.begin(),
                       request.inter_token_latencies_ms.end());
        request_mean_tpot.push_back(tpot_stats.mean_ms);
        request_max_itl.push_back(tpot_stats.max_ms);
        if (request_meets_slo(request, config, tpot_stats)) {
          ++result.slo_compliant_requests;
        }
      }
    } else if (request.phase == RequestPhase::Rejected) {
      ++result.rejected_requests;
      if (request.rejection_reason == "request exceeds KV capacity") {
        ++result.capacity_rejected_requests;
      } else if (request.rejection_reason == "KV growth exhausted capacity") {
        ++result.kv_growth_rejected_requests;
      } else if (request.rejection_reason == "waiting time exceeded TTFT SLO") {
        ++result.ttft_expired_requests;
      }
    }
  }

  if (result.total_prompt_tokens > 0) {
    result.prefix_hit_token_ratio =
        static_cast<double>(result.total_prefix_hit_tokens) /
        static_cast<double>(result.total_prompt_tokens);
  }
  result.ttft_p50_ms = percentile(ttft, 0.50);
  result.ttft_p95_ms = percentile(ttft, 0.95);
  result.ttft_p99_ms = percentile(ttft, 0.99);
  result.itl_p50_ms = percentile(all_itl, 0.50);
  result.itl_p95_ms = percentile(all_itl, 0.95);
  result.itl_p99_ms = percentile(all_itl, 0.99);
  result.mean_tpot_p50_ms = percentile(request_mean_tpot, 0.50);
  result.mean_tpot_p95_ms = percentile(request_mean_tpot, 0.95);
  result.mean_tpot_p99_ms = percentile(request_mean_tpot, 0.99);
  result.max_itl_p50_ms = percentile(request_max_itl, 0.50);
  result.max_itl_p95_ms = percentile(request_max_itl, 0.95);
  result.max_itl_p99_ms = percentile(request_max_itl, 0.99);
  const double seconds = simulated_time_ms / 1'000.0;
  if (seconds > 0.0) {
    result.throughput_tokens_per_second =
        static_cast<double>(result.generated_tokens) / seconds;
    result.slo_goodput_requests_per_second =
        static_cast<double>(result.slo_compliant_requests) / seconds;
  }
  return result;
}

void write_summary_csv(const std::filesystem::path &path,
                       const SimulationResult &result) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Could not write summary: " + path.string());
  }
  output << "scenario,latency_source,tpot_slo_mode,measurement_start_ms,"
            "simulated_time_ms,busy_time_ms,busy_fraction,"
            "workload_arrival_span_ms,realized_arrival_rate_per_second,"
            "completed_requests,admitted_requests,"
            "rejected_requests,capacity_rejected_requests,kv_growth_rejected_"
            "requests,ttft_expired_"
            "requests,generated_tokens,all_generated_tokens,processed_prefill_"
            "tokens,"
            "total_prompt_tokens,total_prefix_hit_tokens,prefix_hit_token_"
            "ratio,slo_compliant_requests,"
            "throughput_tokens_per_second,slo_goodput_requests_per_second,ttft_"
            "p50_ms,ttft_p95_ms,ttft_p99_ms,"
            "itl_p50_ms,itl_p95_ms,itl_p99_ms,mean_tpot_p50_ms,mean_tpot_p95_"
            "ms,mean_tpot_p99_ms,"
            "max_itl_p50_ms,max_itl_p95_ms,max_itl_p99_ms,max_queue_wait_ms,"
            "average_block_utilization,"
            "peak_internal_waste_ratio,peak_external_fragmentation_ratio,peak_"
            "allocated_blocks,"
            "peak_active_requests,cache_evictions\n";
  output << std::fixed << std::setprecision(4) << csv_cell(result.scenario_name)
         << ',' << csv_cell(result.latency_source) << ','
         << result.tpot_slo_mode << ',' << result.measurement_start_ms << ','
         << result.simulated_time_ms << ',' << result.busy_time_ms << ','
         << result.busy_fraction << ',' << result.workload_arrival_span_ms
         << ',' << result.realized_arrival_rate_per_second << ','
         << result.completed_requests << ',' << result.admitted_requests << ','
         << result.rejected_requests << ',' << result.capacity_rejected_requests
         << ',' << result.kv_growth_rejected_requests << ','
         << result.ttft_expired_requests << ',' << result.generated_tokens
         << ',' << result.all_generated_tokens << ','
         << result.processed_prefill_tokens << ',' << result.total_prompt_tokens
         << ',' << result.total_prefix_hit_tokens << ','
         << result.prefix_hit_token_ratio << ','
         << result.slo_compliant_requests << ','
         << result.throughput_tokens_per_second << ','
         << result.slo_goodput_requests_per_second << ',' << result.ttft_p50_ms
         << ',' << result.ttft_p95_ms << ',' << result.ttft_p99_ms << ','
         << result.itl_p50_ms << ',' << result.itl_p95_ms << ','
         << result.itl_p99_ms << ',' << result.mean_tpot_p50_ms << ','
         << result.mean_tpot_p95_ms << ',' << result.mean_tpot_p99_ms << ','
         << result.max_itl_p50_ms << ',' << result.max_itl_p95_ms << ','
         << result.max_itl_p99_ms << ',' << result.max_queue_wait_ms << ','
         << result.average_block_utilization << ','
         << result.peak_internal_waste_ratio << ','
         << result.peak_external_fragmentation_ratio << ','
         << result.peak_allocated_blocks << ',' << result.peak_active_requests
         << ',' << result.cache_evictions << '\n';
}

void write_requests_csv(const std::filesystem::path &path,
                        const std::vector<RequestRuntime> &requests,
                        const ServiceConfig &config) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Could not write request metrics: " +
                             path.string());
  }
  output << "request_id,arrival_ms,prompt_tokens,max_new_tokens,prefix_group,"
            "phase,prefix_hit_tokens,"
            "admitted_ms,first_token_ms,finished_ms,queue_wait_ms,ttft_ms,mean_"
            "tpot_ms,p95_itl_ms,max_itl_ms,"
            "meets_ttft_slo,meets_tpot_slo,meets_slo,generated_tokens,"
            "rejection_reason\n";
  output << std::fixed << std::setprecision(4);
  for (const RequestRuntime &request : requests) {
    const auto finite_or_empty = [](const double value) {
      return std::isfinite(value) ? std::to_string(value) : std::string{};
    };
    const double queue_wait =
        std::isfinite(request.admitted_ms)
            ? request.admitted_ms - request.spec.arrival_ms
            : std::numeric_limits<double>::quiet_NaN();
    const double ttft = std::isfinite(request.first_token_ms)
                            ? request.first_token_ms - request.spec.arrival_ms
                            : std::numeric_limits<double>::quiet_NaN();
    const RequestTpotStats tpot_stats = request_tpot_stats(request);
    const bool finished_with_token =
        request.phase == RequestPhase::Finished && std::isfinite(ttft);
    const bool meets_ttft = finished_with_token && ttft <= config.ttft_slo_ms;
    const bool meets_tpot =
        finished_with_token &&
        selected_tpot_value(tpot_stats, config.tpot_slo_mode) <=
            config.tpot_slo_ms;
    output << request.spec.id << ',' << request.spec.arrival_ms << ','
           << request.prompt_tokens() << ',' << request.spec.max_new_tokens
           << ',' << csv_cell(request.spec.prefix_group) << ','
           << static_cast<int>(request.phase) << ','
           << request.prefix_hit_tokens << ','
           << finite_or_empty(request.admitted_ms) << ','
           << finite_or_empty(request.first_token_ms) << ','
           << finite_or_empty(request.finished_ms) << ','
           << finite_or_empty(queue_wait) << ',' << finite_or_empty(ttft) << ','
           << tpot_stats.mean_ms << ',' << tpot_stats.p95_ms << ','
           << tpot_stats.max_ms << ',' << meets_ttft << ',' << meets_tpot << ','
           << (meets_ttft && meets_tpot) << ',' << request.generated_tokens
           << ',' << csv_cell(request.rejection_reason) << '\n';
  }
}

} // namespace serving
