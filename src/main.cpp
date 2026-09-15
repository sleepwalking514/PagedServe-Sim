#include "serving/cuda_bench.hpp"
#include "serving/latency_model.hpp"
#include "serving/metrics.hpp"
#include "serving/simulator.hpp"
#include "serving/workload.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using serving::KvLayout;
using serving::SchedulerPolicy;
using serving::TpotSloMode;

struct CliOptions {
  serving::SimulationConfig simulation;
  serving::WorkloadConfig workload;
  std::string scenario = "all";
  std::filesystem::path output_directory = "results";
  std::filesystem::path config_path;
  std::filesystem::path decode_profile_path;
  std::filesystem::path cuda_profile_output;
  bool validate_cuda = false;
};

std::string trim(std::string value) {
  const auto not_space = [](const unsigned char character) {
    return !std::isspace(character);
  };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

std::unordered_map<std::string, std::string>
read_config(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open config: " + path.string());
  }
  std::unordered_map<std::string, std::string> values;
  std::string section;
  std::string line;
  while (std::getline(input, line)) {
    line = trim(line);
    if (line.empty() || line.front() == '#' || line.front() == ';') {
      continue;
    }
    if (line.front() == '[' && line.back() == ']') {
      section = trim(line.substr(1, line.size() - 2));
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, separator));
    const std::string value = trim(line.substr(separator + 1));
    values[section.empty() ? key : section + "." + key] = value;
  }
  return values;
}

template <typename T>
void set_if_present(const std::unordered_map<std::string, std::string> &values,
                    const std::string &key, T &target);

template <>
void set_if_present(const std::unordered_map<std::string, std::string> &values,
                    const std::string &key, std::size_t &target) {
  if (const auto it = values.find(key); it != values.end()) {
    target = static_cast<std::size_t>(std::stoull(it->second));
  }
}

template <>
void set_if_present(const std::unordered_map<std::string, std::string> &values,
                    const std::string &key, std::uint32_t &target) {
  if (const auto it = values.find(key); it != values.end()) {
    target = static_cast<std::uint32_t>(std::stoul(it->second));
  }
}

template <>
void set_if_present(const std::unordered_map<std::string, std::string> &values,
                    const std::string &key, double &target) {
  if (const auto it = values.find(key); it != values.end()) {
    target = std::stod(it->second);
  }
}

template <>
void set_if_present(const std::unordered_map<std::string, std::string> &values,
                    const std::string &key, bool &target) {
  if (const auto it = values.find(key); it != values.end()) {
    target = it->second == "true" || it->second == "1" || it->second == "on";
  }
}

void apply_config_file(CliOptions &options) {
  if (options.config_path.empty()) {
    return;
  }
  const auto values = read_config(options.config_path);
  auto &model = options.simulation.model;
  auto &service = options.simulation.service;
  auto &workload = options.workload;
  set_if_present(values, "model.num_layers", model.num_layers);
  set_if_present(values, "model.kv_heads", model.kv_heads);
  set_if_present(values, "model.head_dim", model.head_dim);
  set_if_present(values, "model.block_tokens", model.block_tokens);
  set_if_present(values, "model.total_blocks", model.total_blocks);
  set_if_present(values, "model.dtype_bytes", model.dtype_bytes);
  set_if_present(values, "service.token_budget", service.token_budget);
  set_if_present(values, "service.prefill_chunk_tokens",
                 service.prefill_chunk_tokens);
  set_if_present(values, "service.max_active_requests",
                 service.max_active_requests);
  set_if_present(values, "service.ttft_slo_ms", service.ttft_slo_ms);
  set_if_present(values, "service.tpot_slo_ms", service.tpot_slo_ms);
  set_if_present(values, "service.prefix_cache_enabled",
                 service.prefix_cache_enabled);
  set_if_present(values, "workload.request_count", workload.request_count);
  set_if_present(values, "workload.seed", workload.seed);
  set_if_present(values, "workload.arrival_rate_per_second",
                 workload.arrival_rate_per_second);
  set_if_present(values, "workload.shared_prefix_tokens",
                 workload.shared_prefix_tokens);
  if (const auto it = values.find("service.kv_layout"); it != values.end()) {
    if (it->second == "contiguous") {
      service.kv_layout = KvLayout::Contiguous;
    } else if (it->second == "paged") {
      service.kv_layout = KvLayout::Paged;
    } else {
      throw std::invalid_argument("Unknown KV layout: " + it->second);
    }
  }
  if (const auto it = values.find("service.scheduler_policy");
      it != values.end()) {
    if (it->second == "prefill_first") {
      service.scheduler_policy = SchedulerPolicy::PrefillFirst;
    } else if (it->second == "decode_first") {
      service.scheduler_policy = SchedulerPolicy::DecodeFirst;
    } else {
      throw std::invalid_argument("Unknown scheduler policy: " + it->second);
    }
  }
  if (const auto it = values.find("service.tpot_slo_mode");
      it != values.end()) {
    if (it->second == "mean") {
      service.tpot_slo_mode = TpotSloMode::Mean;
    } else if (it->second == "p95") {
      service.tpot_slo_mode = TpotSloMode::P95;
    } else if (it->second == "max") {
      service.tpot_slo_mode = TpotSloMode::Max;
    } else {
      throw std::invalid_argument("Unknown TPOT SLO mode: " + it->second);
    }
  }
  if (const auto it = values.find("workload.kind"); it != values.end()) {
    workload.kind = serving::workload_kind_from_string(it->second);
  }
}

std::string require_value(int &index, const int argc, char **argv) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string("Missing value for ") +
                                argv[index]);
  }
  return argv[++index];
}

void parse_cli_overrides(CliOptions &options, const int argc, char **argv) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--config") {
      options.config_path = require_value(index, argc, argv);
    } else if (argument == "--scenario") {
      options.scenario = require_value(index, argc, argv);
    } else if (argument == "--workload") {
      options.workload.kind =
          serving::workload_kind_from_string(require_value(index, argc, argv));
    } else if (argument == "--requests") {
      options.workload.request_count = static_cast<std::size_t>(
          std::stoull(require_value(index, argc, argv)));
    } else if (argument == "--seed") {
      options.workload.seed = static_cast<std::uint32_t>(
          std::stoul(require_value(index, argc, argv)));
    } else if (argument == "--output") {
      options.output_directory = require_value(index, argc, argv);
    } else if (argument == "--decode-profile") {
      options.decode_profile_path = require_value(index, argc, argv);
    } else if (argument == "--cuda-profile") {
      options.cuda_profile_output = require_value(index, argc, argv);
    } else if (argument == "--cuda-validate") {
      options.validate_cuda = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: serving_sim [options]\n"
             "  --config PATH             Read an ini-like config file\n"
             "  --scenario "
             "all|baseline|paged|prefix|continuous_prefill|continuous_decode|"
             "paged_prefill|paged_decode\n"
             "  --workload short|mixed_poisson|mixed_burst|prefix_heavy\n"
             "  --requests N --seed N --output DIR\n"
             "  --decode-profile PATH     Use a measured CUDA profile\n"
             "  --cuda-profile PATH       Generate a CUDA profile and exit\n"
             "  --cuda-validate           Compare the CUDA kernel with a CPU "
             "reference\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }
}

serving::SimulationConfig scenario_config(const serving::SimulationConfig &base,
                                          const std::string &scenario) {
  serving::SimulationConfig config = base;
  if (scenario == "baseline" || scenario == "continuous_prefill") {
    config.service.kv_layout = KvLayout::Contiguous;
    config.service.scheduler_policy = SchedulerPolicy::PrefillFirst;
    config.service.prefix_cache_enabled = false;
  } else if (scenario == "continuous_decode") {
    config.service.kv_layout = KvLayout::Contiguous;
    config.service.scheduler_policy = SchedulerPolicy::DecodeFirst;
    config.service.prefix_cache_enabled = false;
  } else if (scenario == "paged_prefill") {
    config.service.kv_layout = KvLayout::Paged;
    config.service.scheduler_policy = SchedulerPolicy::PrefillFirst;
    config.service.prefix_cache_enabled = false;
  } else if (scenario == "paged" || scenario == "paged_decode") {
    config.service.kv_layout = KvLayout::Paged;
    config.service.scheduler_policy = SchedulerPolicy::DecodeFirst;
    config.service.prefix_cache_enabled = false;
  } else if (scenario == "prefix") {
    config.service.kv_layout = KvLayout::Paged;
    config.service.scheduler_policy = SchedulerPolicy::DecodeFirst;
    config.service.prefix_cache_enabled = true;
  } else {
    throw std::invalid_argument("Unknown scenario: " + scenario);
  }
  return config;
}

void write_comparison_csv(
    const std::filesystem::path &path,
    const std::vector<serving::SimulationResult> &results) {
  if (results.empty()) {
    return;
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Could not write comparison: " + path.string());
  }
  output << "scenario,tpot_slo_mode,realized_arrival_rate_per_second,busy_time_"
            "ms,busy_fraction,"
            "processed_prefill_tokens,slo_compliant_requests,"
            "slo_goodput_requests_per_second,throughput_tokens_per_second,ttft_"
            "p50_ms,ttft_p95_ms,"
            "itl_p50_ms,itl_p95_ms,mean_tpot_p95_ms,max_itl_p95_ms,average_"
            "block_utilization,"
            "peak_internal_waste_ratio,peak_external_fragmentation_ratio,peak_"
            "allocated_blocks,"
            "total_prefix_hit_tokens,prefix_hit_token_ratio,cache_evictions\n";
  output << std::fixed << std::setprecision(4);
  for (const auto &result : results) {
    output << result.scenario_name << ',' << result.tpot_slo_mode << ','
           << result.realized_arrival_rate_per_second << ','
           << result.busy_time_ms << ',' << result.busy_fraction << ','
           << result.processed_prefill_tokens << ','
           << result.slo_compliant_requests << ','
           << result.slo_goodput_requests_per_second << ','
           << result.throughput_tokens_per_second << ',' << result.ttft_p50_ms
           << ',' << result.ttft_p95_ms << ',' << result.itl_p50_ms << ','
           << result.itl_p95_ms << ',' << result.mean_tpot_p95_ms << ','
           << result.max_itl_p95_ms << ',' << result.average_block_utilization
           << ',' << result.peak_internal_waste_ratio << ','
           << result.peak_external_fragmentation_ratio << ','
           << result.peak_allocated_blocks << ','
           << result.total_prefix_hit_tokens << ','
           << result.prefix_hit_token_ratio << ',' << result.cache_evictions
           << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    CliOptions options;
    for (int index = 1; index < argc; ++index) {
      if (std::string(argv[index]) == "--config") {
        options.config_path = require_value(index, argc, argv);
      }
    }
    apply_config_file(options);
    parse_cli_overrides(options, argc, argv);

    if (!options.cuda_profile_output.empty()) {
      serving::CudaBenchmarkConfig benchmark_config;
      serving::write_cuda_decode_profile(options.cuda_profile_output,
                                         benchmark_config);
      std::cout << "Wrote CUDA decode profile to "
                << options.cuda_profile_output << '\n';
      return 0;
    }
    if (options.validate_cuda) {
      const serving::CudaValidationResult validation =
          serving::validate_cuda_paged_attention();
      std::cout << "CUDA validation passed: max_abs_error="
                << validation.max_absolute_error
                << ", mean_abs_error=" << validation.mean_absolute_error
                << '\n';
      return 0;
    }

    std::filesystem::create_directories(options.output_directory);
    const std::vector<serving::RequestSpec> workload =
        serving::generate_workload(options.workload);
    const serving::WorkloadStats workload_stats =
        serving::summarize_workload(workload);
    if (!workload_stats.arrivals_monotonic) {
      throw std::runtime_error(
          "Generated workload has non-monotonic arrival timestamps");
    }
    serving::write_trace_csv(options.output_directory / "workload.csv",
                             workload);
    std::cout << "workload: configured_rate="
              << options.workload.arrival_rate_per_second
              << " req/s, realized_rate="
              << workload_stats.realized_arrival_rate_per_second
              << " req/s, arrival_span=" << workload_stats.arrival_span_ms
              << " ms\n";

    std::vector<std::string> scenarios;
    if (options.scenario == "all") {
      scenarios = {"baseline", "paged", "prefix"};
    } else {
      scenarios = {options.scenario};
    }

    serving::LatencyModel latency_model =
        options.decode_profile_path.empty()
            ? serving::LatencyModel::analytical()
            : serving::LatencyModel::from_decode_profile(
                  options.decode_profile_path);
    std::vector<serving::SimulationResult> results;
    for (const std::string &scenario : scenarios) {
      const serving::SimulationConfig config =
          scenario_config(options.simulation, scenario);
      serving::ServingSimulator simulator(config, latency_model);
      serving::SimulationResult result = simulator.run(workload, scenario);
      serving::write_summary_csv(
          options.output_directory / (scenario + "_summary.csv"), result);
      serving::write_requests_csv(options.output_directory /
                                      (scenario + "_requests.csv"),
                                  simulator.requests(), config.service);
      results.push_back(result);

      std::cout << scenario
                << ": goodput=" << result.slo_goodput_requests_per_second
                << " req/s, throughput=" << result.throughput_tokens_per_second
                << " tok/s, TTFT p95=" << result.ttft_p95_ms
                << " ms, mean-TPOT p95=" << result.mean_tpot_p95_ms
                << " ms, max-ITL p95=" << result.max_itl_p95_ms << " ms\n";
    }
    if (scenarios.size() > 1) {
      write_comparison_csv(options.output_directory / "comparison.csv",
                           results);
    }
    std::cout << "Wrote trace and metrics to " << options.output_directory
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
