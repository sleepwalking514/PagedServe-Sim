#include "serving/kv_cache.hpp"
#include "serving/latency_model.hpp"
#include "serving/metrics.hpp"
#include "serving/simulator.hpp"
#include "serving/workload.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
concept HasActualOutputTokens =
    requires(T value) { value.spec.actual_output_tokens; };
static_assert(!HasActualOutputTokens<serving::RequestRuntime>,
              "runtime requests must not expose the output-length oracle");

void expect(const bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

serving::RequestRuntime make_request(const std::uint64_t id,
                                     const double arrival_ms,
                                     const std::vector<std::uint32_t> &prompt,
                                     const std::size_t max_new_tokens) {
  serving::RequestRuntime request;
  request.spec.id = id;
  request.spec.arrival_ms = arrival_ms;
  request.spec.prompt_tokens = prompt;
  request.spec.max_new_tokens = max_new_tokens;
  return request;
}

serving::RequestSpec make_trace(const std::uint64_t id, const double arrival_ms,
                                const std::vector<std::uint32_t> &prompt,
                                const std::size_t max_new_tokens,
                                const std::size_t actual_output_tokens) {
  serving::RequestSpec spec;
  spec.id = id;
  spec.arrival_ms = arrival_ms;
  spec.prompt_tokens = prompt;
  spec.max_new_tokens = max_new_tokens;
  spec.actual_output_tokens = actual_output_tokens;
  return spec;
}

void test_prefix_reuse() {
  serving::ModelConfig model;
  model.block_tokens = 4;
  model.total_blocks = 16;
  serving::ServiceConfig service;
  service.kv_layout = serving::KvLayout::Paged;
  service.prefix_cache_enabled = true;
  auto cache = serving::make_kv_cache(model, service);

  const std::vector<std::uint32_t> prompt = {11, 12, 13, 14, 15, 16, 17, 18};
  auto first = make_request(1, 0.0, prompt, 2);
  expect(cache->admit(first) == serving::AdmissionResult::Admitted,
         "first request should fit");
  expect(cache->append_tokens(first, prompt.size()),
         "first prefill should allocate blocks");
  for (std::size_t block = 0; block < 2; ++block) {
    const auto begin = prompt.begin() +
                       static_cast<std::ptrdiff_t>(block * model.block_tokens);
    cache->cache_completed_prompt_block(
        first, block, serving::hash_prompt_block(std::span(begin, begin + 4)));
  }
  cache->release(first);

  auto second = make_request(2, 1.0, prompt, 2);
  expect(cache->admit(second) == serving::AdmissionResult::Admitted,
         "cached request should fit");
  expect(second.prefix_hit_tokens == prompt.size(),
         "complete prefix blocks should be reused");
  expect(second.stored_tokens == prompt.size(),
         "cache hit should restore the logical context length");
  cache->release(second);
}

void test_simulator_completes_requests() {
  serving::SimulationConfig config;
  config.model.block_tokens = 4;
  config.model.total_blocks = 64;
  config.service.token_budget = 16;
  config.service.prefill_chunk_tokens = 8;
  config.service.max_active_requests = 8;
  config.service.kv_layout = serving::KvLayout::Paged;
  config.service.scheduler_policy = serving::SchedulerPolicy::DecodeFirst;
  config.service.prefix_cache_enabled = true;

  const std::vector<std::uint32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<serving::RequestSpec> workload;
  workload.push_back(make_trace(1, 0.0, prompt, 8, 4));
  workload.push_back(make_trace(2, 10.0, prompt, 8, 4));
  serving::ServingSimulator simulator(config,
                                      serving::LatencyModel::analytical());
  const serving::SimulationResult result = simulator.run(workload, "test");

  expect(result.completed_requests == 2,
         "simulator should complete both requests");
  expect(result.rejected_requests == 0,
         "small workload should not be rejected");
  expect(simulator.requests().at(1).prefix_hit_tokens == prompt.size(),
         "later request should hit cache");

  std::vector<serving::RequestSpec> cold_workload;
  cold_workload.push_back(make_trace(3, 0.0, prompt, 8, 4));
  const serving::SimulationResult cold_result =
      simulator.run(cold_workload, "cold_again");
  expect(cold_result.completed_requests == 1, "simulator should be reusable");
  expect(
      simulator.requests().at(0).prefix_hit_tokens == 0,
      "a new run should not inherit prefix-cache state from the previous run");
}

void test_workload_arrivals_and_unique_prefixes() {
  serving::WorkloadConfig config;
  config.kind = serving::WorkloadKind::MixedBurst;
  config.request_count = 500;
  config.seed = 42;
  config.arrival_rate_per_second = 100.0;
  config.shared_prefix_tokens = 16;
  const auto requests = serving::generate_workload(config);
  const serving::WorkloadStats stats = serving::summarize_workload(requests);
  expect(stats.arrivals_monotonic,
         "mixed_burst timestamps must never move backwards");
  expect(
      stats.realized_arrival_rate_per_second > 70.0 &&
          stats.realized_arrival_rate_per_second < 130.0,
      "finite-trace realized rate should remain close to the configured mean");
  expect(requests.at(0).prefix_group != requests.at(3).prefix_group,
         "requests marked unique must not share a prefix group");
  expect(!std::equal(requests.at(0).prompt_tokens.begin(),
                     requests.at(0).prompt_tokens.begin() + 16,
                     requests.at(3).prompt_tokens.begin()),
         "unique requests must not accidentally reuse identical prefix tokens");
}

void test_external_fragmentation_metric() {
  serving::ModelConfig model;
  model.block_tokens = 1;
  model.total_blocks = 12;
  serving::ServiceConfig service;
  service.kv_layout = serving::KvLayout::Contiguous;
  service.prefix_cache_enabled = false;
  auto cache = serving::make_kv_cache(model, service);

  auto first = make_request(1, 0.0, {}, 3);
  auto middle = make_request(2, 0.0, {}, 3);
  auto last = make_request(3, 0.0, {}, 3);
  expect(cache->admit(first) == serving::AdmissionResult::Admitted,
         "first allocation should fit");
  expect(cache->admit(middle) == serving::AdmissionResult::Admitted,
         "middle allocation should fit");
  expect(cache->admit(last) == serving::AdmissionResult::Admitted,
         "last allocation should fit");
  cache->release(middle);
  const serving::KvCacheStats stats = cache->stats();
  expect(stats.available_blocks == 6,
         "six blocks should be free after releasing the middle run");
  expect(stats.largest_free_run_blocks == 3,
         "largest contiguous free run should be three blocks");
  expect(std::abs(stats.external_fragmentation_ratio - 0.5) < 1e-9,
         "external fragmentation must be based on largest free run, not block "
         "fill");
}

void test_explicit_tpot_slo_modes() {
  serving::RequestRuntime request = make_request(1, 0.0, {1, 2, 3, 4}, 4);
  request.phase = serving::RequestPhase::Finished;
  request.admitted_ms = 0.0;
  request.first_token_ms = 0.1;
  request.finished_ms = 1.3;
  request.generated_tokens = 4;
  request.inter_token_latencies_ms = {0.1, 0.1, 1.0};
  serving::ServiceConfig service;
  service.ttft_slo_ms = 1.0;
  service.tpot_slo_ms = 0.5;
  service.tpot_slo_mode = serving::TpotSloMode::Mean;
  const std::vector<serving::RequestRuntime> requests = {request};
  const auto mean_result = serving::summarize_result(
      "mean", "test", requests, 2.0, 2.0, 0, 0.0, 0.0, 0.0, 0, 1, 0, service);
  expect(mean_result.slo_compliant_requests == 1,
         "mean TPOT should pass for this request");
  service.tpot_slo_mode = serving::TpotSloMode::Max;
  const auto max_result = serving::summarize_result(
      "max", "test", requests, 2.0, 2.0, 0, 0.0, 0.0, 0.0, 0, 1, 0, service);
  expect(max_result.slo_compliant_requests == 0,
         "max ITL should fail for the same request");
}

void test_unknown_output_length_and_kv_growth() {
  serving::SimulationConfig config;
  config.model.block_tokens = 4;
  config.model.total_blocks = 2;
  config.service.token_budget = 16;
  config.service.prefill_chunk_tokens = 8;
  config.service.max_active_requests = 4;
  config.service.prefix_cache_enabled = false;
  config.service.scheduler_policy = serving::SchedulerPolicy::DecodeFirst;
  config.service.kv_layout = serving::KvLayout::Paged;

  const auto trace = make_trace(1, 0.0, {1, 2, 3, 4}, 8, 5);
  serving::ServingSimulator paged(config, serving::LatencyModel::analytical());
  const auto paged_result = paged.run({trace}, "paged_growth");
  expect(paged_result.admitted_requests == 1,
         "paged admission should use prompt size, not the final output length");
  expect(paged_result.kv_growth_rejected_requests == 1,
         "paged growth must report capacity exhaustion explicitly");
  expect(paged_result.all_generated_tokens == 4,
         "a failed fifth decode token must not be counted as generated");

  config.model.total_blocks = 3;
  serving::ServingSimulator enough_pages(config,
                                         serving::LatencyModel::analytical());
  const auto enough_result = enough_pages.run({trace}, "paged_enough");
  expect(enough_result.completed_requests == 1 &&
             enough_pages.requests().at(0).generated_tokens == 5,
         "paged allocation should grow one physical block at a time");

  config.model.total_blocks = 2;
  config.service.kv_layout = serving::KvLayout::Contiguous;
  serving::ServingSimulator contiguous(config,
                                       serving::LatencyModel::analytical());
  const auto contiguous_result = contiguous.run({trace}, "contiguous_max");
  expect(contiguous_result.capacity_rejected_requests == 1 &&
             contiguous_result.admitted_requests == 0,
         "contiguous allocation should reserve prompt plus max_new_tokens");

  auto invalid = trace;
  invalid.actual_output_tokens = 9;
  try {
    (void)enough_pages.run({invalid}, "invalid_truth");
    throw std::runtime_error("invalid trace unexpectedly passed validation");
  } catch (const std::invalid_argument &) {
    // The completion oracle cannot exceed the client-visible generation cap.
  }
}

} // namespace

int main() {
  try {
    test_prefix_reuse();
    test_simulator_completes_requests();
    test_workload_arrivals_and_unique_prefixes();
    test_external_fragmentation_metric();
    test_explicit_tpot_slo_modes();
    test_unknown_output_length_and_kv_growth();
    std::cout << "All serving tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
