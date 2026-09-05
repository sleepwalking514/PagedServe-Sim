#include "serving/kv_cache.hpp"
#include "serving/latency_model.hpp"
#include "serving/simulator.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

serving::RequestRuntime make_request(const std::uint64_t id, const double arrival_ms,
                                     const std::vector<std::uint32_t>& prompt, const std::size_t output_tokens) {
    serving::RequestRuntime request;
    request.spec.id = id;
    request.spec.arrival_ms = arrival_ms;
    request.spec.prompt_tokens = prompt;
    request.spec.output_tokens = output_tokens;
    return request;
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
    expect(cache->admit(first) == serving::AdmissionResult::Admitted, "first request should fit");
    expect(cache->append_tokens(first, prompt.size()), "first prefill should allocate blocks");
    for (std::size_t block = 0; block < 2; ++block) {
        const auto begin = prompt.begin() + static_cast<std::ptrdiff_t>(block * model.block_tokens);
        cache->cache_completed_prompt_block(first, block,
                                            serving::hash_prompt_block(std::span(begin, begin + 4)));
    }
    cache->release(first);

    auto second = make_request(2, 1.0, prompt, 2);
    expect(cache->admit(second) == serving::AdmissionResult::Admitted, "cached request should fit");
    expect(second.prefix_hit_tokens == prompt.size(), "complete prefix blocks should be reused");
    expect(second.stored_tokens == prompt.size(), "cache hit should restore the logical context length");
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
    workload.push_back(make_request(1, 0.0, prompt, 4).spec);
    workload.push_back(make_request(2, 10.0, prompt, 4).spec);
    serving::ServingSimulator simulator(config, serving::LatencyModel::analytical());
    const serving::SimulationResult result = simulator.run(workload, "test");

    expect(result.completed_requests == 2, "simulator should complete both requests");
    expect(result.rejected_requests == 0, "small workload should not be rejected");
    expect(simulator.requests().at(1).prefix_hit_tokens == prompt.size(), "later request should hit cache");
}

}  // namespace

int main() {
    try {
        test_prefix_reuse();
        test_simulator_completes_requests();
        std::cout << "All serving tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
