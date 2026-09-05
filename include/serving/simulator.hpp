#pragma once

#include "serving/kv_cache.hpp"
#include "serving/latency_model.hpp"
#include "serving/metrics.hpp"
#include "serving/types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace serving {

class ServingSimulator {
public:
    ServingSimulator(SimulationConfig config, LatencyModel latency_model);

    [[nodiscard]] SimulationResult run(std::vector<RequestSpec> workload, const std::string& scenario_name);
    [[nodiscard]] const std::vector<RequestRuntime>& requests() const;

private:
    struct ScheduledWork {
        std::size_t request_index = 0;
        std::size_t prefill_tokens = 0;
        bool decode_token = false;
    };

    void admit_waiting_requests();
    void reject_expired_waiting_requests();
    [[nodiscard]] std::vector<ScheduledWork> schedule_iteration();
    void complete_iteration(const std::vector<ScheduledWork>& work, double completion_time_ms);
    [[nodiscard]] std::size_t active_request_count() const;
    [[nodiscard]] std::size_t average_active_context_length() const;
    [[nodiscard]] bool all_requests_terminal() const;

    SimulationConfig config_;
    LatencyModel latency_model_;
    std::unique_ptr<KvCache> kv_cache_;
    std::vector<RequestRuntime> requests_;
    double now_ms_ = 0.0;
    std::size_t next_arrival_index_ = 0;
    std::size_t decode_round_robin_offset_ = 0;
    double utilization_time_integral_ = 0.0;
    double observed_duration_ms_ = 0.0;
    double peak_fragmentation_ratio_ = 0.0;
};

}  // namespace serving
