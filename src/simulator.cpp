#include "serving/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace serving {

ServingSimulator::ServingSimulator(SimulationConfig config, LatencyModel latency_model)
    : config_(std::move(config)), latency_model_(std::move(latency_model)),
      kv_cache_(make_kv_cache(config_.model, config_.service)) {
    if (config_.model.block_tokens == 0 || config_.model.total_blocks == 0 ||
        config_.service.token_budget == 0 || config_.service.prefill_chunk_tokens == 0 ||
        config_.service.max_active_requests == 0) {
        throw std::invalid_argument("Block size, cache capacity, and scheduling budgets must be positive");
    }
}

SimulationResult ServingSimulator::run(std::vector<RequestSpec> workload, const std::string& scenario_name) {
    std::stable_sort(workload.begin(), workload.end(), [](const RequestSpec& lhs, const RequestSpec& rhs) {
        if (lhs.arrival_ms != rhs.arrival_ms) {
            return lhs.arrival_ms < rhs.arrival_ms;
        }
        return lhs.id < rhs.id;
    });

    requests_.clear();
    requests_.reserve(workload.size());
    for (RequestSpec& spec : workload) {
        RequestRuntime runtime;
        runtime.spec = std::move(spec);
        requests_.push_back(std::move(runtime));
    }
    now_ms_ = 0.0;
    next_arrival_index_ = 0;
    decode_round_robin_offset_ = 0;
    utilization_time_integral_ = 0.0;
    observed_duration_ms_ = 0.0;
    peak_fragmentation_ratio_ = 0.0;

    while (!all_requests_terminal()) {
        if (active_request_count() == 0 && next_arrival_index_ < requests_.size() &&
            requests_[next_arrival_index_].spec.arrival_ms > now_ms_) {
            now_ms_ = requests_[next_arrival_index_].spec.arrival_ms;
        }

        while (next_arrival_index_ < requests_.size() &&
               requests_[next_arrival_index_].spec.arrival_ms <= now_ms_) {
            ++next_arrival_index_;
        }
        reject_expired_waiting_requests();
        admit_waiting_requests();

        if (active_request_count() == 0) {
            if (next_arrival_index_ < requests_.size()) {
                continue;
            }
            break;
        }

        const std::vector<ScheduledWork> work = schedule_iteration();
        if (work.empty()) {
            throw std::runtime_error("Scheduler produced no work while requests were active");
        }

        std::size_t prefill_tokens = 0;
        std::size_t decode_tokens = 0;
        std::size_t prefill_sequences = 0;
        for (const ScheduledWork& item : work) {
            prefill_tokens += item.prefill_tokens;
            if (item.prefill_tokens > 0) {
                ++prefill_sequences;
            }
            if (item.decode_token) {
                ++decode_tokens;
            }
        }

        const double prefill_ms = prefill_tokens > 0
                                      ? latency_model_.prefill_latency_ms(prefill_tokens, prefill_sequences)
                                      : 0.0;
        const double decode_ms = decode_tokens > 0
                                     ? latency_model_.decode_latency_ms(decode_tokens, average_active_context_length())
                                     : 0.0;
        const double iteration_ms = prefill_ms + decode_ms;
        const KvCacheStats before = kv_cache_->stats();
        utilization_time_integral_ += before.physical_block_utilization * iteration_ms;
        peak_fragmentation_ratio_ = std::max(peak_fragmentation_ratio_, before.fragmentation_ratio);
        observed_duration_ms_ += iteration_ms;
        now_ms_ += iteration_ms;
        complete_iteration(work, now_ms_);
    }

    const KvCacheStats final_stats = kv_cache_->stats();
    peak_fragmentation_ratio_ = std::max(peak_fragmentation_ratio_, final_stats.fragmentation_ratio);
    const double average_utilization = observed_duration_ms_ > 0.0
                                           ? utilization_time_integral_ / observed_duration_ms_
                                           : 0.0;
    return summarize_result(scenario_name, latency_model_.source(), requests_, now_ms_, average_utilization,
                            peak_fragmentation_ratio_, final_stats.cache_evictions, config_.service);
}

const std::vector<RequestRuntime>& ServingSimulator::requests() const {
    return requests_;
}

void ServingSimulator::admit_waiting_requests() {
    while (active_request_count() < config_.service.max_active_requests) {
        bool found_waiting = false;
        for (RequestRuntime& request : requests_) {
            if (request.phase != RequestPhase::Queued || request.spec.arrival_ms > now_ms_) {
                continue;
            }
            found_waiting = true;
            const AdmissionResult admission = kv_cache_->admit(request);
            if (admission == AdmissionResult::Impossible) {
                request.phase = RequestPhase::Rejected;
                request.rejection_reason = "request exceeds KV capacity";
                continue;
            }
            if (admission == AdmissionResult::TemporarilyUnavailable) {
                // Keep FIFO order: a later request cannot jump over a request waiting for memory.
                return;
            }
            request.admitted_ms = now_ms_;
            if (request.prefetched_tokens == request.prompt_tokens()) {
                if (request.spec.output_tokens == 0) {
                    request.phase = RequestPhase::Finished;
                    request.finished_ms = now_ms_;
                    kv_cache_->release(request);
                } else {
                    request.phase = RequestPhase::Decode;
                }
            } else {
                request.phase = RequestPhase::Prefill;
            }
            break;
        }
        if (!found_waiting) {
            return;
        }
    }
}

void ServingSimulator::reject_expired_waiting_requests() {
    for (RequestRuntime& request : requests_) {
        if (request.phase != RequestPhase::Queued || request.spec.arrival_ms > now_ms_) {
            continue;
        }
        if (now_ms_ - request.spec.arrival_ms > config_.service.ttft_slo_ms) {
            request.phase = RequestPhase::Rejected;
            request.rejection_reason = "waiting time exceeded TTFT SLO";
        }
    }
}

std::vector<ServingSimulator::ScheduledWork> ServingSimulator::schedule_iteration() {
    std::vector<ScheduledWork> work;
    std::size_t remaining_budget = config_.service.token_budget;
    if (remaining_budget == 0) {
        return work;
    }

    const auto schedule_prefill = [&]() {
        for (std::size_t index = 0; index < requests_.size() && remaining_budget > 0; ++index) {
            RequestRuntime& request = requests_[index];
            if (request.phase != RequestPhase::Prefill) {
                continue;
            }
            const std::size_t remaining_prompt = request.prompt_tokens() - request.prefetched_tokens;
            const std::size_t chunk = std::min({remaining_prompt, config_.service.prefill_chunk_tokens,
                                                remaining_budget});
            if (chunk > 0) {
                work.push_back({index, chunk, false});
                remaining_budget -= chunk;
            }
        }
    };

    const auto schedule_decode = [&]() {
        std::vector<std::size_t> decode_candidates;
        for (std::size_t index = 0; index < requests_.size(); ++index) {
            if (requests_[index].phase == RequestPhase::Decode) {
                decode_candidates.push_back(index);
            }
        }
        if (decode_candidates.empty()) {
            return;
        }
        const std::size_t offset = decode_round_robin_offset_ % decode_candidates.size();
        std::rotate(decode_candidates.begin(), decode_candidates.begin() + static_cast<std::ptrdiff_t>(offset),
                    decode_candidates.end());
        for (const std::size_t index : decode_candidates) {
            if (remaining_budget == 0) {
                break;
            }
            work.push_back({index, 0, true});
            --remaining_budget;
        }
        decode_round_robin_offset_ = (offset + 1) % decode_candidates.size();
    };

    if (config_.service.scheduler_policy == SchedulerPolicy::DecodeFirst) {
        schedule_decode();
        schedule_prefill();
    } else {
        schedule_prefill();
        schedule_decode();
    }
    return work;
}

void ServingSimulator::complete_iteration(const std::vector<ScheduledWork>& work, const double completion_time_ms) {
    for (const ScheduledWork& item : work) {
        RequestRuntime& request = requests_[item.request_index];
        if (item.prefill_tokens > 0) {
            const std::size_t old_tokens = request.prefetched_tokens;
            if (!kv_cache_->append_tokens(request, item.prefill_tokens)) {
                throw std::runtime_error("KV cache ran out of reserved blocks during prefill");
            }
            request.prefetched_tokens += item.prefill_tokens;
            request.stored_tokens = request.prefetched_tokens;
            const std::size_t first_full_block = old_tokens / config_.model.block_tokens;
            const std::size_t full_block_end = request.prefetched_tokens / config_.model.block_tokens;
            for (std::size_t block_index = first_full_block; block_index < full_block_end; ++block_index) {
                const auto begin = request.spec.prompt_tokens.begin() +
                                   static_cast<std::ptrdiff_t>(block_index * config_.model.block_tokens);
                const auto end = begin + static_cast<std::ptrdiff_t>(config_.model.block_tokens);
                kv_cache_->cache_completed_prompt_block(request, block_index, hash_prompt_block(std::span(begin, end)));
            }
            if (request.prefetched_tokens == request.prompt_tokens()) {
                if (request.spec.output_tokens == 0) {
                    request.phase = RequestPhase::Finished;
                    request.finished_ms = completion_time_ms;
                    kv_cache_->release(request);
                } else {
                    request.phase = RequestPhase::Decode;
                }
            }
        }

        if (!item.decode_token) {
            continue;
        }
        if (!kv_cache_->append_tokens(request, 1)) {
            throw std::runtime_error("KV cache ran out of reserved blocks during decode");
        }
        ++request.generated_tokens;
        if (std::isfinite(request.last_token_ms)) {
            request.inter_token_latencies_ms.push_back(completion_time_ms - request.last_token_ms);
        }
        if (!std::isfinite(request.first_token_ms)) {
            request.first_token_ms = completion_time_ms;
        }
        request.last_token_ms = completion_time_ms;
        if (request.generated_tokens >= request.spec.output_tokens) {
            request.phase = RequestPhase::Finished;
            request.finished_ms = completion_time_ms;
            kv_cache_->release(request);
        }
    }
}

std::size_t ServingSimulator::active_request_count() const {
    return static_cast<std::size_t>(std::count_if(
        requests_.begin(), requests_.end(), [](const RequestRuntime& request) { return request.is_active(); }));
}

std::size_t ServingSimulator::average_active_context_length() const {
    std::size_t active = 0;
    std::size_t total_context = 0;
    for (const RequestRuntime& request : requests_) {
        if (request.is_active()) {
            ++active;
            total_context += request.stored_tokens;
        }
    }
    return active == 0 ? 0 : total_context / active;
}

bool ServingSimulator::all_requests_terminal() const {
    return std::all_of(requests_.begin(), requests_.end(), [](const RequestRuntime& request) {
        return request.phase == RequestPhase::Finished || request.phase == RequestPhase::Rejected;
    });
}

}  // namespace serving
