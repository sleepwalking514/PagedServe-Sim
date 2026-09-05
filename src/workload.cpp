#include "serving/workload.hpp"

#include "serving/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace serving {
namespace {

std::size_t sample_prompt_length(const WorkloadKind kind, std::mt19937& generator) {
    switch (kind) {
    case WorkloadKind::Short: {
        std::uniform_int_distribution<std::size_t> distribution(32, 192);
        return distribution(generator);
    }
    case WorkloadKind::PrefixHeavy: {
        std::uniform_int_distribution<std::size_t> distribution(256, 768);
        return distribution(generator);
    }
    case WorkloadKind::MixedBurst: {
        std::bernoulli_distribution long_request(0.30);
        if (long_request(generator)) {
            std::uniform_int_distribution<std::size_t> distribution(512, 1'536);
            return distribution(generator);
        }
        std::uniform_int_distribution<std::size_t> distribution(32, 256);
        return distribution(generator);
    }
    }
    return 128;
}

std::size_t sample_output_length(const WorkloadKind kind, std::mt19937& generator) {
    switch (kind) {
    case WorkloadKind::Short: {
        std::uniform_int_distribution<std::size_t> distribution(16, 96);
        return distribution(generator);
    }
    case WorkloadKind::PrefixHeavy: {
        std::uniform_int_distribution<std::size_t> distribution(64, 384);
        return distribution(generator);
    }
    case WorkloadKind::MixedBurst: {
        std::bernoulli_distribution long_request(0.30);
        if (long_request(generator)) {
            std::uniform_int_distribution<std::size_t> distribution(256, 768);
            return distribution(generator);
        }
        std::uniform_int_distribution<std::size_t> distribution(16, 128);
        return distribution(generator);
    }
    }
    return 64;
}

std::vector<std::uint32_t> make_prompt(const std::size_t length, const std::size_t shared_length,
                                       const std::vector<std::uint32_t>& shared_prefix,
                                       std::mt19937& generator) {
    std::vector<std::uint32_t> prompt;
    prompt.reserve(length);
    const std::size_t prefix_length = std::min({length, shared_length, shared_prefix.size()});
    prompt.insert(prompt.end(), shared_prefix.begin(),
                  shared_prefix.begin() + static_cast<std::ptrdiff_t>(prefix_length));
    std::uniform_int_distribution<std::uint32_t> token_distribution(1, 32'000);
    while (prompt.size() < length) {
        prompt.push_back(token_distribution(generator));
    }
    return prompt;
}

}  // namespace

std::vector<RequestSpec> generate_workload(const WorkloadConfig& config) {
    if (config.request_count == 0) {
        return {};
    }

    std::mt19937 generator(config.seed);
    const double rate_per_ms = std::max(config.arrival_rate_per_second / 1'000.0, 0.000001);
    std::exponential_distribution<double> interarrival(rate_per_ms);
    std::bernoulli_distribution burst_request(config.kind == WorkloadKind::MixedBurst ? 0.18 : 0.0);
    std::unordered_map<std::string, std::vector<std::uint32_t>> shared_prefixes;
    std::vector<RequestSpec> requests;
    requests.reserve(config.request_count);

    double arrival_ms = 0.0;
    for (std::size_t index = 0; index < config.request_count; ++index) {
        if (index > 0) {
            arrival_ms += interarrival(generator);
        }
        if (burst_request(generator)) {
            arrival_ms = std::floor(arrival_ms / 100.0) * 100.0;
        }

        const bool reuse_prefix = config.kind == WorkloadKind::PrefixHeavy ||
                                  (config.kind == WorkloadKind::MixedBurst && index % 3 != 0);
        const std::size_t group_index = reuse_prefix ? index % 8 : index;
        const std::string group = reuse_prefix ? "prefix_" + std::to_string(group_index) : "unique";
        auto prefix_it = shared_prefixes.find(group);
        if (prefix_it == shared_prefixes.end()) {
            std::vector<std::uint32_t> prefix(config.shared_prefix_tokens);
            std::uniform_int_distribution<std::uint32_t> token_distribution(1, 32'000);
            for (std::uint32_t& token : prefix) {
                token = token_distribution(generator);
            }
            prefix_it = shared_prefixes.emplace(group, std::move(prefix)).first;
        }

        const std::size_t prompt_length = sample_prompt_length(config.kind, generator);
        RequestSpec request;
        request.id = static_cast<std::uint64_t>(index);
        request.arrival_ms = arrival_ms;
        request.prefix_group = group;
        request.prompt_tokens = make_prompt(prompt_length, config.shared_prefix_tokens, prefix_it->second, generator);
        request.output_tokens = sample_output_length(config.kind, generator);
        requests.push_back(std::move(request));
    }
    return requests;
}

void write_trace_csv(const std::filesystem::path& path, const std::vector<RequestSpec>& requests) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not write workload trace: " + path.string());
    }
    output << "request_id,arrival_ms,prompt_tokens,output_tokens,prefix_group\n";
    output << std::fixed << std::setprecision(4);
    for (const RequestSpec& request : requests) {
        output << request.id << ',' << request.arrival_ms << ',' << request.prompt_tokens.size() << ','
               << request.output_tokens << ',' << request.prefix_group << '\n';
    }
}

std::string to_string(const WorkloadKind kind) {
    switch (kind) {
    case WorkloadKind::Short:
        return "short";
    case WorkloadKind::MixedBurst:
        return "mixed_burst";
    case WorkloadKind::PrefixHeavy:
        return "prefix_heavy";
    }
    return "mixed_burst";
}

WorkloadKind workload_kind_from_string(const std::string& value) {
    if (value == "short") {
        return WorkloadKind::Short;
    }
    if (value == "prefix_heavy") {
        return WorkloadKind::PrefixHeavy;
    }
    if (value == "mixed_burst") {
        return WorkloadKind::MixedBurst;
    }
    throw std::invalid_argument("Unknown workload kind: " + value);
}

}  // namespace serving
