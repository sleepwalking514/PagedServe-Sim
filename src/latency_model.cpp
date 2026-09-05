#include "serving/latency_model.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace serving {

LatencyModel LatencyModel::analytical() {
    LatencyModel model;
    model.source_ = "analytical (replace with CUDA profile for hardware claims)";
    return model;
}

LatencyModel LatencyModel::from_decode_profile(const std::filesystem::path& path) {
    LatencyModel model;
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open decode profile: " + path.string());
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::stringstream stream(line);
        std::string batch_text;
        std::string sequence_text;
        std::string latency_text;
        if (!std::getline(stream, batch_text, ',') || !std::getline(stream, sequence_text, ',') ||
            !std::getline(stream, latency_text, ',')) {
            continue;
        }
        if (batch_text.empty() || !std::isdigit(static_cast<unsigned char>(batch_text.front()))) {
            continue;
        }
        model.decode_profile_.push_back({static_cast<std::size_t>(std::stoull(batch_text)),
                                         static_cast<std::size_t>(std::stoull(sequence_text)),
                                         std::stod(latency_text)});
    }
    if (model.decode_profile_.empty()) {
        throw std::runtime_error("Decode profile has no data rows: " + path.string());
    }
    model.source_ = path.string();
    return model;
}

double LatencyModel::decode_latency_ms(const std::size_t batch_size,
                                       const std::size_t average_sequence_length) const {
    if (!decode_profile_.empty()) {
        const auto closest = std::min_element(
            decode_profile_.begin(), decode_profile_.end(),
            [batch_size, average_sequence_length](const DecodeProfilePoint& lhs, const DecodeProfilePoint& rhs) {
                const auto lhs_distance = std::llabs(static_cast<long long>(lhs.batch_size) -
                                                      static_cast<long long>(batch_size)) +
                                          std::llabs(static_cast<long long>(lhs.sequence_length) -
                                                     static_cast<long long>(average_sequence_length));
                const auto rhs_distance = std::llabs(static_cast<long long>(rhs.batch_size) -
                                                      static_cast<long long>(batch_size)) +
                                          std::llabs(static_cast<long long>(rhs.sequence_length) -
                                                     static_cast<long long>(average_sequence_length));
                return lhs_distance < rhs_distance;
            });
        return closest->latency_ms;
    }

    // CPU-only fallback; this is not a hardware measurement.
    const double batch = static_cast<double>(std::max<std::size_t>(batch_size, 1));
    const double context = static_cast<double>(average_sequence_length);
    return 0.035 + 0.0035 * std::sqrt(batch) + 0.000018 * context * std::sqrt(batch);
}

double LatencyModel::prefill_latency_ms(const std::size_t token_count,
                                        const std::size_t active_sequences) const {
    const double tokens = static_cast<double>(token_count);
    const double sequences = static_cast<double>(std::max<std::size_t>(active_sequences, 1));
    return 0.008 + 0.00020 * tokens + 0.006 * std::sqrt(sequences);
}

const std::string& LatencyModel::source() const {
    return source_;
}

}  // namespace serving
