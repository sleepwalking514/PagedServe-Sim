#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace serving {

struct DecodeProfilePoint {
    std::size_t batch_size = 0;
    std::size_t sequence_length = 0;
    double latency_ms = 0.0;
};

class LatencyModel {
public:
    static LatencyModel analytical();
    static LatencyModel from_decode_profile(const std::filesystem::path& path);

    [[nodiscard]] double decode_latency_ms(std::size_t batch_size, std::size_t average_sequence_length) const;
    [[nodiscard]] double prefill_latency_ms(std::size_t token_count, std::size_t active_sequences) const;
    [[nodiscard]] const std::string& source() const;

private:
    std::vector<DecodeProfilePoint> decode_profile_;
    std::string source_ = "analytical";
};

}  // namespace serving
