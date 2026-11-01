#pragma once
#include "complex_sample.hpp"
#include "signal_constants.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace radar {
// Reference/diagnostic carrier dump; production soft demapping stays on GPU.
struct Symbol {
    std::vector<C> equalized, measured;
    double sample{}, sample_step{}, phase{}, frequency_hz{};
    int modulo4{};
    bool discontinuity{};
    // Complex noise variance E[|n|^2] in the measured FFT carrier units.
    // Diagnostic pilot telemetry; production soft metrics use GPU covariance.
    float noise_variance{};
};
struct Tps {
    int frame{}, constellation{}, rate{}, guard{}, mode{}, hierarchy{};
    std::array<std::uint8_t, 68> bits{};
};
struct Reconstruction {
    // Same symbol order as the input; invalid samples are masked, never removed.
    std::vector<C> carriers;
    std::vector<std::uint8_t> valid;
    std::size_t packets{}, bad_packets{}, corrected_bytes{};
    std::vector<std::array<std::uint8_t, 188>> corrected_packets;
    std::vector<std::uint8_t> packet_valid;
};
unsigned qam_bits(const Tps &mode); // Validates the supported non-hierarchical 8K mode.
std::vector<C> qam_constellation(int bits_per_symbol);
struct PilotPattern {
    std::array<float, 6817> values{};
    std::vector<std::uint32_t> continual, tps;
};
class Modem {
  public:
    explicit Modem(int guard = 16);
    std::vector<int> data_carriers(int modulo4) const;
    // Fused gather table: transmitted convolutional-code bit -> carrier*8+QAM bit.
    std::vector<std::uint32_t> payload_bit_indices(const Tps &mode) const;
    PilotPattern pilot_pattern() const;
    bool tps_bits(std::span<const std::uint8_t> bits, Tps &result) const;
    // Deterministic known waveform for acquisition warmup and qualification.
    std::vector<C> synthesize(std::span<const std::uint8_t> packets, const Tps &, std::size_t symbols) const;
};

} // namespace radar
