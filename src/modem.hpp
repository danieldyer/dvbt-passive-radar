#pragma once
#include "complex_sample.hpp"
#include "signal_constants.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace radar {
struct Symbol {
    std::vector<C> equalized, measured;
    double sample{}, sample_step{}, phase{}, frequency_hz{};
    int modulo4{};
    bool discontinuity{};
    // Complex noise variance E[|n|^2] in the measured FFT carrier units.
    // Pilot telemetry; hard-decision FEC does not weight metrics with it.
    float noise_variance{};
};
struct Tps {
    int frame{}, constellation{}, rate{}, guard{}, mode{}, hierarchy{};
    std::array<std::uint8_t,68> bits{};
};
struct Reconstruction {
    // Same symbol order as the input; invalid samples are masked, never removed.
    std::vector<C> carriers;
    std::vector<std::uint8_t> valid;
    std::size_t packets{}, bad_packets{}, corrected_bytes{};
    std::vector<std::array<std::uint8_t,188>> corrected_packets;
    std::vector<std::uint8_t> packet_valid;
};
struct ReconstructionTimings {
    double demap{}, viterbi{}, reed_solomon{}, reencode{}, remap{};
};
unsigned qam_bits(const Tps& mode); // Validates the supported non-hierarchical 8K mode.
std::vector<C> qam_constellation(int bits_per_symbol);
class CpuFec;
struct PilotPattern {
    std::array<float,6817> values{};
    std::vector<std::uint32_t> continual,tps;
};
class Modem {
public:
    explicit Modem(int guard = 16);
    ~Modem();
    std::vector<int> data_carriers(int modulo4) const;
    // Fused gather table: transmitted convolutional-code bit -> carrier*8+QAM bit.
    std::vector<std::uint32_t> payload_bit_indices(const Tps& mode) const;
    PilotPattern pilot_pattern() const;
    bool tps(std::span<const Symbol> frame, Tps& result) const;
    bool tps_bits(std::span<const std::uint8_t> bits,Tps& result) const;
    Reconstruction reconstruct(std::span<const Symbol> symbols, const Tps& mode,
                               ReconstructionTimings* timings = nullptr,CpuFec* decoder=nullptr) const;
    std::vector<C> synthesize(std::span<const std::uint8_t> randomized_packets,
                             const Tps& mode, std::size_t symbols) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace radar
