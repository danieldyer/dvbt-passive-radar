#pragma once
#include "complex_sample.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <vector>

namespace radar {
class Fft {
public:
    explicit Fft(std::size_t size);
    ~Fft();
    Fft(const Fft&) = delete;
    Fft& operator=(const Fft&) = delete;
    void transform(std::span<C> data, bool inverse = false) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct WaveInfo {
    std::uint32_t sample_rate{};
    std::uint64_t samples{}, data_offset{};
    bool wrapped_size{};
};
class WaveReader {
public:
    explicit WaveReader(const std::filesystem::path& path);
    WaveInfo info;
    void read(std::uint64_t first, std::span<C> output);
private:
    // Larger filebuf reads; storage outlives the stream that uses it.
    std::vector<char> file_buffer_;
    std::ifstream file_;
};

// Offline, centred symmetric FIR: no filter group delay in the sample coordinates.
// First and last half-filter supports are excluded, never silently zero padded.
struct ChannelizerTimings {
    double read{}, mix{}, forward{}, fold{}, inverse{};
};
class Channelizer {
public:
    Channelizer(WaveReader& input, double offset_hz, double start_seconds,
                double seconds);
    // Optional accumulated timings for the component benchmark; no clock reads otherwise.
    std::vector<C> next(ChannelizerTimings* timings = nullptr);
    std::uint64_t origin{}, output_count{}, produced{};
    std::uint32_t decimation{};
    static constexpr double output_rate = 8e6;
    static std::vector<float> prototype(double sample_rate,double cutoff_hz=3.65e6,unsigned half_at_56mhz=512);
private:
    static constexpr std::size_t block_size=8192;
    WaveReader& input_;
    double offset_;
    std::size_t fft_size_, hop_, half_;
    Fft fft_, inverse_{block_size};
    std::vector<C> response_, work_, oscillator_, folded_;
};

struct ResponseFit {
    std::vector<C> response;
    double relative_error{};
};
// Fit one fixed, short, causal log-response to a batch. A scalar is the safe default.
// Nonzero cepstral order is an explicit modelling assumption, not an echo separator.
ResponseFit fit_minimum_phase(std::span<const C> measured,
                             std::span<const C> reference,
                             std::size_t bins, unsigned order, unsigned iterations = 8);
} // namespace radar
