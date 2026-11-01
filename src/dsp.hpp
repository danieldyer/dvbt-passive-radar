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
    Fft(const Fft &) = delete;
    Fft &operator=(const Fft &) = delete;
    void transform(std::span<C> data, bool inverse = false) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct WaveInfo {
    std::uint32_t sample_rate{};
    std::uint64_t samples{}, data_offset{};
    bool wrapped_size{};
    std::uint16_t sample_bytes = 8; // One complex sample: cf32 or signed ci16.
};
class WaveReader {
  public:
    explicit WaveReader(const std::filesystem::path &path);
    WaveInfo info;
    void read(std::uint64_t first, std::span<C> output);

  private:
    // Larger filebuf reads; storage outlives the stream that uses it.
    std::vector<char> file_buffer_;
    std::ifstream file_;
    std::vector<std::int16_t> integer_samples_;
};

// Conservative Kaiser prototype for uncommon ADC rates above the stored tables.
std::vector<float> kaiser_prototype(double sample_rate, double cutoff_hz, unsigned half_at_56mhz);
// Real-orthonormal coordinates for the same causal log-response space. The
// imaginary companions undergo exactly the same basis transformations.
std::vector<std::complex<double>> minimum_phase_basis(std::size_t bins, unsigned order);
} // namespace radar
