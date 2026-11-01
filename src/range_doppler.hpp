#pragma once
#include "dsp.hpp"
#include <ostream>
namespace radar {
struct DopplerMap {
    std::vector<C> values; // range-major, ascending signed Doppler
    std::size_t ranges{}, frequencies{}, samples{}, fft_size{};
    bool dc_removed = false;
    std::uint64_t sequence{}, segment{};
    double range_step = 299792458.0 / 8e6, frequency_start{}, frequency_step{};
    double time_start{}, time_end{}, integration{}, coherence{};
    double history_limit{}, requested_integration{};
    std::string window = "hann", dc_method = "hann_weighted_mean", motion_model = "stationary_range";
    double enbw{}, phase_epoch{};
    double rf_hz{}, reference_origin_samples{};
    unsigned channel{};
    std::string stationary_prediction = "none";
    double background_start{}, background_end{};
    std::string amplitude_units = "cross_correlation";
    std::vector<float> noise_range, noise_doppler;
    std::string json() const;
    void write(std::ostream &stream) const;
    void write(const std::filesystem::path &path) const;
};
} // namespace radar
