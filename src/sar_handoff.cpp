#include "sar_handoff.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace radar {
namespace {
constexpr std::uint64_t exact_limit = std::uint64_t{1} << 53;

}
void SarCapture::validate() const {
    const auto raw_rate = source_rate ? source_rate : input_rate;
    if (!input_rate || input_rate % 8'000'000 || input_rate > 128'000'000 ||
        decimation != input_rate / 8'000'000 || origin >= exact_limit || !std::isfinite(recording_rf_hz) ||
        recording_rf_hz <= 0 || !std::isfinite(clock_phase_degrees) || clock_phase_degrees <= 0 ||
        clock_phase_degrees > 30 || channels.empty() || channels.size() > 64 || raw_rate < 8'000'000 ||
        raw_rate > input_rate)
        throw std::invalid_argument("Invalid SAR capture coordinates");
    for (const auto &c : channels)
        if (!std::isfinite(c.rf_hz) || c.rf_hz <= 0 || !std::isfinite(c.nco_hz) ||
            std::abs(c.nco_hz) + 3.5e6 > raw_rate / 2.0 ||
            (c.guard != 4 && c.guard != 8 && c.guard != 16 && c.guard != 32))
            throw std::invalid_argument("Invalid SAR channel coordinates");
}
SarBatch::SarBatch(const SarCapture &c, const SpectralBatch &s, std::vector<ClockSegment> trajectory)
    : capture(c), spectra(s), clock(std::move(trajectory)) {
    capture.validate();
    const auto &f = spectra.frame;
    if (f.channel >= capture.channels.size() || f.first >= f.last || f.last > f.symbols.size() ||
        f.carriers.count != f.symbols.size() || spectra.transmitted.count != f.symbols.size() ||
        spectra.residual.count != f.symbols.size() || !std::isfinite(spectra.estimate.reference_origin) ||
        clock.empty())
        throw std::invalid_argument("Invalid SAR spectral batch");
    for (std::size_t i = 0; i < clock.size(); ++i) {
        const auto &t = clock[i];
        if (t.start >= exact_limit || !std::isfinite(t.offset) || !std::isfinite(t.rate) || t.rate <= 0 ||
            (i && (t.start <= clock[i - 1].start ||
                   std::abs(t.offset - clock[i - 1].offset_at(double(t.start))) > 1e-7)))
            throw std::invalid_argument("Discontinuous SAR clock trajectory");
    }
    for (std::size_t row = 0; row < f.symbols.size(); ++row) {
        const auto &symbol = f.symbols[row];
        if (symbol.channel != f.channel || symbol.segment != f.segment ||
            !std::isfinite(symbol.window.phase) || !std::isfinite(symbol.window.frequency_hz) ||
            !std::isfinite(symbol.source_offset_samples) || !std::isfinite(symbol.source_rate_error) ||
            symbol.modulo4 > 3)
            throw std::invalid_argument("Inconsistent SAR symbol coordinates");
        if (corrected_adc(row) < clock.front().start)
            throw std::invalid_argument("SAR clock trajectory does not cover the spectral batch");
    }
}
std::uint64_t SarBatch::corrected_adc(std::size_t row) const {
    const auto sample = spectra.frame.symbols.at(row).window.sample;
    const auto available = (exact_limit - 1 - capture.origin) / capture.decimation;
    if (available < 8192 || sample > available - 8192)
        throw std::out_of_range("SAR sample coordinate exceeds exact double range");
    return capture.origin + sample * capture.decimation;
}
double SarBatch::seconds(std::size_t row) const {
    return double(corrected_adc(row)) / capture.input_rate;
}
double SarBatch::original_adc(std::size_t row, double within_window) const {
    if (!std::isfinite(within_window) || within_window < 0 || within_window > 8192)
        throw std::out_of_range("SAR within-window coordinate out of range");
    const auto n = double(corrected_adc(row)) + within_window * capture.decimation;
    auto it = std::upper_bound(clock.begin(), clock.end(), n,
                               [](double x, const ClockSegment &s) { return x < double(s.start); });
    --it;
    return (n + it->offset_at(n)) *
           (capture.source_rate ? double(capture.source_rate) / capture.input_rate : 1);
}
double SarBatch::carrier_rf_hz(unsigned k) const {
    if (k >= 6817)
        throw std::out_of_range("SAR carrier out of range");
    return capture.channels[spectra.frame.channel].rf_hz + (int(k) - 3408) * (8e6 / 8192);
}
} // namespace radar
