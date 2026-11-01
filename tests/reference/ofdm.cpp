#include "ofdm.hpp"
#include <bit>
#include <cmath>
#include <numbers>
namespace radar::validation {
std::vector<Symbol> ofdm_symbols(const OfdmBatch &batch, std::span<const OfdmWindow> windows) {
    if ((batch.carrier_stride != 6817 && batch.carrier_stride != 8192) ||
        (batch.carrier_stride == 8192 &&
         (!batch.phase_corrections.mapped || batch.phase_corrections.bytes < batch.count * sizeof(C))) ||
        windows.size() != batch.count || !batch.measured.mapped || !batch.pilots.mapped ||
        batch.measured.bytes < batch.count * batch.carrier_stride * sizeof(C) || batch.pilot_stride < 2 ||
        batch.pilots.bytes < batch.count * batch.pilot_stride * sizeof(C))
        throw std::invalid_argument("Invalid OFDM reference readback");
    static const auto pattern = Modem{}.pilot_pattern();
    const auto *y = static_cast<const C *>(batch.measured.mapped);
    const auto *stats = static_cast<const C *>(batch.pilots.mapped);
    std::vector<Symbol> result(batch.count);
    constexpr double tau = 2 * std::numbers::pi;
    for (std::size_t s = 0; s < batch.count; ++s) {
        auto &r = result[s];
        r.measured.resize(6817);
        const auto *corrections = static_cast<const C *>(batch.phase_corrections.mapped);
        const C phase = corrections ? corrections[s] : C{1, 0};
        for (unsigned k = 0; k < 6817; ++k) {
            const auto bin = batch.carrier_stride == 8192 ? (k + 8192 - 3408) % 8192 : k;
            r.measured[k] = y[s * batch.carrier_stride + bin] * phase;
        }
        r.equalized.resize(6817);
        r.sample = double(windows[s].sample);
        r.sample_step = 1;
        r.phase = windows[s].phase;
        r.frequency_hz = windows[s].frequency_hz;
        r.modulo4 = int(std::bit_cast<std::uint32_t>(stats[s * batch.pilot_stride].real()));
        r.noise_variance = stats[s * batch.pilot_stride + 1].real();
        const unsigned start = 3 * r.modulo4;
        auto channel = [&](unsigned k) {
            return std::complex<double>(r.measured[k]) *
                   std::polar(1.0, tau * (int(k) - 3408) * fft_window_margin / 8192) /
                   double(pattern.values[k]);
        };
        for (unsigned k = 0; k < 6817; ++k) {
            unsigned left = start + (k > start ? (k - start) / 12 * 12 : 0),
                     right = std::min(left + 12, 6816U);
            if (left == 6816) {
                left -= 12;
                right = 6816;
            }
            const auto a = channel(left), b = channel(right);
            const double mu = double(int(k) - int(left)) / (right - left);
            const auto h =
                (a + (b - a) * mu) * std::polar(1.0, -tau * (int(k) - 3408) * fft_window_margin / 8192);
            r.equalized[k] =
                C(std::complex<double>(r.measured[k]) * std::conj(h) / std::max(std::norm(h), 1e-30));
        }
    }
    return result;
}
std::vector<Symbol> receiver_symbols(const ReceiverBatch &batch) {
    if (batch.symbols.empty())
        return {};
    std::vector<OfdmWindow> windows;
    windows.reserve(batch.symbols.size());
    for (const auto &s : batch.symbols)
        windows.push_back(s.window);
    auto result = ofdm_symbols(batch.carriers, windows);
    for (unsigned i = 0; i < result.size(); ++i)
        result[i].discontinuity = batch.symbols[i].discontinuity;
    return result;
}
} // namespace radar::validation
