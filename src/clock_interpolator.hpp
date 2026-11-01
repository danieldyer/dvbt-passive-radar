#pragma once
#include "dsp.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace radar {
// Clear scalar reference and shared coefficients for the raw-IQ GPU warp.
// Unlike the 8 MS/s symbol interpolator, its flat band covers ALL selected RF
// channels. Sinc interpolation is finite and approximate, not a periodic FFT.
class ClockInterpolator {
  public:
    static constexpr unsigned phases = 2048;
    unsigned taps, left;
    std::vector<float> coefficients;
    explicit ClockInterpolator(double highest_frequency_fraction) {
        if (!std::isfinite(highest_frequency_fraction) || highest_frequency_fraction < 0 ||
            highest_frequency_fraction >= .5)
            throw std::runtime_error("Invalid raw-IQ interpolation band");
        taps =
            std::max(32U, 16 * static_cast<unsigned>(std::ceil(4 / (.5 - highest_frequency_fraction) / 16)));
        left = taps / 2 - 1;
        coefficients.resize((phases + 1) * taps);
        auto i0 = [](double x) {
            double sum = 1, term = 1;
            for (unsigned k = 1; k < 80; ++k) {
                term *= x * x / (4 * k * k);
                sum += term;
                if (term < sum * 1e-16)
                    break;
            }
            return sum;
        };
        const double normalization = i0(12);
        for (unsigned p = 0; p <= phases; ++p) {
            double sum = 0;
            for (unsigned j = 0; j < taps; ++j) {
                const double x = double(j) - left - double(p) / phases;
                const double z = x / (taps / 2);
                const double h =
                    (std::abs(x) < 1e-14 ? 1 : std::sin(std::numbers::pi * x) / (std::numbers::pi * x)) *
                    i0(12 * std::sqrt(std::max(0.0, 1 - z * z))) / normalization;
                coefficients[p * taps + j] = float(h);
                sum += h;
            }
            for (unsigned j = 0; j < taps; ++j)
                coefficients[p * taps + j] /= float(sum);
        }
    }
    C interpolate(std::span<const C> input, double at) const {
        const auto integer = static_cast<std::int64_t>(std::floor(at));
        if (integer < left || std::uint64_t(integer - left + taps) > input.size())
            throw std::runtime_error("Clock interpolation support missing");
        const double phase = (at - double(integer)) * phases;
        const auto p = std::min(phases - 1, static_cast<unsigned>(phase));
        const double alpha = phase - p;
        std::complex<double> value{};
        for (unsigned j = 0; j < taps; ++j)
            value += std::complex<double>(input[integer - left + j]) *
                     ((1 - alpha) * coefficients[p * taps + j] + alpha * coefficients[(p + 1) * taps + j]);
        return static_cast<C>(value);
    }
};
} // namespace radar
