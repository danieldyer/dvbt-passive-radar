#pragma once
#include "signal_constants.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <span>
#include <vector>

namespace radar {
// Centred 96-tap Kaiser-windowed sinc, 2048 phases with linear phase-table interpolation.
// Input points to sample floor(t)-47; fractional coordinate mu is in [0,1).
// Timing latency is represented by the caller's absolute sample coordinate.
class FractionalResampler {
    static constexpr unsigned phases = 2048, taps = 96;
    std::vector<float> table_;
    static double i0(double x) {
        double term = 1, sum = 1;
        for (unsigned k = 1; k < 80; ++k) {
            term *= x * x / (4.0 * k * k);
            sum += term;
            if (term < sum * 1e-16)
                break;
        }
        return sum;
    }

  public:
    FractionalResampler() : table_((phases + 1) * taps) {
        for (unsigned p = 0; p <= phases; ++p) {
            const double mu = static_cast<double>(p) / phases;
            double sum = 0;
            for (unsigned j = 0; j < taps; ++j) {
                const double x = static_cast<double>(j) - 47 - mu, z = x / 48;
                const double h =
                    (std::abs(x) < 1e-14 ? 0.98
                                         : std::sin(0.98 * std::numbers::pi * x) / (std::numbers::pi * x)) *
                    i0(10 * std::sqrt(std::max(0.0, 1 - z * z))) / i0(10);
                table_[p * taps + j] = static_cast<float>(h);
                sum += h;
            }
            for (unsigned j = 0; j < taps; ++j)
                table_[p * taps + j] /= static_cast<float>(sum);
        }
    }
    std::complex<float> interpolate(const std::complex<float> *in, double mu) const {
        const double phase = mu * phases;
        const auto p = std::min(phases - 1, static_cast<unsigned>(phase));
        // Independent double-precision reference; four partial sums keep the
        // dot product clear and permit ordinary compiler vectorisation.
        std::array<std::complex<double>, 4> a{}, b{};
        for (unsigned j = 0; j < taps; j += a.size())
            for (unsigned k = 0; k < a.size(); ++k) {
                const auto value = std::complex<double>(in[j + k]);
                a[k] += value * double(table_[p * taps + j + k]);
                b[k] += value * double(table_[(p + 1) * taps + j + k]);
            }
        std::complex<double> first{}, second{};
        for (unsigned k = 0; k < a.size(); ++k) {
            first += a[k];
            second += b[k];
        }
        return static_cast<std::complex<float>>(first + (phase - p) * (second - first));
    }
    unsigned ntaps() const {
        return taps;
    }
    std::span<const float> coefficients() const {
        return table_;
    }
};
} // namespace radar
