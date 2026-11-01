#pragma once
#include "dsp.hpp"
#include <array>

namespace radar {
// Once the block's fractional shift is applied to its spectrum, oscillator
// drift leaves only a tiny residual shift. A centred differentiator and the
// second difference evaluate that residual without a fractional-delay table.
// |mu| <= .002; complex response error < 5e-6 through 3.328125 MHz at 16 MS/s.
struct SmallClockInterpolator {
    static constexpr double limit = .002;
    static constexpr std::array<double, 3> weights{.79893657459545331, -.19395204396696067,
                                                   .030257224953225224};
    static C interpolate(std::span<const C> input, std::size_t integer, double mu) {
        if (integer < 3 || integer + 3 >= input.size() || std::abs(mu) > limit)
            throw std::runtime_error("Small clock residual exceeds interpolation support or error bound");
        const auto centre = std::complex<double>(input[integer]);
        std::complex<double> derivative{};
        for (unsigned j = 1; j <= 3; ++j)
            derivative += weights[j - 1] * (std::complex<double>(input[integer + j]) -
                                            std::complex<double>(input[integer - j]));
        const auto curvature = std::complex<double>(input[integer + 1]) +
                               std::complex<double>(input[integer - 1]) - 2.0 * centre;
        return static_cast<C>(centre + mu * derivative + .5 * mu * mu * curvature);
    }
};
} // namespace radar
