#pragma once
#include "modem.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>

namespace radar::validation {
// Decoder-only channel estimate, with the production 8-point frequency
// interpolator. Raw observations are immutable. Radius zero reproduces the
// production temporal interpolation. Positive radius uses local-linear LS
// averaging of pilots, phase-aligned using continual pilots only.
class PilotDenoiseReference {
    static constexpr unsigned bins = 6817, dense = 2273;
    PilotPattern pattern_;
    std::array<bool, bins> continual_{};
    std::array<C, bins> centre_, restore_;
    std::array<std::array<float, 8>, 24> weights_{};

  public:
    explicit PilotDenoiseReference(PilotPattern pattern, unsigned guard = 1) : pattern_(std::move(pattern)) {
        if (guard > 3)
            throw std::invalid_argument("Invalid guard");
        for (auto k : pattern_.continual)
            continual_[k] = true;
        for (unsigned k = 0; k < bins; ++k) {
            const double angle = 2 * std::numbers::pi * (int(k) - 3408) *
                                 (fft_window_margin + .5 * (8192 / (32U >> guard))) / 8192;
            restore_[k] = static_cast<C>(std::polar(1., -angle));
            centre_[k] = std::conj(restore_[k]) / pattern_.values[k];
        }
        for (unsigned p = 0; p < 24; ++p)
            for (unsigned j = 0; j < 8; ++j) {
                double weight = 1;
                for (unsigned l = 0; l < 8; ++l)
                    if (l != j)
                        weight *= (double(p) / 3 - l) / (int(j) - int(l));
                weights_[p][j] = float(weight);
            }
    }
    unsigned select_radius(std::span<const C> raw, unsigned rows, std::span<const double> offsets,
                           unsigned maximum = 32) const {
        if (rows <= 64)
            return 8;
        if (raw.size() != rows * bins || offsets.size() != rows)
            throw std::invalid_argument("Invalid pilot prediction data");
        std::vector<std::complex<double>> aligned(raw.size());
        for (unsigned s = 0; s < rows; ++s)
            for (unsigned k = 0; k < bins; ++k)
                aligned[s * bins + k] =
                    std::complex<double>(raw[s * bins + k]) *
                    std::polar(1., -2 * std::numbers::pi * (int(k) - 3408) * offsets[s] / 8192);
        std::complex<double> phase{1, 0};
        for (unsigned s = 0; s < rows; ++s) {
            std::complex<double> cross{};
            for (auto k : pattern_.continual)
                cross += aligned[s * bins + k] * std::conj(aligned[(s ? s - 1 : 0) * bins + k]);
            // The previous row is already aligned, so this is cumulative CPE.
            phase = std::polar(1., std::arg(cross));
            for (unsigned k = 0; k < bins; ++k)
                aligned[s * bins + k] *= std::conj(phase);
        }
        const unsigned margin = maximum >= 64 && rows > 128 ? 64 : 32;
        std::array<double, 5> risk{};
        for (unsigned j = 0; j < dense; ++j)
            if (!continual_[3 * j])
                for (unsigned s = margin + j % 4; s + margin < rows; s += 4) {
                    std::complex<double> sum{};
                    for (unsigned d = 1; d <= margin / 4; ++d) {
                        sum += aligned[(s - 4 * d) * bins + 3 * j] + aligned[(s + 4 * d) * bins + 3 * j];
                        if (std::has_single_bit(d))
                            risk[std::bit_width(d) - 1] +=
                                std::norm(aligned[s * bins + 3 * j] - sum / double(2 * d));
                    }
                }
        return 4U << (std::min_element(risk.begin(), risk.begin() + (margin == 64 ? 5 : 4)) - risk.begin());
    }
    std::vector<C> estimate(std::span<const C> raw, unsigned rows, unsigned radius,
                            std::span<const double> source_offsets,
                            std::span<const float> frequency_filter = {}) const {
        if (raw.size() != rows * bins || source_offsets.size() != rows || rows < 8)
            throw std::invalid_argument("Invalid pilot denoising input");
        if (!frequency_filter.empty() &&
            (frequency_filter.size() % 2 != 1 || frequency_filter.size() > 129 ||
             !std::ranges::all_of(frequency_filter, [](float x) { return std::isfinite(x); })))
            throw std::invalid_argument("Invalid pilot frequency filter");
        std::vector<C> phases(rows, C{1, 0}), aligned(raw.size()), grid(rows * dense), result(raw.size());
        // Known sample-coordinate offsets are removed before measuring CPE.
        for (unsigned s = 0; s < rows; ++s)
            for (unsigned k = 0; k < bins; ++k) {
                const auto angle = -2 * std::numbers::pi * (int(k) - 3408) * source_offsets[s] / 8192;
                aligned[s * bins + k] = raw[s * bins + k] * static_cast<C>(std::polar(1., angle));
            }
        if (radius)
            for (unsigned s = 0; s < rows; ++s) {
                std::complex<double> cross{};
                // Adjacent-symbol reference: a distant reference can become nearly
                // orthogonal under rotating multipath and produce false phase jumps.
                const auto previous = s ? s - 1 : 0;
                for (auto k : pattern_.continual)
                    cross += std::complex<double>(aligned[s * bins + k]) *
                             std::conj(std::complex<double>(aligned[previous * bins + k]));
                phases[s] = static_cast<C>(std::polar(1., std::arg(cross)));
                for (unsigned k = 0; k < bins; ++k)
                    aligned[s * bins + k] *= std::conj(phases[s]);
            }
        for (unsigned s = 0; s < rows; ++s)
            for (unsigned j = 0; j < dense; ++j) {
                const auto k = 3 * j;
                C value{};
                if (!radius) {
                    int a = int(s) - int((s % 4 + 4 - j % 4) % 4), b = a + 4;
                    if (continual_[k])
                        a = b = int(s);
                    if (a < 0)
                        a = b;
                    if (b >= int(rows))
                        b = a;
                    const float mu = a == b ? 0.F : float(int(s) - a) / (b - a);
                    value = (1 - mu) * aligned[a * bins + k] + mu * aligned[b * bins + k];
                } else {
                    const int begin = std::max(0, int(s) - int(radius)),
                              end = std::min(int(rows) - 1, int(s) + int(radius));
                    double n = 0, t = 0, tt = 0;
                    std::complex<double> y{}, ty{};
                    for (int r = begin; r <= end; ++r)
                        if (continual_[k] || unsigned(r) % 4 == j % 4) {
                            const double dt = r - int(s);
                            const std::complex<double> z = aligned[r * bins + k];
                            ++n;
                            t += dt;
                            tt += dt * dt;
                            y += z;
                            ty += dt * z;
                        }
                    const auto determinant = n * tt - t * t;
                    value = static_cast<C>(determinant > 0 ? (tt * y - t * ty) / determinant : y / n);
                }
                grid[s * dense + j] = value * centre_[k];
            }
        if (!frequency_filter.empty()) {
            const auto original = grid;
            const auto half = frequency_filter.size() / 2;
            // At band edges retain the unsmoothed estimate: extrapolating or
            // zero-padding unobserved carriers would introduce channel bias.
            for (unsigned s = 0; s < rows; ++s)
                for (unsigned j = half; j + half < dense; ++j) {
                    std::complex<double> sum{};
                    for (unsigned k = 0; k < frequency_filter.size(); ++k)
                        sum += double(frequency_filter[k]) *
                               std::complex<double>(original[s * dense + j + k - half]);
                    grid[s * dense + j] = static_cast<C>(sum);
                }
        }
        for (unsigned s = 0; s < rows; ++s)
            for (unsigned k = 0; k < bins; ++k) {
                const auto first = std::clamp(int(k / 3) - 3, 0, 2265);
                C h{};
                for (unsigned j = 0; j < 8; ++j)
                    h += weights_[k - 3 * first][j] * grid[s * dense + first + j];
                const auto angle = 2 * std::numbers::pi * (int(k) - 3408) * source_offsets[s] / 8192;
                result[s * bins + k] = h * restore_[k] * phases[s] * static_cast<C>(std::polar(1., angle));
            }
        return result;
    }
};
} // namespace radar::validation
