#include "modem.hpp"
#include "pilot_phase.hpp"
#include "receiver_clock.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

using namespace radar;
namespace {
constexpr double tau = 2 * std::numbers::pi;
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
// Previous sequential unwrap, retained only to demonstrate the regression.
PilotPhase sequential(std::span<const C> now, std::span<const C> anchor,
                      std::span<const std::uint32_t> indices) {
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, previous = 0;
    for (std::size_t j = 0; j < indices.size(); ++j) {
        const auto z = std::complex<double>(now[j]) * std::conj(std::complex<double>(anchor[j]));
        const double x = double(indices[j]) - 3408, w = std::abs(z);
        previous += std::remainder(std::arg(z) - previous, tau);
        sw += w;
        sx += w * x;
        sy += w * previous;
        sxx += w * x * x;
        sxy += w * x * previous;
    }
    const double slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
    return {(sy - slope * sx) / sw, slope};
}
} // namespace
int main() {
    try {
        Modem modem;
        const auto indices = modem.pilot_pattern().continual;
        std::mt19937 random(91573);
        std::normal_distribution<float> noise(0, .025F);
        std::vector<C> anchor(indices.size()), now(indices.size());
        double old_square = 0, new_square = 0, worst = 0, seed = 0;
        unsigned slips = 0;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned n = 0; n < 3000; ++n) {
            const double delay = .008 * n, phase = std::remainder(.731 + .271 * n, tau);
            for (std::size_t j = 0; j < indices.size(); ++j) {
                const double k = double(indices[j]) - 3408;
                const auto h = 1.0 + std::polar(.95, -tau * k * 17.5 / 8192) +
                               std::polar(.65, -tau * k * 71.25 / 8192 + .3);
                anchor[j] = static_cast<C>(h);
                now[j] = static_cast<C>(h * std::polar(1.0, phase + tau * k * delay / 8192)) +
                         C{noise(random), noise(random)};
            }
            const auto before = sequential(now, anchor, indices),
                       after = pilot_phase(now, anchor, indices, seed);
            seed = after.slope;
            const double old_error = before.slope * 8192 / tau - delay,
                         new_error = after.slope * 8192 / tau - delay;
            slips += std::abs(old_error) > .5;
            old_square += old_error * old_error;
            new_square += new_error * new_error;
            worst = std::max(worst, std::abs(new_error));
            if (n == 1700) {
                auto reverse_indices = indices;
                auto reverse_now = now, reverse_anchor = anchor;
                std::ranges::reverse(reverse_indices);
                std::ranges::reverse(reverse_now);
                std::ranges::reverse(reverse_anchor);
                const auto reversed = pilot_phase(reverse_now, reverse_anchor, reverse_indices, seed);
                check(std::abs(reversed.slope - after.slope) < 1e-9, "Pilot fit depends on traversal order");
            }
        }
        std::cout << "Fading pilot regression: old timing RMS=" << std::sqrt(old_square / 3000)
                  << " samples, branch slips=" << slips << "; circular RMS=" << std::sqrt(new_square / 3000)
                  << ", maximum=" << worst << "; combined comparison CPU="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " s\n";
        check(worst < .01, "Faded pilots corrupted the timing branch");
        check(slips > 0 && old_square > 100 * new_square,
              "Regression did not exercise sequential unwrap failure");
        for (double phase : {-3.14, -.7, .731, 3.14})
            for (double delay : {-.4, 0., .4, 32.25}) {
                for (std::size_t j = 0; j < indices.size(); ++j) {
                    anchor[j] = {.8F, .3F};
                    now[j] = anchor[j] * static_cast<C>(std::polar(
                                             1.0, phase + tau * (int(indices[j]) - 3408) * delay / 8192));
                }
                const auto fit = pilot_phase(now, anchor, indices, tau * delay / 8192);
                check(std::abs(std::remainder(fit.phase - phase, tau)) < 1e-7 &&
                          std::abs(fit.slope * 8192 / tau - delay) < 1e-7,
                      "Pilot fit lost analytic phase/timing coordinates");
            }
        std::cout << "Pilot timing branch, traversal invariance and analytic phase: PASS\n";
        std::vector<double> precision(indices.size());
        std::vector<C> white_now(indices.size()), white_anchor(indices.size());
        double unweighted_square = 0, weighted_square = 0;
        for (unsigned trial = 0; trial < 2000; ++trial) {
            for (std::size_t j = 0; j < indices.size(); ++j) {
                const double sigma = j % 5 == 0 ? 10 : 1;
                anchor[j] = {.8F, .3F};
                now[j] =
                    anchor[j] *
                        static_cast<C>(std::polar(1., .731 + tau * (int(indices[j]) - 3408) * .173 / 8192)) +
                    float(sigma) * C{noise(random), noise(random)};
                precision[j] = 1 / (sigma * sigma);
                white_now[j] = now[j] / float(sigma);
                white_anchor[j] = anchor[j] / float(sigma);
            }
            const auto fitted = pilot_phase(now, anchor, indices, 0, precision),
                       whitened = pilot_phase(white_now, white_anchor, indices);
            check(std::abs(fitted.slope - whitened.slope) * 8192 / tau < 1e-7,
                  "Pilot precision differs from explicit whitening");
            check(std::abs(fitted.delay_variance / whitened.delay_variance - 1) < 2e-7 &&
                      std::abs(fitted.phase_variance / whitened.phase_variance - 1) < 2e-7,
                  "Pilot Fisher covariance differs from explicit whitening");
            unweighted_square += std::pow(pilot_phase(now, anchor, indices).slope * 8192 / tau - .173, 2);
            weighted_square += std::pow(fitted.slope * 8192 / tau - .173, 2);
        }
        check(weighted_square < unweighted_square / 10,
              "Known coloured pilot noise was not downweighted by its precision");
        std::cout << "Coloured pilot noise: timing MSE improvement=" << unweighted_square / weighted_square
                  << " times\n";
        // A whole-band phase change belongs to the carrier phase, never to the
        // clock slope. Use asymmetric faded pilots and changing precision so this
        // tests profiling of the intercept, not symmetry of the pilot locations.
        constexpr double adc_rate = 56e6, channel_rate = 8e6;
        constexpr unsigned decimation = 7, length = 8704;
        const std::array centres{-21e6, 137123.25, 21e6};
        SampleClock baseline(adc_rate, centres), rotated(adc_rate, centres);
        std::array<double, 3> previous{}, previous_rotated{}, seeded{}, rotated_seed{};
        std::vector<C> changed(indices.size());
        double maximum_delay = 0, maximum_phase = 0, maximum_clock = 0, maximum_covariance = 0;
        for (unsigned symbol = 1; symbol <= 2000; ++symbol) {
            const double n = double(symbol) * length, t = n / channel_rate;
            for (unsigned channel = 0; channel < centres.size(); ++channel) {
                const double delay = .03 * channel + .01 * t + .002 * channel * t * t;
                const double phase = .37 * channel + .09 * t;
                const double injected =
                    (channel + 1) * std::sin(tau * (.3 + .17 * channel) * t) + (.2 - .3 * channel) * t * t;
                const auto rotation = static_cast<C>(std::polar(1., injected));
                for (std::size_t j = 0; j < indices.size(); ++j) {
                    const double k = double(indices[j]) - 3408;
                    anchor[j] = static_cast<C>(1. + std::polar(.8, -tau * k * 23.25 / 8192 + .13 * channel));
                    now[j] = anchor[j] * static_cast<C>(std::polar(1., phase + tau * k * delay / 8192)) +
                             C{noise(random), noise(random)};
                    changed[j] = now[j] * rotation;
                    precision[j] = (j % 7 == 0 ? .02 : 1.) * (indices[j] > 3408 ? .3 : 2.);
                }
                const auto a = pilot_phase(now, anchor, indices, seeded[channel], precision);
                const auto b = pilot_phase(changed, anchor, indices, rotated_seed[channel], precision);
                seeded[channel] = a.slope;
                rotated_seed[channel] = b.slope;
                const double da = a.slope * 8192 / tau, db = b.slope * 8192 / tau;
                maximum_delay = std::max(maximum_delay, std::abs(da - db));
                maximum_phase =
                    std::max(maximum_phase, std::abs(std::remainder(b.phase - a.phase - injected, tau)));
                maximum_covariance =
                    std::max(maximum_covariance, std::abs(a.delay_variance / b.delay_variance - 1));
                const std::array oa{
                    TimingObservation{TimingObservation::Kind::rate, channel, n,
                                      (da - previous[channel]) / length},
                    TimingObservation{TimingObservation::Kind::offset, channel, n, da, a.delay_variance}};
                const std::array ob{
                    TimingObservation{TimingObservation::Kind::rate, channel, n,
                                      (db - previous_rotated[channel]) / length},
                    TimingObservation{TimingObservation::Kind::offset, channel, n, db, b.delay_variance}};
                observe_timing(baseline, oa, 0, decimation);
                observe_timing(rotated, ob, 0, decimation);
                previous[channel] = da;
                previous_rotated[channel] = db;
            }
            if (symbol % 20 == 0) {
                const auto at = std::uint64_t(n * decimation) + 1'000'000;
                baseline.update(at);
                rotated.update(at);
                maximum_clock =
                    std::max(maximum_clock, std::abs(baseline.at(double(at)).offset_at(double(at)) -
                                                     rotated.at(double(at)).offset_at(double(at))));
            }
        }
        const double clock_phase =
            maximum_clock * (centres.back() - centres.front() + 6656250) / adc_rate * 360;
        std::cout << "Independent time-varying band phases: maximum pilot timing change=" << maximum_delay
                  << " channel samples, recovered phase error=" << maximum_phase * 180 / std::numbers::pi
                  << " degrees, covariance change=" << maximum_covariance
                  << ", common-clock change=" << maximum_clock << " ADC samples (" << clock_phase
                  << " degrees across the occupied span)\n";
        // Rotating FP32 samples introduces quantisation, so require sub-0.05-degree
        // controller invariance rather than bitwise identity of the two estimates.
        check(maximum_delay < 5e-7 && maximum_phase < 2e-6 && maximum_covariance < 1e-6 && clock_phase < .05,
              "Carrier phase variation leaked into the common timing controller");
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
