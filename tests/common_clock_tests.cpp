#include "clock_interpolator.hpp"
#include "frontend_reference.hpp"
#include "gpu_frontend.hpp"
#include <algorithm>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double fs = 56e6, tau = 2 * std::numbers::pi;
void require(bool test, const char *message) {
    if (!test)
        throw std::runtime_error(message);
}
} // namespace
int main(int argc, char **) {
    try {
        const std::array offsets{-21e6, 137123.25, 21e6};
        // Verify the dynamically sized raw interpolator against an analytic
        // fractional delay over its entire declared passband and phase table.
        for (double band : {.1420201, .4344844, .4785157}) {
            ClockInterpolator filter(band);
            double maximum = 0;
            for (unsigned p = 0; p < 65; ++p) {
                const double mu = (p + .371) / 66, phase = mu * filter.phases;
                const auto index = unsigned(phase);
                const double alpha = phase - index;
                for (unsigned k = 0; k <= 256; ++k) {
                    const double f = band * k / 256;
                    std::complex<double> response{};
                    for (unsigned j = 0; j < filter.taps; ++j) {
                        const double h = (1 - alpha) * filter.coefficients[index * filter.taps + j] +
                                         alpha * filter.coefficients[(index + 1) * filter.taps + j];
                        response += h * std::polar(1.0, tau * f * (double(j) - filter.left - mu));
                    }
                    maximum = std::max(maximum, std::abs(response - 1.0));
                }
            }
            std::cout << "Raw interpolator: band=" << band << ", taps=" << filter.taps
                      << ", complex error=" << maximum << '\n';
            require(maximum < 5e-6, "Raw clock interpolator does not preserve its declared passband");
        }
        auto clock = std::make_shared<SampleClock>(fs, offsets, 3);
        require(std::abs(clock->timing_budget_samples() - 3.0 / 360 * fs / 48656250) < 1e-12,
                "Incorrect full-span phase budget");
        // Invalid precision must stay visible in diagnostics without entering
        // the estimator or publishing a correction from unusable evidence.
        SampleClock unusable(fs, offsets, 3);
        unusable.enable_diagnostics();
        for (double variance : {0., -1., std::numeric_limits<double>::infinity()})
            unusable.observe_offset(0, fs * .01, 123, variance);
        unusable.update(std::uint64_t(fs * .1));
        const auto ignored = unusable.take_observations();
        require(ignored.size() == 3 && std::ranges::all_of(ignored,
                                                           [](const auto &o) {
                                                               return o.kind ==
                                                                          ClockObservation::Kind::offset &&
                                                                      !o.innovation.gaussian_inlier &&
                                                                      o.innovation.robust_weight == 0;
                                                           }),
                "Unusable timing precision was hidden or given nonzero weight");
        require(unusable.segments().size() == 1 && std::isnan(unusable.estimate(fs * .1).offset_variance),
                "Unusable timing precision initialized or changed the clock");
        clock->schedule(1024, 1.00003);
        clock->schedule(13237, .999985);
        const auto segments = clock->segments();
        require(clock->segment_count() == segments.size(), "Clock count disagrees with its trajectory");
        for (auto from : {0ULL, 1023ULL, 1024ULL, 12000ULL, 13237ULL, 99999ULL}) {
            const auto snapshot = clock->segments(from);
            require(!snapshot.empty() && snapshot.front().start == clock->at(double(from)).start &&
                        snapshot.back().start == segments.back().start,
                    "Clock snapshot lost the covering segment or a subsequent rate change");
        }
        SampleClock long_capture(fs, offsets, 3);
        for (std::uint64_t i = 1; i <= 10000; ++i)
            long_capture.schedule(i * 1024, i % 2 ? 1.00001 : .99999);
        require(long_capture.segment_count() == 10001 && long_capture.segments(9999 * 1024).size() == 2,
                "A late batch copied the complete long-capture trajectory");
        for (std::size_t i = 1; i < segments.size(); ++i)
            require(std::abs(segments[i].offset - segments[i - 1].offset_at(double(segments[i].start))) <
                        1e-12,
                    "Clock update jumps sample position");
        // Independent receivers have different fixed propagation delays, but
        // observe one ADC oscillator. Neither delay may become a clock rate.
        SampleClock ensemble(fs, offsets, 3);
        std::mt19937 random(199);
        std::normal_distribution<double> noise(0, .0005);
        constexpr double ppm = 4e-6, drift = .15e-6;
        auto actual_offset = [](double n) { return ppm * n + .5 * drift * n * n / fs; };
        double worst = 0;
        for (unsigned i = 1; i <= 10000; ++i) {
            const double n = fs * i * .001;
            const double actual = actual_offset(n);
            for (std::size_t c = 0; c < offsets.size(); ++c) {
                if ((c == 2 && i < 5000) || (c == 0 && i > 7000))
                    continue;
                const auto s = ensemble.at(n);
                const auto previous = n - fs * .001;
                const double applied =
                    1 + (s.offset_at(n) - ensemble.at(previous).offset_at(previous)) / (fs * .001);
                const double true_rate = 1 + (actual - actual_offset(previous)) / (fs * .001);
                ensemble.observe_rate(c, n, applied / true_rate - 1);
                if (i >= 200)
                    ensemble.observe_offset(c, n,
                                            (s.offset_at(n) - actual + 17 * c) / s.rate + noise(random));
            }
            if (i % 20 == 0)
                ensemble.update(std::uint64_t(n + fs * .03));
            if (i > 2000) {
                // Absolute epoch is arbitrary; compare a 1-second interval.
                const double old = n - fs;
                const double error = (ensemble.at(n).offset_at(n) - ensemble.at(old).offset_at(old)) -
                                     (actual - actual_offset(old));
                worst = std::max(worst, std::abs(error));
            }
        }
        std::cout << "Common clock: maximum 1 s timing error=" << worst
                  << " ADC samples; budget=" << clock->timing_budget_samples() << '\n';
        require(worst < clock->timing_budget_samples(),
                "Ensemble clock fails 3 degree relative-phase budget");
        // A transmitter's fixed frequency bias must not turn into a changing
        // ADC phase origin when the precision of its pilot measurements changes.
        // The reference is the arithmetic mean of these three clock standards.
        SampleClock standards(fs, offsets, 3);
        constexpr std::array biases{-.04e-6, 0.0, .04e-6};
        double standard_error = 0, standard_square = 0;
        unsigned standard_count = 0;
        for (unsigned i = 1; i <= 10000; ++i) {
            const double n = fs * i * .001, previous = n - fs * .001;
            for (std::size_t c = 0; c < offsets.size(); ++c) {
                const auto s = standards.at(n);
                const double actual = actual_offset(n) + biases[c] * n;
                const double old = actual_offset(previous) + biases[c] * previous;
                const double applied =
                    1 + (s.offset_at(n) - standards.at(previous).offset_at(previous)) / (fs * .001);
                standards.observe_rate(c, n, applied / (1 + (actual - old) / (fs * .001)) - 1);
                const double sigma = ((i / 700 + c) % 3 == 0 ? .03 : .002);
                if (i >= 200)
                    standards.observe_offset(
                        c, n, (s.offset_at(n) - actual + 17 * c) / s.rate + noise(random) * (sigma / .0005),
                        sigma * sigma);
            }
            if (i % 20 == 0)
                standards.update(std::uint64_t(n + fs * .03));
            if (i > 2000) {
                const double old = n - fs;
                const double error = (standards.at(n).offset_at(n) - standards.at(old).offset_at(old)) -
                                     (actual_offset(n) - actual_offset(old));
                standard_error = std::max(standard_error, std::abs(error));
                standard_square += error * error;
                ++standard_count;
            }
        }
        const double standard_rms = std::sqrt(standard_square / standard_count);
        std::cout << "Changing pilot precision, biased transmitter clocks: 1 s ensemble RMS=" << standard_rms
                  << ", maximum=" << standard_error << " ADC samples\n";
        require(standard_rms < clock->timing_budget_samples() &&
                    standard_error < 3 * clock->timing_budget_samples(),
                "Pilot precision changes the ensemble phase reference");
        // Zero-sum channel timing disturbances must cancel in the arithmetic
        // ensemble. Their different curvature changes each channel's proposed
        // extrapolation horizon. Averaging the resulting rates would secretly
        // weight channel phases by 1/horizon and turn fixed TX frequency biases
        // into growing common-clock excursions.
        SampleClock horizons(fs, offsets, 3);
        double horizon_square = 0, horizon_max = 0;
        unsigned horizon_count = 0;
        for (unsigned i = 1; i <= 80000; ++i) {
            const double n = fs * i * .001, previous = n - fs * .001, t = i * .001;
            const std::array perturbation{3 * std::sin(tau * .7 * t), 3 * std::cos(tau * .7 * t),
                                          -3 * (std::sin(tau * .7 * t) + std::cos(tau * .7 * t))};
            for (unsigned c = 0; c < offsets.size(); ++c) {
                const auto s = horizons.at(n);
                const double actual = actual_offset(n) + biases[c] * n;
                const double old = actual_offset(previous) + biases[c] * previous;
                const double applied =
                    1 + (s.offset_at(n) - horizons.at(previous).offset_at(previous)) / (fs * .001);
                horizons.observe_rate(c, n, applied / (1 + (actual - old) / (fs * .001)) - 1);
                if (i >= 200)
                    horizons.observe_offset(c, n,
                                            (s.offset_at(n) - actual + 17 * c + perturbation[c]) / s.rate);
            }
            if (i % 20 == 0)
                horizons.update(std::uint64_t(n + fs * .03));
            if (i > 5000) {
                const double old = n - fs;
                const double error = (horizons.at(n).offset_at(n) - horizons.at(old).offset_at(old)) -
                                     (actual_offset(n) - actual_offset(old));
                horizon_square += error * error;
                horizon_max = std::max(horizon_max, std::abs(error));
                ++horizon_count;
            }
        }
        std::cout << "Unequal-horizon, fixed-frequency standards: 1 s RMS="
                  << std::sqrt(horizon_square / horizon_count) << ", maximum=" << horizon_max
                  << " ADC samples\n";
        require(std::sqrt(horizon_square / horizon_count) < .001 && horizon_max < .003,
                "Channel-dependent horizons redefine the arithmetic clock reference");
        // A missing/reacquired band must not change the ensemble definition.
        // Inject an isolated 50-sigma pilot fault and a new anchor on return.
        SampleClock robust(fs, offsets, 3);
        robust.enable_diagnostics();
        double robust_max = 0;
        unsigned downweighted = 0, late = 0;
        for (unsigned i = 1; i <= 12000; ++i) {
            const double n = fs * i * .001, previous = n - fs * .001;
            if (i == 7000)
                robust.reset_observer(1);
            for (unsigned c = 0; c < 3; ++c) {
                if (c == 1 && i >= 5000 && i < 7000)
                    continue;
                const auto s = robust.at(n);
                const double actual = actual_offset(n) + biases[c] * n;
                const double old = actual_offset(previous) + biases[c] * previous;
                const double applied =
                    1 + (s.offset_at(n) - robust.at(previous).offset_at(previous)) / (fs * .001);
                robust.observe_rate(c, n, applied / (1 + (actual - old) / (fs * .001)) - 1);
                if (i >= 200) {
                    const double fault = c == 2 && i == 8500 ? .5 : 0;
                    const double anchor = c == 1 && i >= 7000 ? 30 : 0;
                    robust.observe_offset(
                        c, n, (s.offset_at(n) - actual + 17 * c + anchor + fault) / s.rate + noise(random),
                        .0001);
                }
            }
            if (i == 9000)
                robust.observe_offset(0, n - fs, 0, .0001); // Late batch must be explicit, never rewind.
            if (i % 20 == 0) {
                robust.update(std::uint64_t(n + fs * .03));
                for (const auto &o : robust.take_observations()) {
                    downweighted += o.kind == ClockObservation::Kind::offset && !o.innovation.gaussian_inlier;
                    late += o.late;
                }
            }
            if (i > 2500) {
                const double old = n - fs;
                robust_max =
                    std::max(robust_max, std::abs(robust.at(n).offset_at(n) - robust.at(old).offset_at(old) -
                                                  (actual_offset(n) - actual_offset(old))));
            }
        }
        std::cout << "Clock holdover/reacquisition: 1 s maximum=" << robust_max
                  << " samples, downweighted=" << downweighted << ", late=" << late << '\n';
        require(robust_max < clock->timing_budget_samples() && downweighted > 0 && late == 1,
                "Clock fault handling changed reference, lost coherence or hid stale timing");
        const auto stable_segments = robust.segments();
        for (std::size_t i = 1; i < stable_segments.size(); ++i)
            require(stable_segments[i].offset ==
                        stable_segments[i - 1].offset_at(double(stable_segments[i].start)),
                    "Estimator update/reacquisition introduced a time-map discontinuity");
        // Deliberately wrong process-noise power by 100x either way, changing
        // pilot precision, and a physical common timing random walk. This is a
        // bounded robustness case, not a hardware Allan-deviation calibration.
        for (double scale : {.1, 10.}) {
            ClockControlSettings settings;
            settings.receiver.white_fm *= scale;
            SampleClock mismatch(fs, offsets, 3, settings);
            std::vector<double> truth(10001);
            double wander = 0, square = 0;
            unsigned count = 0;
            for (unsigned i = 1; i < truth.size(); ++i) {
                const double n = fs * i * .001, previous = n - fs * .001;
                wander += noise(random) / .0005 * (fs * 1e-10 * std::sqrt(.001));
                truth[i] = actual_offset(n) + wander;
                for (unsigned c = 0; c < 3; ++c) {
                    const auto s = mismatch.at(n);
                    const double actual = truth[i] + biases[c] * n, old = truth[i - 1] + biases[c] * previous;
                    const double applied =
                        1 + (s.offset_at(n) - mismatch.at(previous).offset_at(previous)) / (fs * .001);
                    mismatch.observe_rate(c, n, applied / (1 + (actual - old) / (fs * .001)) - 1);
                    const double sigma = (i / 900 + c) % 3 ? .0025 : .02;
                    if (i >= 200)
                        mismatch.observe_offset(
                            c, n, (s.offset_at(n) - actual + 17 * c) / s.rate + noise(random) * sigma / .0005,
                            sigma * sigma);
                }
                if (i % 20 == 0)
                    mismatch.update(std::uint64_t(n + fs * .03));
                if (i > 3000) {
                    const double error = mismatch.at(n).offset_at(n) - mismatch.at(n - fs).offset_at(n - fs) -
                                         (truth[i] - truth[i - 1000]);
                    square += error * error;
                    ++count;
                }
            }
            const double rms = std::sqrt(square / count);
            std::cout << "Clock noise-amplitude mismatch=" << scale << ": 1 s RMS=" << rms
                      << " ADC samples\n";
            require(rms < clock->timing_budget_samples(),
                    "Clock process-noise mismatch exceeded the coherence budget in its declared stress case");
        }
        if (argc == 1)
            return 0;

        GpuChannelBank bank(56000000, offsets);
        bank.set_clock(clock);
        constexpr std::uint64_t first = 7000, input_start = 6000;
        constexpr std::size_t count = 2048;
        std::vector<C> input(18000);
        auto unwarp = [&](double u) {
            auto s = segments.front();
            for (auto next : segments)
                if (u >= double(next.start) + next.offset)
                    s = next;
            return double(s.start) + (u - double(s.start) - s.offset) / s.rate;
        };
        for (std::size_t j = 0; j < input.size(); ++j) {
            const double n = unwarp(double(input_start + j));
            std::complex<double> value{};
            for (std::size_t c = 0; c < offsets.size(); ++c)
                value += std::polar(.2, tau * (offsets[c] + 437500) * n / fs + .31 * c);
            input[j] = static_cast<C>(value);
        }
        const auto output = bank.process_clocked(input, input_start, first, count);
        double phase_error = 0, amplitude_error = 0;
        for (std::size_t c = 0; c < offsets.size(); ++c)
            for (std::size_t m = 0; m < count; ++m) {
                const auto expected = std::polar(.2, tau * 437500 * double(first + 7 * m) / fs + .31 * c);
                const auto ratio = std::complex<double>(output[c][m]) / expected;
                phase_error = std::max(phase_error, std::abs(std::arg(ratio)));
                amplitude_error = std::max(amplitude_error, std::abs(std::abs(ratio) - 1));
            }
        // A separate scalar interpolation/convolution reference checks random
        // samples, not just tones that could conceal matching phase mistakes.
        for (auto &z : input)
            z = {float(noise(random) * 100), float(noise(random) * 100)};
        const auto random_output = bank.process_clocked(input, input_start, first, count);
        ClockInterpolator reference((21e6 + 3328125) / fs);
        const auto taps = GpuChannelBank::prototype(fs);
        std::vector<C> corrected(63 * 7 + taps.size());
        for (std::size_t j = 0; j < corrected.size(); ++j) {
            const auto n = first - taps.size() / 2 + j;
            corrected[j] = reference.interpolate(input, double(n - input_start) +
                                                            clock->at(double(n)).offset_at(double(n)));
        }
        double scalar_error = 0;
        for (std::size_t c = 0; c < offsets.size(); ++c) {
            const auto expected = channel_reference(corrected, taps, offsets[c], fs, first, 7, 64);
            for (std::size_t m = 0; m < 64; ++m)
                scalar_error = std::max(scalar_error, double(std::abs(expected[m] - random_output[c][m])));
        }
        // Repeat with a different chunk boundary. Historical segments and
        // overlapping FIR supports must yield identical common-grid samples.
        const auto overlapping = bank.process_clocked(input, input_start, first + 7 * 777, count - 777);
        double boundary_error = 0;
        for (std::size_t c = 0; c < offsets.size(); ++c)
            for (std::size_t m = 0; m < count - 777; ++m)
                boundary_error =
                    std::max(boundary_error, double(std::abs(overlapping[c][m] - random_output[c][m + 777])));
        std::cout << "Three-channel raw clock warp: phase=" << phase_error * 180 / std::numbers::pi
                  << " degrees, gain=" << amplitude_error << ", scalar reference=" << scalar_error
                  << ", chunk boundary=" << boundary_error << '\n';
        require(phase_error < 2e-4 && amplitude_error < 2e-4,
                "Shared raw-IQ retiming loses channel phase or gain");
        require(scalar_error < 2e-6 && boundary_error < 2e-6,
                "Common-clock GPU disagrees with scalar reference or chunk coordinates");
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
