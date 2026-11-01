#include "reference/waveform.hpp"
// Physical echo phase through independent DVB-T receivers, FEC, reconstruction,
// response fitting and finite range compression. Input is channelised IQ on the
// common time grid; channelisation/ADC-clock tests are separate. The oracle uses
// the transmitted time waveform, never the reconstructed reference or fitted H.
#include "dsp.hpp"
#include "fractional_resampler.hpp"
#include "gpu_compute.hpp"
#include "gpu_pipeline.hpp"
#include "matched_profiles.hpp"
#include <algorithm>
#include <array>
#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double tau = 2 * std::numbers::pi, fs = 8e6, light = 299792458.;
constexpr unsigned nfft = 8192, bins = 6817, length = 8704, capacity = 131071;
unsigned symbols = 1360;
constexpr std::uint64_t epoch = (1ULL << 40) + 37;
constexpr std::array rf{177.5e6, 184.5e6, 191.625e6, 219.5e6, 226.5e6};
constexpr std::array cfo{35585.25, -23137.125, 8512.125, 35037.25, -937.5};
constexpr std::array direct_delay{73U, 221U, 127U, 43U, 193U};
constexpr std::array delays{391.375, 703.25}, speeds{45., -27.};
constexpr std::array amplitudes{.03, .02}, phases{.43, -.91};
constexpr std::array rate_errors{.25e-6, -.19e-6, .12e-6, -.31e-6, .17e-6};
// A common 100 ns, 3 Hz time warp has 1.885 ppm peak rate excursion.
// Its channel-NCO phase is included, not independently aligned afterwards.
double warp_amplitude_ns = 100, warp_frequency_hz = 3, warp_initial_phase = 0;
double warp(double n, bool enabled) {
    return enabled ? fs * warp_amplitude_ns * 1e-9 *
                         std::sin(tau * warp_frequency_hz * n / fs + warp_initial_phase)
                   : 0.;
}
double warp_phase(double n, unsigned channel, bool enabled) {
    return tau * (rf[channel] - 202e6 + cfo[channel]) * warp(n, enabled) / fs;
}
double carrier_phase(double n, unsigned channel, bool enabled) {
    const double t = n / fs;
    return enabled
               ? (channel + 1) * .8 * std::sin(tau * (.35 + .17 * channel) * t) + (.3 - .2 * channel) * t * t
               : 0.;
}
double echo_phase(double n, unsigned channel, unsigned echo, bool enabled) {
    const double t = n / fs;
    return enabled
               ? (1. + .2 * channel) * std::sin(tau * (.6 + .11 * channel + .13 * echo) * t + .31 * echo) +
                     (.25 * channel - .4) * (1 + echo) * t * t
               : 0.;
}
struct Source {
    std::vector<C> reference, echo, received;
};
struct Check {
    std::complex<double> correlation{};
    double energy{}, error{}, phase_square{}, maximum_phase{};
    std::uint64_t count{};
    std::vector<std::array<double, 2>> phase_trace;
    void add(C actual, C expected, double time) {
        correlation += std::complex<double>(actual) * std::conj(std::complex<double>(expected));
        energy += std::norm(std::complex<double>(expected));
        error += std::norm(std::complex<double>(actual) - std::complex<double>(expected));
        ++count;
        const auto phase = std::arg(std::complex<double>(actual) * std::conj(std::complex<double>(expected)));
        phase_square += std::norm(std::complex<double>(expected)) * phase * phase;
        maximum_phase = std::max(maximum_phase, std::abs(phase));
        phase_trace.push_back({time, phase * 180 / std::numbers::pi});
    }
    boost::json::object json(bool trace = false) const {
        const auto ratio = correlation / energy;
        boost::json::object result{
            {"comparisons", count},
            {"phase_error_degrees", std::arg(ratio) * 180 / std::numbers::pi},
            {"phase_rms_degrees", std::sqrt(phase_square / energy) * 180 / std::numbers::pi},
            {"maximum_phase_error_degrees", maximum_phase * 180 / std::numbers::pi},
            {"amplitude_ratio", std::abs(ratio)},
            {"relative_complex_rms", std::sqrt(error / energy)}};
        if (trace) {
            boost::json::array samples;
            for (const auto &point : phase_trace)
                samples.emplace_back(boost::json::array{point[0], point[1]});
            result["phase_error_time_s_degrees"] = std::move(samples);
        }
        return result;
    }
};
Source synthesize(unsigned channel, bool clocks, bool wander = false, bool histories = false,
                  bool path_wander = false) {
    validation::ReferenceModem modem;
    Tps mode;
    mode.mode = 1;
    mode.constellation = 2;
    mode.rate = 2;
    mode.guard = 1;
    std::mt19937 random(9581 + channel * 1973);
    const auto packets = (std::size_t(symbols) * 6048 * 6 * 3 / 4 + 8 * 204 - 1) / (8 * 204) + 12;
    std::vector<std::uint8_t> bytes(packets * 188);
    for (auto &b : bytes)
        b = static_cast<std::uint8_t>(random());
    for (std::size_t i = 0; i < packets; ++i)
        bytes[i * 188] = i % 8 ? 0x47 : 0xb8;
    const auto carriers = modem.synthesize(bytes, mode, symbols);
    Source out;
    out.reference.resize(symbols * length);
    out.echo.resize(out.reference.size());
    out.received.resize(out.reference.size());
    std::vector<C> base(symbols * length), work(nfft);
    Fft fft(nfft);
    for (unsigned s = 0; s < symbols; ++s) {
        std::fill(work.begin(), work.end(), C{});
        for (unsigned k = 0; k < bins; ++k)
            work[(k + nfft - 3408) % nfft] = carriers[s * bins + k];
        fft.transform(work, true);
        std::copy(work.end() - 512, work.end(), base.begin() + s * length);
        std::copy(work.begin(), work.end(), base.begin() + s * length + 512);
    }
    FractionalResampler interpolation;
    auto sample = [&](double at) {
        const auto i = std::int64_t(std::floor(at));
        if (i < 47 || i + 49 >= std::int64_t(base.size()))
            return C{};
        return interpolation.interpolate(base.data() + i - 47, at - i);
    };
    const double rate = 1 + (clocks ? rate_errors[channel] : 0.);
    for (std::size_t n = direct_delay[channel]; n < base.size(); ++n) {
        const double time = double(n) / fs;
        const double physical = double(n) + warp(double(n), wander);
        // A reference-only propagation change is not an ADC clock error. Keep
        // the target paths fixed, and include both envelope delay and RF phase.
        const double path_delay = warp(double(n), path_wander);
        out.reference[n] = (clocks ? sample((physical - direct_delay[channel] - path_delay) * rate)
                                   : base[n - direct_delay[channel]]) *
                           static_cast<C>(std::polar(1., -tau * rf[channel] * path_delay / fs));
        C echo{};
        for (unsigned e = 0; e < delays.size(); ++e) {
            const double delay = delays[e] + speeds[e] / light * physical;
            echo +=
                sample((physical - direct_delay[channel] - delay) * rate) *
                static_cast<C>(std::polar(amplitudes[e], phases[e] - tau * rf[channel] * delay / fs +
                                                             echo_phase(physical, channel, e, histories)));
        }
        out.echo[n] = echo;
        out.received[n] =
            (out.reference[n] + echo) *
            static_cast<C>(std::polar(1. - .1 * channel, .71 * channel + tau * cfo[channel] * time +
                                                             warp_phase(double(n), channel, wander) +
                                                             carrier_phase(double(n), channel, histories)));
    }
    return out;
}
} // namespace
int main(int argc, char **argv) {
    try {
        const bool wander = argc >= 2 && argc <= 5 && std::string_view(argv[1]) == "--retiming-wander";
        const bool path_wander =
            argc >= 2 && argc <= 5 && std::string_view(argv[1]) == "--reference-path-wander";
        const bool long_histories = argc == 2 && std::string_view(argv[1]) == "--phase-histories-long";
        const bool histories =
            long_histories || (argc == 2 && std::string_view(argv[1]) == "--phase-histories");
        if (long_histories)
            symbols = 4080;
        const bool clocks = wander || path_wander || histories ||
                            (argc == 2 && std::string_view(argv[1]) == "--source-clocks");
        if (argc > 1 && !clocks)
            throw std::invalid_argument(
                "Use no argument, --source-clocks, --phase-histories, "
                "--phase-histories-long, --retiming-wander, or --reference-path-wander");
        if ((wander || path_wander) && argc >= 3)
            warp_amplitude_ns = std::stod(argv[2]);
        if ((wander || path_wander) && argc >= 4)
            warp_frequency_hz = std::stod(argv[3]);
        if ((wander || path_wander) && argc >= 5)
            warp_initial_phase = std::stod(argv[4]);
        if (!std::isfinite(warp_amplitude_ns) || warp_amplitude_ns < 0 || warp_amplitude_ns > 1000 ||
            !std::isfinite(warp_frequency_hz) || warp_frequency_hz <= 0 || warp_frequency_hz > 100 ||
            !std::isfinite(warp_initial_phase))
            throw std::invalid_argument("Invalid common clock warp");
        std::array<Source, 5> sources;
        for (unsigned c = 0; c < sources.size(); ++c) {
            sources[c] = synthesize(c, clocks, wander, histories, path_wander);
            std::cerr << "Synthesized channel " << c << '\n';
        }
        auto context = std::make_shared<gpu::Context>();
        auto storage = gpu::make_buffer(context->gpu, sources.size() * capacity * sizeof(C));
        auto *mapped = static_cast<C *>(
            context->gpu.device->mapMemory(*storage.memory, 0, sources.size() * capacity * sizeof(C)));
        const gpu::BufferView input{context, *storage.handle, sources.size() * capacity * sizeof(C), mapped};
        auto noise_storage = gpu::make_buffer(context->gpu, bins * sizeof(float));
        auto *noise = static_cast<float *>(
            context->gpu.device->mapMemory(*noise_storage.memory, 0, bins * sizeof(float)));
        std::fill_n(noise, bins, 1.F);
        const gpu::BufferView variance{context, *noise_storage.handle, bins * sizeof(float), noise};
        GpuPipelineConfig config;
        config.guards.assign(sources.size(), 16);
        config.max_samples = capacity;
        config.origin = epoch;
        config.residual_spectra = false;
        GpuPipeline pipeline(config, context, {});
        GpuMatchedProfiles profiles(context);
        std::array<std::array<Check, 2>, 5> checks;
        std::array<Check, 5> carrier_checks;
        std::array<std::uint64_t, 5> packets{}, failed{}, rows{}, last_window{}, steered_windows{};
        std::vector<C> work(nfft);
        Fft fft(nfft);
        auto consume = [&](const SpectralBatch &batch) {
            const auto c = batch.frame.channel;
            const auto &source = sources[c];
            for (const auto &packet : batch.packets.packets) {
                ++packets[c];
                failed[c] += !packet.valid;
            }
            const auto compressed = profiles.calculate(batch, variance, 16);
            for (std::size_t row = 0; row < compressed.count; ++row) {
                const auto &w = batch.frame.symbols[compressed.first + row].window;
                const auto first = w.sample - epoch;
                if (rows[c] && w.sample - last_window[c] != length)
                    ++steered_windows[c];
                last_window[c] = w.sample;
                ++rows[c];
                if (rows[c] < 50 || rows[c] % 13)
                    continue;
                if (first < direct_delay[c] + 800 || first + nfft >= source.echo.size())
                    continue;
                if (histories || path_wander) {
                    // Recover the whole-band carrier phase from exported metadata,
                    // not from the echo oracle or a fitted output phase origin.
                    // The response's centre carrier has no delay phase ramp.
                    const double centre = double(first) + 4095.5;
                    const auto *h = static_cast<const C *>(batch.estimate.response.mapped);
                    const double restored = std::arg(std::complex<double>(h[3408])) - w.phase +
                                            tau * w.frequency_hz * 4095.5 / fs;
                    const double expected = .71 * c + tau * cfo[c] * centre / fs +
                                            carrier_phase(centre, c, histories) -
                                            tau * rf[c] * warp(centre, path_wander) / fs;
                    carrier_checks[c].add(static_cast<C>(std::polar(1., restored)),
                                          static_cast<C>(std::polar(1., expected)), centre / fs);
                }
                // Independent CPU finite-waveform oracle: retain only measured
                // occupied bins, then correlate with the original source waveform.
                // The tiny residual CFO uses the exported applied FFT frequency;
                // its centre phase is removed by the direct-path reference.
                for (unsigned j = 0; j < nfft; ++j)
                    work[j] = source.echo[first + j] *
                              static_cast<C>(
                                  std::polar(1., tau * (cfo[c] - w.frequency_hz) * (double(j) - 4095.5) / fs +
                                                     warp_phase(double(first + j), c, wander) -
                                                     warp_phase(double(first) + 4095.5, c, wander) +
                                                     carrier_phase(double(first + j), c, histories) -
                                                     carrier_phase(double(first) + 4095.5, c, histories)));
                fft.transform(work);
                for (unsigned k = 3409; k < nfft - 3408; ++k)
                    work[k] = {};
                fft.transform(work, true);
                for (unsigned e = 0; e < delays.size(); ++e) {
                    const auto lag =
                        std::int64_t(std::lround(delays[e] + speeds[e] / light * (double(first) + 4095.5)));
                    std::complex<double> expected{};
                    for (unsigned j = 0; j < nfft; ++j) {
                        const auto reference = std::complex<double>(source.reference[first + j - lag]);
                        expected += std::complex<double>(work[j]) * std::conj(reference);
                    }
                    // Archived profiles use unit-carrier reference energy. Actual
                    // QAM/pilot power remains in the pulse response, not in this
                    // fixed scale. No measured output gain is fitted here.
                    expected /= double(bins) / nfft;
                    checks[c][e].add(compressed.values[row * matched_profile_size + lag],
                                     static_cast<C>(expected), (double(first) + 4095.5) / fs);
                }
            }
        };
        unsigned block = 0;
        for (std::size_t first = 0; first < sources[0].received.size();) {
            const auto count = std::min<std::size_t>(block % 3 == 0   ? 137
                                                     : block % 3 == 1 ? 65537
                                                                      : capacity,
                                                     sources[0].received.size() - first);
            for (unsigned c = 0; c < sources.size(); ++c)
                std::copy_n(sources[c].received.data() + first, count, mapped + c * capacity);
            pipeline.process({input, epoch + first, count, sources.size(), capacity}, consume);
            first += count;
            ++block;
        }
        pipeline.finish(consume);
        boost::json::array channels;
        bool passed = true;
        for (unsigned c = 0; c < sources.size(); ++c) {
            boost::json::array echoes;
            for (const auto &check : checks[c]) {
                echoes.push_back(check.json(histories || path_wander));
                const auto ratio = check.correlation / check.energy;
                passed &= check.count >= 30 && std::abs(std::arg(ratio)) * 180 / std::numbers::pi < .37 &&
                          std::sqrt(check.phase_square / check.energy) * 180 / std::numbers::pi < .37 &&
                          std::abs(std::abs(ratio) - 1) < .02;
            }
            passed &= packets[c] > 1000 && failed[c] == 0 && (!clocks || steered_windows[c] > 0);
            boost::json::object channel{{"channel", c},
                                        {"rf_hz", rf[c]},
                                        {"source_rate_error_ppm", clocks ? rate_errors[c] * 1e6 : 0.},
                                        {"steered_fft_windows", steered_windows[c]},
                                        {"packets", packets[c]},
                                        {"failed_packets", failed[c]},
                                        {"rows", rows[c]},
                                        {"echoes", std::move(echoes)}};
            if (histories || path_wander) {
                const auto &check = carrier_checks[c];
                channel["restored_carrier_phase"] = check.json(true);
                passed &= check.count >= 30 &&
                          std::sqrt(check.phase_square / check.energy) * 180 / std::numbers::pi < .37;
            }
            channels.emplace_back(std::move(channel));
        }
        std::cout << boost::json::serialize(boost::json::object{
                         {"passed", passed},
                         {"channels", std::move(channels)},
                         {"common_time_warp_amplitude_ns", wander ? warp_amplitude_ns : 0.},
                         {"common_time_warp_hz", wander ? warp_frequency_hz : 0.},
                         {"common_time_warp_initial_phase", wander ? warp_initial_phase : 0.},
                         {"reference_path_wander_amplitude_ns", path_wander ? warp_amplitude_ns : 0.},
                         {"reference_path_wander_hz", path_wander ? warp_frequency_hz : 0.},
                         {"independent_carrier_and_echo_phase_histories", histories},
                         {"scope", "Five independent 64-QAM signals on the common channelised grid; two "
                                   "physical moving echoes, including one beyond the guard interval; no "
                                   "phase/gain alignment fitted to the output."}})
                  << '\n';
        context->gpu.device->unmapMemory(*noise_storage.memory);
        context->gpu.device->unmapMemory(*storage.memory);
        return passed ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
