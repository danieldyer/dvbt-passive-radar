#include "gpu_frontend.hpp"
#include "gpu_ofdm.hpp"
#include "reference/fec_decoder.hpp"
#include "reference/ofdm.hpp"
#include "reference/waveform.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>

using namespace radar;
void check(bool b, const char *message) {
    if (!b)
        throw std::runtime_error(message);
}
int main() {
    try {
        constexpr double tau = 2 * std::numbers::pi;
        constexpr unsigned count = 68, cp = 512, length = 8192 + cp;
        validation::ReferenceModem modem;
        const auto pattern = modem.pilot_pattern();
        Tps mode;
        mode.mode = 1;
        mode.constellation = 2;
        mode.rate = 2;
        mode.guard = 1;
        std::mt19937 rng(8471);
        std::vector<std::uint8_t> bytes(2000 * 188);
        for (auto &b : bytes)
            b = static_cast<std::uint8_t>(rng());
        for (unsigned p = 0; p < 2000; ++p)
            bytes[p * 188] = p % 8 ? 0x47 : 0xb8;
        const auto tx = modem.synthesize(bytes, mode, count);
        std::vector<C> input(count * length), expected(count * 6817), work(8192);
        std::vector<OfdmWindow> windows(count);
        Fft fft(8192);
        const std::uint64_t epoch = 1234567890123ULL;
        for (unsigned s = 0; s < count; ++s) {
            const C gain = std::polar(.7F + .015F * s, .13F * s);
            windows[s] = {s * length + cp - fft_window_margin, epoch + s * length + cp - fft_window_margin,
                          std::remainder(.57 * s + 1.3, tau),
                          s % 3 == 0   ? 173125.75
                          : s % 3 == 1 ? -113375.125
                                       : 85.25};
            std::fill(work.begin(), work.end(), C{});
            for (unsigned k = 0; k < 6817; ++k) {
                // Known smooth frequency response, with a nonzero complex gain
                // independently changing for each window (different channels).
                const auto h =
                    gain * (C{1, 0} + C{.11F, .035F} *
                                          static_cast<C>(std::polar(1.0, -tau * (int(k) - 3408) * 3 / 8192)));
                work[(k + 8192 - 3408) % 8192] = h * tx[s * 6817 + k];
                expected[s * 6817 + k] =
                    work[(k + 8192 - 3408) % 8192] *
                    static_cast<C>(std::polar(1.0, -tau * (int(k) - 3408) * fft_window_margin / 8192));
            }
            fft.transform(work, true);
            for (unsigned j = 0; j < length; ++j) {
                const auto at = (j + 8192 - cp) % 8192;
                const double angle = -windows[s].phase + tau * windows[s].frequency_hz *
                                                             (int(j) - int(cp) + fft_window_margin) / 8e6;
                input[s * length + j] = work[at] * static_cast<C>(std::polar(1.0, angle));
            }
        }
        GpuOfdm gpu(pattern, count);
        std::vector<Symbol> symbols;
        double raw_error = 0, eq_error = 0;
        ComputeTimings timing;
        timing.detailed = true;
        for (unsigned first = 0; first < count; first += 32) {
            const auto n = std::min(32U, count - first);
            const auto descriptors = std::span<const OfdmWindow>{windows}.subspan(first, n);
            const auto batch = gpu.transform(input, descriptors, &timing);
            auto out = validation::ofdm_symbols(batch, descriptors);
            for (unsigned s = 0; s < n; ++s) {
                check(out[s].sample == double(windows[first + s].sample),
                      "OFDM absolute sample coordinate lost precision");
                check(out[s].phase == windows[first + s].phase &&
                          out[s].frequency_hz == windows[first + s].frequency_hz,
                      "OFDM phase metadata changed");
                check(out[s].modulo4 == int((first + s) % 4), "GPU scattered-pilot symbol index mismatch");
                for (unsigned k = 0; k < 6817; ++k) {
                    raw_error = std::max(
                        raw_error, double(std::abs(out[s].measured[k] - expected[(first + s) * 6817 + k]) /
                                          std::abs(expected[(first + s) * 6817 + k])));
                    eq_error = std::max(eq_error,
                                        double(std::abs(out[s].equalized[k] - tx[(first + s) * 6817 + k])));
                }
                symbols.push_back(std::move(out[s]));
            }
        }
        std::cout << "Analytic raw spectrum max relative error=" << raw_error
                  << " equalizer max absolute error=" << eq_error << '\n';
        check(raw_error < 2e-5, "GPU NCO/window/FFT phase disagrees with analytic spectrum");
        check(eq_error < .0002, "GPU pilot equalization distorted a smooth channel");
        // Leave integer carrier offsets uncorrected for acquisition. Test both
        // signs; correcting only the CP estimate cannot resolve these shifts.
        for (int bins : {-97, 43}) {
            auto window = windows[0];
            window.frequency_hz -= bins * 8e6 / 8192;
            gpu.transform(input, std::span{&window, 1});
            const auto found = gpu.integer_frequency().front();
            check(found.bins == bins && found.quality > .99, "GPU integer-CFO acquisition failed");
        }
        // A post-FFT pilot phase correction must match the exported oscillator
        // phase. Independent equalization must cancel this common phase.
        const std::array correction{.713};
        const auto rotated = gpu.transform(input, std::span{windows}.first(1));
        const auto before = validation::ofdm_symbols(rotated, std::span{windows}.first(1));
        gpu.rotate(correction);
        auto phase_window = windows[0];
        phase_window.phase += correction[0];
        const auto after = validation::ofdm_symbols(rotated, std::span{&phase_window, 1});
        for (unsigned k = 0; k < 6817; ++k) {
            check(std::abs(after[0].measured[k] -
                           before[0].measured[k] * static_cast<C>(std::polar(1.0, correction[0]))) < 1e-6,
                  "GPU phase rotation mismatch");
            check(std::abs(after[0].equalized[k] - before[0].equalized[k]) < 2e-6,
                  "Common phase did not cancel in reference equalization");
        }
        Tps tps;
        check(modem.tps(symbols, tps), "GPU OFDM output failed TPS BCH");
        validation::FecDecoder fec;
        const auto reconstruction = modem.reconstruct(symbols, mode, nullptr, &fec);
        check(reconstruction.bad_packets == 0, "GPU OFDM/FEC roundtrip failed");
        const auto expected_fec = fec.decode(symbols, mode, modem);
        const auto expected_packets = std::vector(expected_fec.packets.begin(), expected_fec.packets.end());
        const auto resident_carriers = gpu.transform(input, windows);
        const auto packets =
            fec.decode(validation::ofdm_symbols(resident_carriers, windows), mode, modem).packets;
        check(packets.size() == expected_packets.size(), "Resident FEC packet count mismatch");
        for (unsigned p = 0; p < packets.size(); ++p)
            check(packets[p].valid == expected_packets[p].valid &&
                      packets[p].bytes == expected_packets[p].bytes,
                  "Resident OFDM readback disagrees with qualification input");
        const auto preserved = validation::ofdm_symbols(resident_carriers, windows);
        for (unsigned s = 0; s < count; ++s)
            check(preserved[s].measured == symbols[s].measured &&
                      preserved[s].equalized == symbols[s].equalized,
                  "FEC overwrote borrowed OFDM carriers");
        // Validate the CP sum and CFO sign directly with a scalar complex sum,
        // for all guard lengths and boundary candidates (not only the peak).
        for (unsigned guard : {4U, 8U, 16U, 32U}) {
            const unsigned prefix = 8192 / guard;
            std::vector<std::uint32_t> offsets(97);
            std::iota(offsets.begin(), offsets.end(), 0U);
            const auto scores = gpu.correlate(input, offsets, guard);
            for (unsigned p = 0; p < offsets.size(); ++p) {
                std::complex<double> correlation{};
                double energy = 0;
                for (unsigned j = 0; j < prefix; ++j) {
                    const auto a = std::complex<double>(input[p + j]),
                               b = std::complex<double>(input[p + 8192 + j]);
                    correlation += std::conj(a) * b;
                    energy += std::norm(a) + std::norm(b);
                }
                check(std::abs(std::complex<double>(scores[p].correlation) - correlation) < energy * 4e-7,
                      "GPU CP correlation mismatch");
                check(std::abs(scores[p].energy - energy) < energy * 4e-7, "GPU CP energy mismatch");
            }
        }
        std::array<std::uint32_t, 1> zero{0};
        const auto cp_score = gpu.correlate(input, zero, 16).front();
        const double frequency = std::arg(cp_score.correlation) * 8e6 / (tau * 8192);
        check(cp_score.quality > .99999 &&
                  std::abs(std::remainder(frequency - windows[0].frequency_hz, 8e6 / 8192)) < .001,
              "CP CFO sign or normalization mismatch");
        // Device input: consume the channel bank's output buffer directly. Each
        // selected band has its own local offset, all in one shared device context.
        const std::array centres{0.0};
        GpuChannelBank bank(8'000'000, centres);
        bank.set_clock(std::make_shared<SampleClock>(bank.coordinate_rate(), centres));
        constexpr unsigned first = cp - fft_window_margin;
        const auto n = std::min<std::size_t>(bank.capacity(), input.size() - 2 * first);
        auto centred = input;
        for (unsigned s = 0; s < count; ++s)
            for (unsigned j = 0; j < length; ++j) {
                const double angle = windows[s].phase - tau * windows[s].frequency_hz *
                                                            (int(j) - int(cp) + fft_window_margin) / 8e6;
                centred[s * length + j] *= static_cast<C>(std::polar(1.0, angle));
            }
        const auto support = bank.input_support(first, n);
        bank.process_clocked(std::span<const C>{centred}.first(support.last), 0, first, n);
        GpuOfdm chained(pattern, 4, bank.context());
        std::array<OfdmWindow, 4> local;
        for (unsigned s = 0; s < 4; ++s) {
            local[s] = windows[s];
            local[s].input_offset -= first;
            local[s].phase = 0;
            local[s].frequency_hz = 0;
        }
        const auto resident = chained.transform(bank.output_buffer(), local);
        const auto host = validation::ofdm_symbols(resident, local);
        for (unsigned s = 0; s < 4; ++s) {
            check(host[s].modulo4 == int(s), "Device-buffer OFDM pilot indexing failed");
            for (unsigned k = 0; k < 6817; ++k)
                check(std::abs(host[s].measured[k] - expected[s * 6817 + k]) < 2e-5,
                      "Channelizer/OFDM device handoff changed phase");
        }
        bool rejected = false;
        try {
            gpu.transform(bank.output_buffer(), local);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "OFDM accepted a buffer belonging to a different Vulkan device");
        std::cout << "Direct device-buffer handoff, CP, TPS, FEC: PASS\nGPU OFDM stages ms:";
        for (auto t : timing.stages)
            std::cout << ' ' << t * 1000;
        std::cout << "\nPASS\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
