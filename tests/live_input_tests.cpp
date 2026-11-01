#include "live_input.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <numbers>
#include <thread>

using namespace radar;
namespace {
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
void timeline() {
    const auto epoch = (1ULL << 42) + 57;
    check(nominal_corrected_sample(epoch, 61'440'000, 64'000'000) == epoch / 24 * 25 + epoch % 24 * 25 / 24,
          "Long-capture source coordinate overflowed");
    CaptureCounters stats;
    CaptureTimeline time;
    const auto a = time.receive(1'000'000, 123, stats);
    const auto b = time.receive(1'000'123, 71, stats);
    check(a.first == 0 && b.first == 123 && b.segment == 0, "Partial receive lost continuity");
    time.error(true, true, stats);
    check(stats.unknown_gaps == 1, "Unquantified overflow was hidden");
    const auto c = time.receive(1'000'999, 41, stats);
    check(c.first == 999 && c.segment == 1 && stats.missing_samples == 805 && stats.unknown_gaps == 0,
          "Overflow accounting is not timestamp exact");
    time.discard(41, stats);
    const auto d = time.receive(1'001'040, 17, stats);
    check(d.segment == 2 && stats.host_discarded == 41 && stats.missing_samples == 805,
          "Host discard counted twice or failed to break coherence");
    time.error(false, true, stats);
    const auto e = time.receive(1'001'057, 9, stats);
    check(e.segment == 3 && stats.unknown_gaps == 1, "Sequence error with unknown loss was hidden");
    time.receive(1'001'125, 3, stats);
    check(stats.unknown_gaps == 1 && stats.missing_samples == 864,
          "An unrelated later gap erased an earlier unquantified error");
    bool rejected = false;
    try {
        time.receive(5, 1, stats);
    } catch (...) {
        rejected = true;
    }
    check(rejected, "Accepted a backwards hardware timestamp");
    CaptureTimeline missing;
    rejected = false;
    try {
        missing.receive({}, 20, stats);
    } catch (...) {
        rejected = true;
    }
    check(rejected, "Invented a coordinate for timestamp-free data");
    CaptureCounters host_stats;
    CaptureTimeline host;
    host.receive(0, 64, host_stats);
    host.receive(64, 64, host_stats, false);
    host.discard(64, host_stats);
    host.receive(128, 64, host_stats, false);
    host.discard(64, host_stats);
    check(host.receive(192, 64, host_stats).segment == 1 && host_stats.segments == 2,
          "One host outage became multiple accepted capture segments");
    for (auto tick : {INT64_MAX, INT64_MIN}) {
        CaptureTimeline extreme;
        bool rejected = false;
        try {
            extreme.receive(tick, tick == INT64_MAX ? 1 : 64, host_stats);
            extreme.receive(INT64_MAX - 64, 1, host_stats);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected, "Extreme hardware timestamps overflowed the sample timeline");
    }
}
void ring() {
    std::vector<Iq16> storage(1024);
    CaptureRing ring(storage);
    constexpr std::size_t total = 200'000;
    std::jthread producer([&] {
        for (std::size_t n = 0; n < total;) {
            auto target = ring.writable(37);
            if (target.empty()) {
                std::this_thread::yield();
                continue;
            }
            const auto count = std::min(target.size(), total - n);
            for (std::size_t i = 0; i < count; ++i)
                target[i] = {std::int16_t((n + i) % 32000), std::int16_t(-(int((n + i) % 32000)))};
            check(ring.commit(count, n, 0), "Capture descriptor unexpectedly rejected");
            n += count;
        }
        ring.close();
    });
    CaptureChunk chunk;
    std::size_t n = 0;
    while (ring.next(chunk)) {
        check(chunk.source_first == n, "Capture descriptors reordered");
        for (std::size_t i = 0; i < chunk.count; ++i) {
            const auto value = storage[(chunk.ring_first + i) % storage.size()];
            check(value.i == std::int16_t((n + i) % 32000) && value.q == -value.i,
                  "Producer overwrote in-flight ring samples");
        }
        n += chunk.count;
        ring.release(chunk.ring_first + chunk.count);
    }
    check(n == total && ring.head() == ring.tail(), "Ring lost samples at wraparound");
    check(ring.peak() > 0 && ring.peak() <= ring.capacity(),
          "Ring occupancy watermark is outside its bounds");
}
void overrun() {
    std::vector<Iq16> storage(1024);
    CaptureRing ring(storage);
    CaptureCounters counters;
    CaptureTimeline timeline;
    for (unsigned block = 0; block < 7; ++block) {
        auto target = ring.writable(256);
        const bool accepted = !target.empty();
        const auto position = timeline.receive(10000 + block * 256, 256, counters, accepted);
        if (accepted) {
            std::fill(target.begin(), target.end(), Iq16{std::int16_t(block), 0});
            check(ring.commit(256, position.first, position.segment), "Unexpected ring commit failure");
        } else
            timeline.discard(256, counters);
    }
    check(ring.peak() == 1024 && counters.host_discarded == 768 && counters.missing_samples == 0,
          "Full ring concealed or double-counted an overrun");
    CaptureChunk chunk;
    check(ring.next(chunk, std::chrono::milliseconds(1)) && chunk.segment == 0 && chunk.source_first == 0 &&
              chunk.count == 1024,
          "Adjacent receive packets did not coalesce without changing coordinates");
    for (unsigned n = 0; n < 1024; ++n)
        check(storage[(chunk.ring_first + n) % 1024].i == int(n / 256), "Discard overwrote unread samples");
    ring.release(chunk.ring_first + chunk.count);
    const auto resumed = timeline.receive(10000 + 7 * 256, 256, counters);
    check(ring.commit(256, resumed.first, resumed.segment), "Ring could not resume after draining");
    check(ring.next(chunk, std::chrono::milliseconds(1)) && chunk.source_first == 1792 && chunk.segment == 1,
          "Resumed stream bridged discarded samples or reset its absolute time");
    ring.release(chunk.ring_first + chunk.count);
    ring.close();
}
void descriptor_capacity() {
    std::vector<Iq16> storage(16384);
    CaptureRing ring(storage);
    for (unsigned n = 0; n < 8192; ++n) {
        check(!ring.writable(1).empty(), "DMA descriptors shortened the configured sample buffer");
        check(ring.commit(1, n, 0), "Coherent DMA descriptors exhausted before sample capacity");
    }
    check(ring.commit(1, 9000, 1), "Could not retain an explicit source discontinuity");
    check(ring.commit(1, 9001, 2), "Could not retain an acquisition segment boundary");
    CaptureChunk chunk;
    check(ring.next(chunk, std::chrono::milliseconds(1)) && chunk.count == 8192 && chunk.source_first == 0 &&
              chunk.segment == 0,
          "Coalescing changed coherent coordinates");
    check(ring.next(chunk, std::chrono::milliseconds(1)) && chunk.count == 1 && chunk.source_first == 9000 &&
              chunk.segment == 1,
          "Coalescing bridged a source gap");
    check(ring.next(chunk, std::chrono::milliseconds(1)) && chunk.count == 1 && chunk.source_first == 9001 &&
              chunk.segment == 2,
          "Coalescing bridged an acquisition segment");
    ring.close();
}
void gpu_test() {
    constexpr std::uint32_t source_rate = 61'440'000;
    constexpr std::size_t count = 16384;
    const std::array centres{-24.5e6, -17.5e6, -10.375e6, 17.5e6, 24.5e6};
    GpuChannelBank bank(source_rate, centres);
    auto clock = std::make_shared<SampleClock>(bank.coordinate_rate(), centres);
    clock->schedule(64, 1.000005);
    bank.set_clock(clock);
    const auto epoch = ((std::uint64_t{1} << 34) + 65536) / bank.decimation * bank.decimation;
    double maximum_phase_error = 0;
    for (auto first : std::array{epoch, epoch + std::uint64_t(119999 + count) * bank.decimation}) {
        const auto support = bank.input_support(first, count);
        const auto valid = std::size_t(support.last - support.first);
        GpuLiveInput live(bank.context(), std::bit_ceil(bank.input_capacity() * 2), bank.input_capacity());
        const auto ring_first = std::uint64_t(live.storage().size() - 791);
        std::vector<C> reference(valid);
        constexpr double tau = 2 * std::numbers::pi;
        std::array<double, 5> frequencies;
        for (unsigned c = 0; c < 5; ++c)
            frequencies[c] = centres[c] + (c + 1) * 31250.0;
        for (std::size_t n = 0; n < valid; ++n) {
            std::complex<double> value{};
            for (double f : frequencies)
                value += std::polar(.065,
                                    tau * std::remainder((double(support.first) + n) * f / source_rate, 1.0));
            const Iq16 iq{std::int16_t(std::round(value.real() * 32768)),
                          std::int16_t(std::round(value.imag() * 32768))};
            live.storage()[(ring_first + n) % live.storage().size()] = iq;
            reference[n] = {iq.i / 32768.F, iq.q / 32768.F};
        }
        const auto expected = bank.process_clocked(reference, support.first, first, count);
        const auto input = live.gather(ring_first, valid);
        const auto health = live.health();
        double expected_power = 0, expected_peak = 0;
        for (const auto z : reference) {
            expected_power += std::norm(z);
            expected_peak = std::max({expected_peak, double(std::abs(z.real())), double(std::abs(z.imag()))});
        }
        check(health.samples == valid && health.rail_samples == 0 && health.near_rail_samples == 0,
              "RF health counted padding or invented clipping");
        check(std::abs(health.power_sum - expected_power) < 1e-5 * std::max(1., expected_power) &&
                  health.peak_component == expected_peak,
              "GPU RF health differs from CPU reference across a circular-buffer wrap");
        const auto *unpacked = static_cast<const C *>(input.mapped);
        for (std::size_t n = 0; n < valid; ++n)
            check(unpacked[n] == reference[n], "sc16 GPU conversion is not exact");
        const std::array<Iq16, 5> rail_test{{{32752, 0}, {0, -32768}, {32113, 0}, {100, 100}, {-32768, 0}}};
        for (std::size_t n = 0; n < rail_test.size(); ++n)
            live.storage()[(ring_first + n) % live.storage().size()] = rail_test[n];
        live.gather(ring_first, rail_test.size(), nullptr, 1);
        const auto rails = live.health();
        check(rails.samples == 4 && rails.rail_samples == 2 && rails.near_rail_samples == 3,
              "RF rails/near-rails or halo exclusion are incorrect");
        for (std::size_t n = 0; n < rail_test.size(); ++n) {
            const auto z = reference[n];
            live.storage()[(ring_first + n) % live.storage().size()] = {std::int16_t(z.real() * 32768),
                                                                        std::int16_t(z.imag() * 32768)};
        }
        live.gather(ring_first, valid);
        for (std::size_t n = valid; n < bank.input_capacity(); ++n)
            check(unpacked[n] == C{}, "FFT padding retained stale input");
        const auto output = bank.process_clocked_device(input, valid, support.first, first, count);
        const auto *actual = static_cast<const C *>(output.samples.mapped);
        double phase_error = 0;
        for (unsigned c = 0; c < 5; ++c) {
            std::complex<double> correlation{};
            for (std::size_t n = 0; n < count; ++n) {
                check(actual[c * output.stride + n] == expected[c][n],
                      "Live and reference channelizers differ");
                const auto t = double(first + n * bank.decimation);
                const auto segment = clock->at(t);
                const double phase = t * (frequencies[c] - centres[c]) / bank.coordinate_rate() +
                                     frequencies[c] * segment.offset_at(t) / bank.coordinate_rate();
                correlation += std::complex<double>(actual[c * output.stride + n]) *
                               std::polar(1.0, -tau * std::remainder(phase, 1.0));
            }
            phase_error = std::max(phase_error, std::abs(std::arg(correlation)) * 180 / std::numbers::pi);
        }
        check(phase_error < .02, "Live multi-channel NCO/clock phase exceeds the analytic bound");
        maximum_phase_error = std::max(maximum_phase_error, phase_error);
    }
    std::cout << "Five-channel 61.44 MS/s clock, large epoch, wrapped ring and discontinuity: bit-exact; "
                 "phase error "
              << maximum_phase_error << " degrees\n";
}
} // namespace
int main(int argc, char **) {
    try {
        if (argc > 1)
            gpu_test();
        else {
            timeline();
            ring();
            overrun();
            descriptor_capacity();
        }
        std::cout << "Live capture input: PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
