#include "gpu_frontend.hpp"
#include "channel_filters.hpp"
#include "gpu_compute.hpp"
#include "spectral_clock.hpp"
#include <algorithm>
#include <chrono>

namespace radar {
namespace {
std::uint32_t grid_rate(std::uint32_t rate) {
    return (rate + 7'999'999) / 8'000'000 * 8'000'000;
}
} // namespace
struct GpuChannelBank::Impl {
    WaveReader *input = nullptr;
    std::uint32_t input_rate, source_rate;
    std::vector<double> offsets;
    std::vector<float> prototype;
    std::shared_ptr<SampleClock> clock;
    std::shared_ptr<gpu::Context> context = std::make_shared<gpu::Context>();
    std::unique_ptr<SpectralClock> retiming;
    bool download = true;
    std::size_t valid_count = 0;
    Impl(std::uint32_t source, std::span<const double> centres)
        : input_rate(grid_rate(source)), source_rate(source), offsets(centres.begin(), centres.end()),
          prototype(GpuChannelBank::prototype(input_rate)),
          retiming(std::make_unique<SpectralClock>(input_rate, offsets, prototype, context, source_rate)) {}
};
std::vector<float> GpuChannelBank::prototype(double rate) {
    if (!std::isfinite(rate) || rate < 8e6 || rate > 128e6 || std::fmod(rate, 8e6) != 0)
        throw std::runtime_error("Prototype requires an integer 8 MS/s input multiple, up to 128 MS/s");
    // Less common high input rates use the shared conservative Kaiser designer.
    if (rate > 64e6)
        return kaiser_prototype(rate, 3.5e6, 1024);
    const auto taps = filters::prototype(static_cast<unsigned>(rate / 8e6));
    return {taps.begin(), taps.end()};
}
GpuChannelBank::GpuChannelBank(std::uint32_t input_rate, std::span<const double> offsets) {
    const double fs = input_rate;
    if (fs < 8e6 || fs > 128e6 || offsets.empty() || offsets.size() > 64)
        throw std::runtime_error("Invalid GPU channel bank: use 1..64 channels and 8..128 MS/s input");
    for (double f : offsets)
        if (!std::isfinite(f) || std::abs(f) + 3.5e6 > fs / 2)
            throw std::runtime_error("Channel extends beyond input Nyquist interval");
    impl_ = std::make_unique<Impl>(input_rate, offsets);
    auto &p = *impl_;
    decimation = p.input_rate / 8'000'000;
}
GpuChannelBank::GpuChannelBank(WaveReader &wave, std::span<const double> offsets, double start,
                               double seconds)
    : GpuChannelBank(wave.info.sample_rate, offsets) {
    if (!std::isfinite(start) || start < 0 || !std::isfinite(seconds) || seconds <= 0 ||
        wave.info.samples > (std::uint64_t{1} << 53))
        throw std::runtime_error("Invalid channel bank input interval");
    impl_->input = &wave;
    const auto half = taps() / 2;
    const double fs = coordinate_rate();
    const auto available = std::uint64_t(double(wave.info.samples) * fs / wave.info.sample_rate);
    if (start * wave.info.sample_rate >= static_cast<double>(wave.info.samples))
        throw std::runtime_error("Start is beyond recording");
    origin = std::max<std::uint64_t>(half, static_cast<std::uint64_t>(start * fs));
    origin = (origin + decimation - 1) / decimation * decimation;
    if (origin + half >= available)
        throw std::runtime_error("Recording shorter than FIR support");
    output_count = std::min<std::uint64_t>(
        (available - half - 1 - origin) / decimation + 1,
        static_cast<std::uint64_t>(std::min(seconds * 8e6, static_cast<double>(available))));
}
GpuChannelBank::~GpuChannelBank() = default;
std::size_t GpuChannelBank::taps() const {
    return impl_->prototype.size();
}
std::uint32_t GpuChannelBank::coordinate_rate() const {
    return impl_->input_rate;
}
std::size_t GpuChannelBank::capacity() const {
    return impl_->retiming->capacity();
}
unsigned GpuChannelBank::interpolation_taps() const {
    return impl_->retiming->interpolation_taps();
}
void GpuChannelBank::set_clock(std::shared_ptr<SampleClock> clock) {
    auto &p = *impl_;
    if (produced || p.valid_count || p.clock || !clock)
        throw std::runtime_error("Attach the common clock before processing");
    p.clock = std::move(clock);
    if (p.input) {
        const auto old = origin;
        origin = std::max<std::uint64_t>(origin, p.retiming->support() + p.retiming->raw_left() + 8);
        origin = (origin + decimation - 1) / decimation * decimation;
        output_count -= std::min(output_count, (origin - old + decimation - 1) / decimation);
    }
}
std::vector<std::vector<C>> GpuChannelBank::next(FrontendTimings *timings) {
    auto &p = *impl_;
    p.valid_count = 0;
    if (!p.input || !p.clock)
        throw std::runtime_error("Attach a WAV source and common sample clock before reading");
    auto count = static_cast<std::size_t>(std::min<std::uint64_t>(capacity(), output_count - produced));
    if (!count)
        return {};
    using Clock = std::chrono::steady_clock;
    auto mark = timings ? Clock::now() : Clock::time_point{};
    auto measure = [&](double FrontendTimings::*member) {
        if (timings) {
            const auto now = Clock::now();
            timings->*member += std::chrono::duration<double>(now - mark).count();
            mark = now;
        }
    };
    const auto centre = origin + produced * decimation;
    {
        // Previous output used the right FIR halo. Activate the next immutable
        // segment only beyond that halo, so overlapping windows see one warp.
        const auto support = p.retiming->support();
        const auto first = centre - support;
        p.clock->update(centre + ((support + 63) / 64) * 64);
        auto original = [&](std::uint64_t n) {
            const auto s = p.clock->at(double(n));
            const auto product = (n % p.input_rate) * std::uint64_t(p.source_rate);
            const auto integer = n / p.input_rate * p.source_rate + product / p.input_rate;
            return std::int64_t(integer) +
                   std::int64_t(std::floor(double(product % p.input_rate) / p.input_rate +
                                           s.offset_at(double(n)) * (double(p.source_rate) / p.input_rate)));
        };
        const auto begin = original(first) - p.retiming->raw_left();
        if (begin < 0)
            throw std::runtime_error("Clock warp starts before available ADC support");
        std::int64_t end = 0;
        const auto requested = count;
        while (count) {
            end = original(centre + (count - 1) * decimation + support) + p.retiming->raw_right();
            if (end <= std::int64_t(p.input->info.samples))
                break;
            const auto missing = std::size_t(end - p.input->info.samples);
            count -= std::min(count, (missing + decimation - 1) / decimation + 1);
        }
        if (count != requested)
            output_count = produced + count;
        if (!count) {
            produced = output_count;
            return {};
        }
        auto *raw = p.retiming->input_data(std::size_t(end - begin));
        p.input->read(begin, {raw, std::size_t(end - begin)});
        measure(&FrontendTimings::read);
        auto result = process_clocked({raw, std::size_t(end - begin)}, begin, centre, count, timings);
        produced += count;
        return result;
    }
}
ChannelBatch GpuChannelBank::next_device(FrontendTimings *timings) {
    auto &p = *impl_;
    const auto first = produced;
    struct Restore {
        bool &flag;
        ~Restore() {
            flag = true;
        }
    } restore{p.download};
    p.download = false;
    next(timings);
    if (!p.valid_count)
        return {};
    return {output_buffer(), first, p.valid_count, p.offsets.size(), batch_size};
}
gpu::BufferView GpuChannelBank::output_buffer() const {
    return impl_->retiming->output_buffer();
}
std::shared_ptr<gpu::Context> GpuChannelBank::context() const {
    return impl_->context;
}
ChannelBatch GpuChannelBank::process_clocked_device(std::span<const C> samples, std::uint64_t input_start,
                                                    std::uint64_t first_center, std::size_t count,
                                                    FrontendTimings *timings) {
    if (first_center < origin || (first_center - origin) % decimation)
        throw std::invalid_argument("Unaligned corrected channel origin");
    auto &p = *impl_;
    struct Restore {
        bool &flag;
        bool value;
        ~Restore() {
            flag = value;
        }
    } restore{p.download, p.download};
    p.download = false;
    process_clocked(samples, input_start, first_center, count, timings);
    return {output_buffer(), (first_center - origin) / decimation, count, p.offsets.size(), batch_size};
}
std::size_t GpuChannelBank::input_capacity() const {
    if (!impl_->clock)
        throw std::logic_error("Attach the common clock before requesting resident input capacity");
    return impl_->retiming->input_capacity();
}
GpuChannelBank::InputSupport GpuChannelBank::input_support(std::uint64_t first, std::size_t count) const {
    const auto &p = *impl_;
    if (!p.clock || !count || count > capacity() || first < p.retiming->support())
        throw std::invalid_argument("Invalid live channel interval");
    auto original = [&](std::uint64_t n) {
        const auto product = (n % p.input_rate) * std::uint64_t(p.source_rate);
        const auto integral = n / p.input_rate * p.source_rate + product / p.input_rate;
        return std::int64_t(integral) +
               std::int64_t(std::floor(double(product % p.input_rate) / p.input_rate +
                                       p.clock->at(double(n)).offset_at(double(n)) *
                                           (double(p.source_rate) / p.input_rate)));
    };
    return {original(first - p.retiming->support()) - p.retiming->raw_left(),
            original(first + (count - 1) * decimation + p.retiming->support()) + p.retiming->raw_right()};
}
void GpuChannelBank::update_clock(std::uint64_t first) {
    auto &p = *impl_;
    if (!p.clock)
        throw std::logic_error("No common clock attached");
    p.clock->update(first + ((p.retiming->support() + 63) / 64) * 64);
}
ChannelBatch GpuChannelBank::process_clocked_device(const gpu::BufferView &samples, std::size_t valid_samples,
                                                    std::uint64_t input_start, std::uint64_t first_center,
                                                    std::size_t count, FrontendTimings *timings) {
    auto &p = *impl_;
    if (!p.clock || !count || count > capacity() || first_center < origin ||
        (first_center - origin) % decimation || first_center < taps() / 2 ||
        first_center > (std::uint64_t{1} << 53) - (count - 1) * decimation)
        throw std::invalid_argument("Invalid resident common-clock channel coordinates");
    using Timer = std::chrono::steady_clock;
    const auto mark = timings ? Timer::now() : Timer::time_point{};
    p.retiming->process(samples, valid_samples, input_start, first_center, count,
                        p.clock->segments(first_center - p.retiming->support() - 8), timings);
    p.valid_count = count;
    if (timings)
        timings->dispatch_wait += std::chrono::duration<double>(Timer::now() - mark).count();
    return {output_buffer(), (first_center - origin) / decimation, count, p.offsets.size(), batch_size};
}
std::vector<std::vector<C>> GpuChannelBank::process_clocked(std::span<const C> samples,
                                                            std::uint64_t input_start,
                                                            std::uint64_t first_center, std::size_t count,
                                                            FrontendTimings *timings) {
    auto &p = *impl_;
    if (!p.clock || !count || count > capacity() || first_center < taps() / 2 ||
        first_center > (std::uint64_t{1} << 53) - (count - 1) * decimation)
        throw std::runtime_error("Invalid common-clock channel bank coordinates");
    using Timer = std::chrono::steady_clock;
    const auto mark = timings ? Timer::now() : Timer::time_point{};
    p.retiming->process(samples, input_start, first_center, count,
                        p.clock->segments(first_center - p.retiming->support() - 8), timings);
    p.valid_count = count;
    if (timings)
        timings->dispatch_wait += std::chrono::duration<double>(Timer::now() - mark).count();
    if (!p.download)
        return {};
    const auto *output = static_cast<const C *>(p.retiming->output_buffer().mapped);
    std::vector<std::vector<C>> result(p.offsets.size());
    for (std::size_t c = 0; c < result.size(); ++c)
        result[c].assign(output + c * batch_size, output + c * batch_size + count);
    return result;
}
} // namespace radar
