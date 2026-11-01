#include "matched_detector.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <numeric>

namespace radar {
void DetectionOptions::validate() const {
    if (!std::isfinite(integration_seconds) || integration_seconds < .5 || integration_seconds > 10 ||
        !std::isfinite(update_seconds) || update_seconds <= 0 || !std::isfinite(focus.range_metres) ||
        focus.range_metres < 1000 || focus.range_metres > matched_max_range_metres ||
        !std::isfinite(focus.doppler_hz) || focus.doppler_hz < 4 / integration_seconds ||
        focus.doppler_hz > 200)
        throw std::invalid_argument("Detection requires CPI 0.5..10 s, range 1..100 km, Doppler limit "
                                    "4/CPI..200 Hz and a positive update interval");
}
struct MatchedDetector::Impl {
    struct Row {
        std::vector<C> values;
        double time, information;
    };
    struct Completed {
        DopplerMap map;
        ComputeTimings focus;
    };
    struct Channel {
        bool publish_maps = true;
        GpuNoise noise;
        GpuFocus focus;
        std::deque<Row> history;
        std::uint64_t segment = UINT64_MAX, sequence{};
        double last_publish = -1e30;
        // Declared last: its destructor joins before any referenced GPU owner
        // is destroyed, including during exception unwinding.
        std::future<Completed> pending;
        explicit Channel(std::shared_ptr<gpu::Context> context) : noise(context), focus(std::move(context)) {}
    };
    static void collect(Channel &channel, const Consumer &consumer, DetectionTimings *timing, bool wait) {
        if (!channel.pending.valid() ||
            (!wait && channel.pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready))
            return;
        auto completed = channel.pending.get();
        if (timing) {
            timing->focus.gpu += completed.focus.gpu;
            timing->focus.batches += completed.focus.batches;
            timing->focus.gpu_timestamps_available |= completed.focus.gpu_timestamps_available;
            for (unsigned i = 0; i < 5; ++i)
                timing->focus.stages[i] += completed.focus.stages[i];
        }
        consumer(completed.map.channel, std::move(completed.map));
    }
    DetectionOptions options;
    unsigned profile_size;
    GpuMatchedProfiles profiles;
    MatchedProfiles profile_buffer;
    Projector projector;
    ProfileConsumer profile_consumer;
    std::vector<std::unique_ptr<Channel>> channels;
    Impl(std::shared_ptr<gpu::Context> context, std::size_t count, DetectionOptions opts, Projector project,
         ProfileConsumer consume)
        : options(opts), profile_size(matched_size_for_range(opts.focus.range_metres)),
          profiles(context, profile_size), projector(std::move(project)),
          profile_consumer(std::move(consume)) {
        for (std::size_t i = 0; i < count; ++i) {
            channels.push_back(std::make_unique<Channel>(context));
            channels.back()->publish_maps = opts.publish_maps;
        }
    }
};
MatchedDetector::MatchedDetector(std::shared_ptr<gpu::Context> context, std::size_t channels,
                                 DetectionOptions options, Projector projector, ProfileConsumer consume) {
    options.validate();
    if (!channels || channels > 64)
        throw std::invalid_argument("Detection requires 1..64 channels");
    impl_ = std::make_unique<Impl>(std::move(context), channels, options, std::move(projector),
                                   std::move(consume));
}
MatchedDetector::~MatchedDetector() = default;
void MatchedDetector::set_publish_maps(std::size_t channel, bool enabled) {
    impl_->channels.at(channel)->publish_maps = enabled;
}
void MatchedDetector::append(const SarBatch &batch, const Consumer &consume, DetectionTimings *timing) {
    auto &p = *impl_;
    const auto &frame = batch.spectra.frame;
    auto &c = *p.channels.at(frame.channel);
    for (auto &channel : p.channels)
        Impl::collect(*channel, consume, timing, false);
    const unsigned guard = batch.capture.channels.at(frame.channel).guard;
    const double dt = (8192.0 + 8192.0 / guard) / 8e6;
    const auto required = std::size_t(std::ceil(p.options.integration_seconds / dt));
    if (c.segment != frame.segment) {
        Impl::collect(c, consume, timing, true);
        c.history.clear();
        c.noise.reset();
        c.segment = frame.segment;
        c.last_publish = -1e30;
        if (p.options.publish_maps)
            c.focus.prepare(required, p.profile_size);
    }
    if (frame.last <= frame.first + 2)
        return;
    const auto variance = c.noise.estimate(batch.spectra, timing ? &timing->noise : nullptr);
    auto &profiles = p.profile_buffer;
    p.profiles.calculate_into(profiles, batch.spectra, variance, guard, timing ? &timing->profiles : nullptr);
    if (p.profile_consumer)
        p.profile_consumer(batch, profiles, {static_cast<const float *>(variance.mapped), 6817});
    if (!p.options.publish_maps)
        return;
    const auto *valid = static_cast<const std::uint32_t *>(batch.spectra.transmitted.valid.mapped);
    for (std::size_t n = 0; n < profiles.count; ++n) {
        const auto row = profiles.first + n;
        const double time = batch.seconds(row);
        const auto flags = std::span{valid + (row - 1) * 6817, 2 * std::size_t(6817)};
        if (!c.history.empty() && std::abs(time - c.history.back().time - dt) > dt * .01)
            c.history.clear();
        const bool known = std::ranges::all_of(flags, [](auto v) { return v != 0; });
        // A decoder erasure is an unavailable observation at a known time,
        // not a clock discontinuity. Keep its place in the aperture with zero
        // information; only a real time/segment discontinuity clears history.
        auto &stored = c.history.emplace_back();
        stored.time = time;
        stored.information = known ? profiles.information : 0;
        stored.values.resize(2 * p.profile_size);
        if (!known)
            std::ranges::fill(stored.values, C{});
        else
            for (unsigned moment = 0; moment < 2; ++moment)
                std::copy_n(profiles.values.data() + (moment * profiles.count + n) * p.profile_size,
                            p.profile_size, stored.values.data() + moment * p.profile_size);
        if (c.history.size() > required)
            c.history.pop_front();
        if (!c.publish_maps || c.history.size() != required ||
            time - c.last_publish + dt * 1e-8 < p.options.update_seconds)
            continue;
        // Keep every scientific profile, but do not stall live capture for an
        // optional display refresh. A later map still uses the full fixed CPI.
        if (p.options.best_effort_maps && c.pending.valid() &&
            c.pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            continue;
        std::vector<C> input(2 * required * p.profile_size);
        std::vector<double> information(required);
        for (std::size_t r = 0; r < required; ++r)
            information[r] = c.history[r].information;
        if (std::ranges::none_of(information, [](double value) { return value > 0; }))
            continue;
        for (std::size_t r = 0; r < required; ++r)
            for (unsigned moment = 0; moment < 2; ++moment)
                std::copy_n(c.history[r].values.data() + moment * p.profile_size, p.profile_size,
                            input.data() + (moment * required + r) * p.profile_size);
        // One outstanding job per channel bounds both memory and scheduling.
        // The job owns all numerical inputs; it cannot retain receiver buffers,
        // consumers, timing pointers or the shared clock's mutable history.
        Impl::collect(c, consume, timing, true);
        DopplerMap metadata;
        metadata.sequence = ++c.sequence;
        metadata.segment = c.segment;
        metadata.requested_integration = p.options.integration_seconds;
        metadata.channel = frame.channel;
        metadata.reference_origin_samples = batch.spectra.estimate.reference_origin;
        c.pending = std::async(
            std::launch::async,
            [focus = &c.focus, input = std::move(input), information = std::move(information), required, dt,
             rf = batch.capture.channels.at(frame.channel).rf_hz, first = c.history.front().time,
             options = p.options.focus, project = p.projector,
             queries = timing && timing->focus.timestamp_queries, metadata = std::move(metadata)]() mutable {
                Impl::Completed completed;
                completed.focus.timestamp_queries = queries;
                auto result = focus->calculate_device(input, required, dt, rf, first, options,
                                                      &completed.focus, information);
                auto &map = completed.map;
                map = std::move(result.map);
                map.sequence = metadata.sequence;
                map.segment = metadata.segment;
                map.channel = metadata.channel;
                map.requested_integration = metadata.requested_integration;
                map.reference_origin_samples = metadata.reference_origin_samples;
                const auto values =
                    std::span{static_cast<const C *>(result.values.mapped), map.ranges * map.frequencies};
                if (project)
                    project(map, result.values);
                else
                    map.values.assign(values.begin(), values.end());
                return completed;
            });
        c.last_publish = time;
    }
}
void MatchedDetector::finish(const Consumer &consume, DetectionTimings *timing) {
    for (auto &channel : impl_->channels)
        Impl::collect(*channel, consume, timing, true);
}
void MatchedDetector::reset() {
    for (auto &channel : impl_->channels) {
        auto &c = *channel;
        Impl::collect(c, [](std::size_t, DopplerMap &&) {}, nullptr, true);
        c.history.clear();
        c.noise.reset();
        c.segment = UINT64_MAX;
        c.sequence = 0;
        c.last_publish = -1e30;
    }
}
} // namespace radar
