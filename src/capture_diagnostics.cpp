#include "capture_diagnostics.hpp"
#include <boost/json.hpp>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <thread>

namespace radar {
namespace {
boost::json::value finite(double value) {
    return std::isfinite(value) ? boost::json::value(value) : boost::json::value(nullptr);
}
} // namespace
struct CaptureDiagnostics::Impl {
    static constexpr std::size_t limit = 8 * 1024 * 1024;
    std::ofstream file;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::string> records;
    std::size_t queued = 0;
    bool done = false;
    std::exception_ptr failure;
    std::jthread worker;
    explicit Impl(const std::filesystem::path &path) {
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file.open(path, std::ios::out | std::ios::noreplace);
        worker = std::jthread([this] {
            try {
                for (;;) {
                    std::string row;
                    {
                        std::unique_lock lock(mutex);
                        condition.wait(lock, [&] { return done || !records.empty(); });
                        if (records.empty())
                            break;
                        row = std::move(records.front());
                        records.pop_front();
                    }
                    file.write(row.data(), std::streamsize(row.size()));
                    {
                        std::lock_guard lock(mutex);
                        queued -= row.size();
                        condition.notify_all();
                    }
                }
                file.close();
            } catch (...) {
                std::lock_guard lock(mutex);
                failure = std::current_exception();
                done = true;
                condition.notify_all();
            }
        });
    }
    void send(boost::json::object value) {
        auto row = boost::json::serialize(value);
        row += '\n';
        if (row.size() > limit)
            throw std::runtime_error("Oversized capture diagnostic record");
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return failure || done || queued + row.size() <= limit; });
        if (failure)
            std::rethrow_exception(failure);
        if (done)
            throw std::runtime_error("Diagnostics already closed");
        queued += row.size();
        records.push_back(std::move(row));
        condition.notify_one();
    }
    void finish() {
        if (!worker.joinable())
            return;
        {
            std::lock_guard lock(mutex);
            done = true;
            condition.notify_all();
        }
        worker.join();
        if (failure)
            std::rethrow_exception(failure);
    }
    ~Impl() {
        try {
            finish();
        } catch (...) {
        }
    }
};
CaptureDiagnostics::CaptureDiagnostics(const std::filesystem::path &path, boost::json::object config)
    : impl_(std::make_unique<Impl>(path)) {
    config["type"] = "configuration";
    config["version"] = 1;
    impl_->send(std::move(config));
}
CaptureDiagnostics::~CaptureDiagnostics() = default;
std::size_t CaptureDiagnostics::queued_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->queued;
}
void CaptureDiagnostics::observations(std::span<const ClockObservation> observations) {
    if (observations.empty())
        return;
    boost::json::array rows;
    rows.reserve(observations.size());
    for (const auto &o : observations)
        rows.emplace_back(boost::json::array{
            int(o.kind), o.channel, o.sample, finite(o.value), finite(o.variance), finite(o.feedback_removed),
            o.applied_offset, o.applied_rate, finite(o.innovation.value), finite(o.innovation.variance),
            finite(o.innovation.normalized_square), o.innovation.gaussian_inlier, o.late,
            o.innovation.robust_weight});
    impl_->send(
        {{"type", "timing_observations"},
         {"columns",
          boost::json::array{"kind_0_rate_1_offset_2_reset", "channel", "corrected_adc_sample", "value",
                             "conditional_variance", "feedback_removed", "applied_offset", "applied_rate",
                             "innovation", "innovation_variance", "normalized_innovation_square",
                             "gaussian_inlier", "late", "robust_weight"}},
         {"rows", std::move(rows)}});
}
void CaptureDiagnostics::spectra(const SarBatch &batch) {
    const auto &f = batch.spectra.frame;
    boost::json::array rows;
    rows.reserve(f.last - f.first);
    const auto *valid = static_cast<const std::uint32_t *>(batch.spectra.transmitted.valid.mapped);
    for (auto r = f.first; r < f.last; ++r) {
        const auto &s = f.symbols[r];
        const bool known = std::ranges::all_of(std::span{valid + r * 6817, std::size_t(6817)},
                                               [](auto x) { return x != 0; });
        rows.emplace_back(boost::json::array{s.window.sample, batch.seconds(r), s.window.phase,
                                             s.window.frequency_hz, s.source_offset_samples,
                                             s.source_rate_error, known});
    }
    impl_->send(
        {{"type", "phase_history"},
         {"channel", f.channel},
         {"segment", f.segment},
         {"response_relative_error", finite(batch.spectra.estimate.relative_error)},
         {"columns", boost::json::array{"channel_sample", "seconds", "applied_phase", "applied_cfo_hz",
                                        "source_offset_samples", "source_rate_error", "fec_valid"}},
         {"rows", std::move(rows)}});
}
void CaptureDiagnostics::status(const CaptureStatus &s) {
    boost::json::array channels;
    for (const auto &c : s.channels) {
        const auto &r = c.receiver;
        const auto &o = c.clock;
        channels.emplace_back(
            boost::json::object{{"id", c.id},
                                {"ready", r.ready},
                                {"segment", r.segment},
                                {"reacquisitions", r.reacquisitions},
                                {"cfo_hz", finite(r.frequency_hz)},
                                {"source_rate_ppm", finite(r.residual_rate_ppm)},
                                {"pilot_snr_db", finite(r.pilot_snr_db)},
                                {"timing_samples", finite(r.timing_samples)},
                                {"conditional_timing_sigma_samples", finite(r.timing_sigma_samples)},
                                {"phase_rms_degrees", finite(r.phase_rms_degrees)},
                                {"innovation_samples", finite(o.innovation_samples)},
                                {"innovation_variance", finite(o.innovation_variance)},
                                {"normalized_innovation_square", finite(o.normalized_innovation)},
                                {"timing_downweighted", o.downweighted},
                                {"timing_late", o.late},
                                {"rs_good", c.rs_good},
                                {"rs_bad", c.rs_bad},
                                {"rows", c.rows},
                                {"valid_rows", c.valid_rows},
                                {"response_relative_error", finite(c.response_error)}});
    }
    impl_->send({{"type", "status"},
                 {"elapsed_s", s.elapsed},
                 {"processed_s", s.processed_seconds},
                 {"phase", capture_phase_name(s.phase)},
                 {"headroom", finite(s.headroom)},
                 {"recent_headroom", finite(s.recent_headroom)},
                 {"minimum_recent_headroom", finite(s.minimum_recent_headroom)},
                 {"maximum_block_ms", s.maximum_block_ms},
                 {"queue_ms", s.queue_ms},
                 {"peak_queue_ms", s.peak_queue_ms},
                 {"capacity_ms", s.capacity_ms},
                 {"display_pressure_skips", s.display_pressure_skips},
                 {"archive_queued_bytes", s.archive_queued_bytes},
                 {"diagnostic_queued_bytes", queued_bytes()},
                 {"missing", s.missing},
                 {"discarded", s.host_discarded},
                 {"unknown_gaps", s.unknown_gaps},
                 {"clock_ppm", s.clock_ppm},
                 {"clock_sigma_ns", finite(s.clock_sigma_ns)},
                 {"clock_rate_sigma_ppm", finite(s.clock_rate_sigma_ppm)},
                 {"rf_samples", s.rf.samples},
                 {"rf_rms_dbfs", finite(s.rf.rms_dbfs())},
                 {"rf_peak_component_dbfs", finite(s.rf.peak_dbfs())},
                 {"rail_samples", s.rf.rail_samples},
                 {"near_rail_samples", s.rf.near_rail_samples},
                 {"channels", std::move(channels)}});
}
void CaptureDiagnostics::time_anchor(const DeviceTimeAnchor &a) {
    impl_->send({{"type", "utc_anchor"},
                 {"device_seconds", a.device_seconds},
                 {"host_utc_ns", a.host_utc_ns},
                 {"host_monotonic_ns", a.host_monotonic_ns},
                 {"query_uncertainty_ns", a.query_uncertainty_ns},
                 {"absolute_utc_accuracy", "host clock; not calibrated"}});
}
void CaptureDiagnostics::gap(const CaptureChunk &c) {
    impl_->send({{"type", "capture_segment"}, {"source_first", c.source_first}, {"segment", c.segment}});
}
void CaptureDiagnostics::finish() {
    if (!impl_->worker.joinable())
        return;
    impl_->send({{"type", "complete"}});
    impl_->finish();
}
} // namespace radar
