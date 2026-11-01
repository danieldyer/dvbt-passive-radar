#include "live_archive.hpp"
#include <bit>
#include <boost/json.hpp>
#include <condition_variable>
#include <deque>
#include <thread>

namespace radar {
namespace {
template <class T> void append(std::vector<char> &data, std::span<const T> values) {
    if (values.empty())
        return;
    const auto *first = reinterpret_cast<const char *>(values.data());
    data.insert(data.end(), first, first + values.size_bytes());
}
} // namespace
struct LiveArchive::Impl {
    static constexpr std::size_t limit = 64 * 1024 * 1024;
    struct Record {
        std::vector<char> header, payload;
        std::size_t size() const {
            return header.size() + payload.size();
        }
    };
    std::ofstream file;
    std::deque<Record> records;
    std::vector<char> spare;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::size_t queued = 0;
    std::optional<ClockSegment> last_clock;
    bool done = false;
    std::exception_ptr failure;
    std::atomic<std::uint64_t> written = 0;
    std::jthread worker;
    Impl(const std::filesystem::path &path, const RecordingMetadata &description,
         const AppliedCapture &applied, const SarCapture &capture, std::string_view mode) {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(path, std::ios::binary | std::ios::out | std::ios::noreplace);
        static_assert(std::endian::native == std::endian::little);
        file.write("DVBTLIV1", 8);
        auto header = boost::json::parse(description.json());
        auto &root = header.as_object();
        root["archive_mode"] = mode;
        root["corrected_rate_hz"] = capture.input_rate;
        root["archive_sample_format"] = "complex_f32le";
        root["profile_row_policy"] = "fixed_grid_zero_information_erasures";
        root["template_weighting"] = "unmodified_reconstructed_waveform";
        root["noise_weighting"] = "previous_batch_diagonal_residual_covariance";
        root["noise_initialization"] = "first_batch_estimate_then_previous_batch";
        root["stationary_suppression"] = "display_only_constant_projection";
        root["profile_stationary_subtraction"] = false;
        root["profile_training_rows_retained"] = true;
        root["profile_row_coordinates"] =
            boost::json::array{"sample",       "seconds", "original_adc",          "phase",
                               "frequency_hz", "valid",   "source_offset_samples", "source_rate_error"};
        root["source_timing"] = "reference_phase_only; receive samples share one clock";
        root["source_nominal_rate_hz"] = capture.source_rate;
        root["decimation"] = capture.decimation;
        root["origin"] = capture.origin;
        root["applied_capture"] = boost::json::object{{"driver", applied.driver},
                                                      {"device", applied.device},
                                                      {"serial", applied.serial},
                                                      {"rate_hz", applied.rate_hz},
                                                      {"centre_hz", applied.centre_hz},
                                                      {"gain_db", applied.gain_db},
                                                      {"bandwidth_hz", applied.bandwidth_hz},
                                                      {"antenna", applied.antenna},
                                                      {"clock_source", applied.clock_source},
                                                      {"time_source", applied.time_source},
                                                      {"master_clock_hz", applied.master_clock_hz}};
        boost::json::array channels;
        for (const auto &c : capture.channels)
            channels.emplace_back(
                boost::json::object{{"rf_hz", c.rf_hz}, {"nco_hz", c.nco_hz}, {"guard", c.guard}});
        root["applied_channels"] = std::move(channels);
        const auto text = boost::json::serialize(header);
        const auto length = std::uint32_t(text.size());
        file.write(reinterpret_cast<const char *>(&length), 4);
        file.write(text.data(), std::streamsize(text.size()));
        written = 12 + text.size();
        worker = std::jthread([this] {
            try {
                for (;;) {
                    Record record;
                    {
                        std::unique_lock lock(mutex);
                        condition.wait(lock, [&] { return done || !records.empty(); });
                        if (records.empty())
                            break;
                        record = std::move(records.front());
                        records.pop_front();
                    }
                    file.write(record.header.data(), std::streamsize(record.header.size()));
                    if (!record.payload.empty())
                        file.write(record.payload.data(), std::streamsize(record.payload.size()));
                    written += record.size();
                    {
                        std::lock_guard lock(mutex);
                        queued -= record.size();
                        if (record.payload.capacity() <= limit &&
                            record.payload.capacity() > spare.capacity())
                            record.payload.swap(spare);
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
    void close() {
        {
            std::lock_guard lock(mutex);
            done = true;
            condition.notify_all();
        }
        if (worker.joinable())
            worker.join();
        if (failure)
            std::rethrow_exception(failure);
    }
    ~Impl() {
        // Drain owned records on every exit, but only finish() certifies that
        // the producer and all downstream processing completed successfully.
        try {
            close();
        } catch (...) {
        }
    }
    std::vector<char> payload(std::size_t bytes) {
        if (bytes > limit)
            throw std::runtime_error("Live archive payload exceeds bounded output queue");
        std::vector<char> result;
        {
            std::lock_guard lock(mutex);
            // One spare, at most 64 MiB. Do not attach a large profile buffer
            // to tiny IQ records: queued allocation stays <=2x payload bytes.
            if (spare.capacity() <= 2 * bytes)
                result.swap(spare);
        }
        result.clear();
        result.reserve(bytes);
        return result;
    }
    void send(const char *tag, boost::json::object metadata, std::vector<char> payload = {}) {
        const auto text = boost::json::serialize(metadata);
        Record record;
        record.header.reserve(16 + text.size());
        append(record.header, std::span{tag, 4});
        const std::uint32_t length = std::uint32_t(text.size());
        const std::uint64_t bytes = payload.size();
        append(record.header, std::span{&length, 1});
        append(record.header, std::span{&bytes, 1});
        append(record.header, std::span{text.data(), text.size()});
        // The payload is already owned. Do not copy every channel/profile byte
        // a second time merely to prepend a small record header.
        record.payload = std::move(payload);
        if (record.size() > limit)
            throw std::runtime_error("Live archive record exceeds bounded output queue");
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return failure || done || queued + record.size() <= limit; });
        if (failure)
            std::rethrow_exception(failure);
        if (done)
            throw std::runtime_error("Live archive is closed");
        queued += record.size();
        records.push_back(std::move(record));
        condition.notify_one();
    }
    void clock(std::span<const ClockSegment> segments) {
        if (segments.empty())
            return;
        // Batch snapshots may overlap or precede the most recently archived
        // clock change when independent channels finish out of order.
        if ((!last_clock && segments.front().start != 0) ||
            (last_clock && segments.front().start > last_clock->start))
            throw std::runtime_error("Live clock history omitted an unarchived interval");
        boost::json::array values;
        for (const auto &s : segments)
            if (!last_clock || s.start > last_clock->start)
                values.emplace_back(boost::json::array{s.start, s.offset, s.rate});
            else if (s.start == last_clock->start &&
                     (s.offset != last_clock->offset || s.rate != last_clock->rate))
                throw std::runtime_error("Live clock history changed an immutable segment");
        if (!values.empty()) {
            send("CLCK", {{"segments", std::move(values)}});
            last_clock = segments.back();
        }
    }
};
LiveArchive::LiveArchive(const std::filesystem::path &path, const RecordingMetadata &metadata,
                         const AppliedCapture &applied, const SarCapture &capture, std::string_view mode)
    : impl_(std::make_unique<Impl>(path, metadata, applied, capture, mode)) {}
LiveArchive::~LiveArchive() = default;
void LiveArchive::profiles(const SarBatch &batch, const MatchedProfiles &profiles,
                           std::span<const float> variance) {
    auto &p = *impl_;
    p.clock(batch.clock);
    const auto &f = batch.spectra.frame;
    boost::json::array rows;
    const auto *valid = static_cast<const std::uint32_t *>(batch.spectra.transmitted.valid.mapped);
    // Adjacent finite-waveform profiles share one complete symbol. Reduce
    // each carrier mask once; the integer OR also exposes SIMD reduction.
    const auto known = [&](std::size_t row) {
        std::uint32_t unknown = 0;
        for (unsigned k = 0; k < 6817; ++k)
            unknown |= valid[row * 6817 + k] == 0;
        return unknown == 0;
    };
    bool previous = profiles.count && known(profiles.first - 1);
    rows.reserve(profiles.count);
    for (std::size_t i = 0; i < profiles.count; ++i) {
        const auto row = profiles.first + i;
        const auto &symbol = f.symbols.at(row);
        const bool current = known(row);
        rows.emplace_back(boost::json::array{symbol.window.sample, batch.seconds(row),
                                             batch.original_adc(row), symbol.window.phase,
                                             symbol.window.frequency_hz, previous && current,
                                             symbol.source_offset_samples, symbol.source_rate_error});
        previous = current;
    }
    auto payload = p.payload((profiles.values.size() + 6817) * sizeof(C) + variance.size_bytes());
    append(payload, std::span<const C>{profiles.values});
    append(payload,
           std::span{static_cast<const C *>(batch.spectra.estimate.response.mapped), std::size_t(6817)});
    append(payload, variance);
    p.send(
        "PROF",
        {{"channel", f.channel},
         {"segment", f.segment},
         {"count", profiles.count},
         {"size", profiles.size},
         {"positive_context_samples", matched_context_for_size(profiles.size)},
         {"negative_context_samples", matched_negative_context},
         {"range_step_m", 299792458.0 / 8e6},
         {"moments", 2},
         {"order", "moment,row,range"},
         {"information", profiles.information},
         {"reference_origin", batch.spectra.estimate.reference_origin},
         {"stationary_carriers", 0},
         {"response_carriers", 6817},
         {"noise_carriers", variance.size()},
         {"cancellation_variance_carriers", 0},
         {"cancellation_covariance", "diag(D); conditional on fitted H and independent training innovations"},
         {"rows", std::move(rows)}},
        std::move(payload));
}
void LiveArchive::channels(const ChannelBatch &batch, std::span<const ClockSegment> clock) {
    auto &p = *impl_;
    p.clock(clock);
    auto payload = p.payload(batch.channels * batch.count * sizeof(C));
    const auto *input = static_cast<const C *>(batch.samples.mapped);
    for (std::size_t c = 0; c < batch.channels; ++c)
        append(payload, std::span{input + c * batch.stride, batch.count});
    p.send("CHIQ",
           {{"first", batch.first},
            {"count", batch.count},
            {"channels", batch.channels},
            {"order", "channel,sample"}},
           std::move(payload));
}
void LiveArchive::gap(const CaptureChunk &chunk) {
    impl_->send("GAPS", {{"source_first", chunk.source_first}, {"segment", chunk.segment}});
}
void LiveArchive::epoch(std::int64_t tick, double rate) {
    impl_->send("EPOC", {{"first_hardware_tick", tick},
                         {"actual_rate_hz", rate},
                         {"reference", "device clock; uncalibrated UTC"}});
}
void LiveArchive::time_anchor(const DeviceTimeAnchor &anchor) {
    impl_->send("UTCA", {{"device_seconds", anchor.device_seconds},
                         {"host_utc_ns", anchor.host_utc_ns},
                         {"host_monotonic_ns", anchor.host_monotonic_ns},
                         {"query_uncertainty_ns", anchor.query_uncertainty_ns},
                         {"absolute_utc_accuracy", "host clock; not calibrated"}});
}
void LiveArchive::finish() {
    auto &p = *impl_;
    if (!p.worker.joinable())
        return;
    p.send("DONE", {});
    p.close();
}
std::uint64_t LiveArchive::bytes() const {
    return impl_->written.load();
}
std::size_t LiveArchive::queued_bytes() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->queued;
}
} // namespace radar
