#pragma once
#include "capture_source.hpp"
#include "gpu_receiver.hpp"
#include "sample_clock.hpp"
#include <functional>

namespace radar {
enum class CapturePhase { configuring, warming, running, draining, stopped, failed };
struct CaptureChannelStatus {
    std::string id, name;
    double rf_hz{}, nco_hz{}, clock_fit_ns{}, response_error{}, map_time{};
    ReceiverStatus receiver;
    ClockObserverStatus clock;
    std::uint64_t rows{}, valid_rows{}, rs_good{}, rs_bad{}, corrected_bytes{}, maps{};
    unsigned qam{};
    std::string code_rate;
};
// Owned telemetry only. No GPU views, waveform buffers or device API handles
// escape to the UI, which may render on a different thread.
struct CaptureStatus {
    CapturePhase phase = CapturePhase::configuring;
    std::string message, source, output, metadata, archive, display;
    bool simulation{}, looping{}, applied{}, profile{}, remove_dc{};
    AppliedCapture settings;
    RfHealth rf;
    double elapsed{}, processed_seconds{}, headroom{}, queue_ms{}, capacity_ms{}, clock_ppm{}, phase_budget{},
        integration_seconds{}, doppler_hz{}, update_seconds{},
        range_metres = 100000, fec_ms_per_second{}, gpu_ms_per_second{}, recent_headroom{},
        minimum_recent_headroom{}, maximum_block_ms{}, peak_queue_ms{}, clock_sigma_ns{},
        clock_rate_sigma_ppm{};
    std::uint64_t received{}, missing{}, host_discarded{}, unknown_gaps{}, overflows{}, sequence_errors{},
        timeouts{}, timestamp_gaps{}, segments{}, replay_loops{}, archive_bytes{}, archive_queued_bytes{},
        clock_segments{}, display_pressure_skips{};
    std::vector<CaptureChannelStatus> channels;
};
std::string_view capture_phase_name(CapturePhase);
class CaptureMonitor {
  public:
    virtual ~CaptureMonitor() = default;
    virtual void publish(const CaptureStatus &) = 0;
    virtual bool stop_requested() const = 0;
};
using CaptureWork = std::function<int(CaptureMonitor &)>;
using CaptureUiRunner = std::function<int(const CaptureWork &)>;
} // namespace radar
