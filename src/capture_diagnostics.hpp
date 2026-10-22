#pragma once
#include "capture_monitor.hpp"
#include "sar_handoff.hpp"
#include <boost/json/object.hpp>

namespace radar {
// Small, bounded asynchronous sidecar. Only owned JSON enters the writer.
// A write failure is fatal rather than silently losing diagnostic provenance.
class CaptureDiagnostics {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    CaptureDiagnostics(const std::filesystem::path&,boost::json::object configuration);
    ~CaptureDiagnostics();
    void observations(std::span<const ClockObservation>);
    void spectra(const SarBatch&);
    void status(const CaptureStatus&);
    void time_anchor(const DeviceTimeAnchor&);
    void gap(const CaptureChunk&);
    void finish();
    std::size_t queued_bytes() const;
};
}
