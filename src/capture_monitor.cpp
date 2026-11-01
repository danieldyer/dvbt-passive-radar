#include "capture_monitor.hpp"
namespace radar {
std::string_view capture_phase_name(CapturePhase phase) {
    switch (phase) {
    case CapturePhase::configuring:
        return "CONFIGURING";
    case CapturePhase::warming:
        return "WARMING";
    case CapturePhase::running:
        return "RUNNING";
    case CapturePhase::draining:
        return "DRAINING";
    case CapturePhase::stopped:
        return "STOPPED";
    case CapturePhase::failed:
        return "FAILED";
    }
    return "UNKNOWN";
}
} // namespace radar
