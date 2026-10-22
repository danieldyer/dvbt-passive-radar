#pragma once
#include "matched_detector.hpp"
#include "recording_metadata.hpp"
#include "capture_monitor.hpp"

namespace radar {
struct LiveDisplay {
    MatchedDetector::Projector project;
    MatchedDetector::Consumer publish;
    std::function<void(double)> observe_time;
    std::function<bool(std::size_t)> active;
    std::shared_ptr<void> lifetime;
};
using LiveDisplayFactory=std::function<LiveDisplay(std::shared_ptr<gpu::Context>,const RecordingMetadata&)>;
enum class CaptureDisplayMode {interactive_only,headless_projector};
int capture_cli(int argc,char** argv,LiveDisplayFactory={},CaptureUiRunner={},CaptureDisplayMode=CaptureDisplayMode::interactive_only);
}
