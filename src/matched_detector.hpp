#pragma once
#include "gpu_focus.hpp"
#include "gpu_noise.hpp"
#include "sar_handoff.hpp"
#include <functional>

namespace radar {
struct DetectionOptions {
    double integration_seconds=3,update_seconds=1;
    FocusOptions focus;
    bool standardize_noise=false; // Physical complex amplitudes remain the API default.
};
struct DetectionTimings {ComputeTimings noise,profiles,focus;double calibration_cpu{},standardization_cpu{};};
// Positive, robust background calibration changes detection units, not phase.
// The raw complex map remains available until standardize_map() is requested.
void estimate_map_noise(DopplerMap&);
void standardize_map(DopplerMap&);
class MatchedDetector {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    using Consumer=std::function<void(std::size_t channel,DopplerMap&&)>;
    MatchedDetector(std::shared_ptr<gpu::Context>,std::size_t channels,DetectionOptions={});
    ~MatchedDetector();
    void append(const SarBatch&,const Consumer&,DetectionTimings* =nullptr);
    // Drain owned focusing jobs. Borrowed SarBatch buffers never escape append.
    // Consumers and timing accumulators are called only on the caller's thread.
    void finish(const Consumer&,DetectionTimings* =nullptr);
};
}
