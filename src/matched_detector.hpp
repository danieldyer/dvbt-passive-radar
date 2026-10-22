#pragma once
#include "gpu_focus.hpp"
#include "gpu_noise.hpp"
#include "sar_handoff.hpp"
#include <functional>

namespace radar {
struct DetectionOptions {
    double integration_seconds=3,update_seconds=1;
    FocusOptions focus;
    bool publish_maps=true; // Range-history archiving can skip focusing entirely.
    bool best_effort_maps=false; // Live display may skip refreshes while its previous map is busy.
    // Offline compatibility. Live archives disable this: stationary nuisance
    // projection belongs to the display, after the scientific profile export.
    bool stationary_prediction=true;
};
struct DetectionTimings {ComputeTimings noise,profiles,focus;};
class MatchedDetector {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    using Consumer=std::function<void(std::size_t channel,DopplerMap&&)>;
    // Optional device consumer receives physical complex amplitudes on the owned
    // focus worker. It must finish borrowing the buffer before returning. The
    // later caller-thread Consumer receives metadata only.
    using Projector=std::function<void(DopplerMap&,const gpu::BufferView&)>;
    using ProfileConsumer=std::function<void(const SarBatch&,const MatchedProfiles&,std::span<const C> stationary,std::span<const float> variance)>;
    MatchedDetector(std::shared_ptr<gpu::Context>,std::size_t channels,DetectionOptions={},Projector={},ProfileConsumer={});
    ~MatchedDetector();
    void append(const SarBatch&,const Consumer&,DetectionTimings* =nullptr);
    // Drain owned focusing jobs. Borrowed SarBatch buffers never escape append.
    // Consumers and timing accumulators are called only on the caller's thread.
    void finish(const Consumer&,DetectionTimings* =nullptr);
    void reset(); // Drains jobs and clears observations; keeps compiled plans.
    // Called on the append/finish owner thread. Keeps bounded range history
    // while hiding a display, and avoids scheduling unnecessary focusing jobs.
    void set_publish_maps(std::size_t channel,bool);
};
}
