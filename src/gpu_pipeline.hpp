#pragma once
#include "gpu_frames.hpp"
#include "gpu_response.hpp"

namespace radar {
class SampleClock;
struct GpuPipelineConfig {
    std::vector<unsigned> guards;
    std::size_t max_samples{};
    std::uint64_t origin{};
    unsigned decimation=1,response_order=8;
    double reference_delay=0;
    bool automatic_origin=true;
    unsigned fec_workers=0; // Zero: one per channel, capped by hardware concurrency.
    unsigned frame_slots=0; // Zero: workers + two; bounded backpressure, no dropped frames.
};
struct GpuPipelineTimings {
    ComputeTimings receiver,frames,reconstruction,response;
    FecTimings fec;
};
struct SpectralBatch {
    const FrameBatch& frame;
    FecBatch packets;
    ReferenceBatch transmitted;
    ResponseEstimate estimate;
    ResidualBatch residual;
};
// ChannelBank -> this core -> complex spectral consumer. FEC runs on CPU;
// spectrum processing uses the supplied Vulkan context. Complete frame batches
// decode independently on CPU workers. The callback consumes borrowed buffers
// synchronously on the caller thread, in per-channel capture order; it may be deferred until
// a later process/drain/finish call. Each call supplies its own consumer.
// SarBatch attaches the physical RF declarations and common-clock trajectory.
// Propagation geometry and any physical calibration belong to the SAR consumer.
class GpuPipeline {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    using Consume=std::function<void(const SpectralBatch&)>;
    GpuPipeline(const GpuPipelineConfig& config,std::shared_ptr<gpu::Context> context,std::shared_ptr<SampleClock> clock);
    ~GpuPipeline();
    void process(const ChannelBatch& input,const Consume& consume,GpuPipelineTimings* timings=nullptr);
    void finish(const Consume& consume,GpuPipelineTimings* timings=nullptr);
    // Complete queued batches without finalizing partial TPS frames.
    void drain(const Consume& consume,GpuPipelineTimings* timings=nullptr);
};
}
