#pragma once
#include "gpu_receiver.hpp"
#include <functional>

namespace radar {
struct FrameBatch {
    OfdmBatch carriers;
    std::vector<TrackedSymbol> symbols;
    std::vector<Tps> frames;
    // Only these rows are emitted. Other rows supply decoder/interleaver history.
    std::size_t first{},last{};
    unsigned channel{},segment{};
};
struct FrameSelection {
    FrameBatch frame; // Carriers refer to a per-channel ring, not contiguous rows.
    std::vector<std::uint32_t> rows;
};
namespace gpu {class Commands;}
// Owns a contiguous snapshot independently of the receiver and frame rings.
// Capture must be recorded during the selection callback, before ring reuse.
class GpuFrameStorage {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuFrameStorage(std::shared_ptr<gpu::Context> context);
    ~GpuFrameStorage();
    FrameBatch capture(const FrameSelection&,ComputeTimings* =nullptr,gpu::Commands* =nullptr);
    void wait();
};
// Bounded per-channel rings and TPS alignment. Selection is CPU metadata only;
// the consumer chooses the lifetime and readiness of its own carrier snapshot.
class GpuFrames {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    using Consume=std::function<void(const FrameBatch&)>;
    using Select=std::function<void(const FrameSelection&)>;
    GpuFrames(std::span<const unsigned> guards,std::shared_ptr<gpu::Context> context);
    ~GpuFrames();
    void append(const ReceiverBatch& input,const Consume& consume,ComputeTimings* timings=nullptr);
    void finish(const Consume& consume,ComputeTimings* timings=nullptr);
    void select(const ReceiverBatch&,const Select&,ComputeTimings* =nullptr,gpu::Commands* =nullptr);
    void finish_select(const Select&);
    void wait();
};
}
