#pragma once
#include "gpu_frames.hpp"
#include "cpu_fec.hpp"

namespace radar {
struct DecodedFrame {
    FrameBatch frame;
    std::vector<fec::RsResult> packets;
    std::size_t trellis_bits{};
    FecTimings timing;
    FecBatch decoded() const {return {packets,packets.size(),trellis_bits};}
};
// Bounded GPU snapshots and CPU FEC workers. Submission, completion ordering
// and ownership are independent of response fitting or downstream consumers.
// The immutable modem tables must outlive this queue. All public methods run
// on the producer thread; only decode scratch is private to each worker.
class FrameDecoder {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    FrameDecoder(unsigned workers,unsigned slots,const Modem&,std::shared_ptr<gpu::Context>);
    ~FrameDecoder();
    bool full() const;
    bool empty() const;
    bool ready() const;
    void submit(const FrameSelection&,gpu::Commands&,ComputeTimings* =nullptr);
    // Prefer a ready per-channel head; otherwise wait for the oldest head.
    // Channels may complete independently; each channel retains capture order.
    const DecodedFrame& front(); // Propagates worker errors; borrowed until pop.
    void pop();
};
}
