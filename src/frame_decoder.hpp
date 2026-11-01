#pragma once
#include "cpu_fec.hpp"
#include "gpu_frames.hpp"

namespace radar {
struct DecodedFrame {
    FrameBatch frame;
    std::vector<fec::RsResult> packets;
    std::size_t trellis_bits{}, reused_trellis_bits{};
    FecTimings timing;
    unsigned passes{}, rescue_rounds{}, rescued_words{};
    FecBatch decoded() const {
        return {packets, packets.size(), trellis_bits, reused_trellis_bits,
                passes,  rescue_rounds,  rescued_words};
    }
};
// Bounded GPU snapshots and CPU FEC workers. Submission, completion ordering
// and ownership are independent of response fitting or downstream consumers.
// All public methods run on the producer thread. Workers own independent
// scratch and full traceback; GPU tables stay within GPU frame storage.
class FrameDecoder {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    FrameDecoder(unsigned workers, unsigned slots, std::shared_ptr<gpu::Context>);
    ~FrameDecoder();
    bool full() const;
    bool empty() const;
    bool ready() const;
    void submit(const FrameSelection &, gpu::Commands &, ComputeTimings * = nullptr);
    // Prefer a ready per-channel head; otherwise wait for the oldest head.
    // Channels may complete independently; each channel retains capture order.
    const DecodedFrame &front(); // Propagates worker errors; borrowed until pop.
    void pop();
};
} // namespace radar
