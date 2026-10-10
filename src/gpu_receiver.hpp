#pragma once
#include "gpu_ofdm.hpp"
#include "channel_batch.hpp"

namespace radar {
struct TrackedSymbol {
    OfdmWindow window;
    unsigned channel{},segment{},modulo4{};
    C tps{}; // Gain-weighted, PRBS-corrected repeated TPS carrier observation.
    bool ready=false,discontinuity=false;
};
struct TimingObservation {
    enum class Kind {reset,rate,offset};
    Kind kind;unsigned channel;
    double sample,value; // Corrected channel-grid coordinate; offset in channel samples.
};
struct ReceiverBatch {
    OfdmBatch carriers;
    std::vector<TrackedSymbol> symbols;
    std::vector<TimingObservation> timing;
};
// Streaming control uses reduced CP/pilot observations; samples and spectra
// remain device-resident. Every channel observes the same external ADC clock.
class GpuReceiver {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    GpuReceiver(const PilotPattern& pilots,std::span<const unsigned> guards,std::size_t max_samples,
        std::shared_ptr<gpu::Context> context);
    ~GpuReceiver();
    ReceiverBatch process(const ChannelBatch& input,ComputeTimings* timings=nullptr,gpu::Commands* commands=nullptr);
    void wait();
    std::vector<Symbol> download(const ReceiverBatch& batch) const;
};
}
