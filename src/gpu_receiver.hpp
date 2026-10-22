#pragma once
#include "gpu_ofdm.hpp"
#include "channel_batch.hpp"
#include <limits>

namespace radar {
struct TrackedSymbol {
    OfdmWindow window;
    unsigned channel{},segment{},modulo4{};
    C tps{}; // Gain-weighted, PRBS-corrected repeated TPS carrier observation.
    bool ready=false,discontinuity=false;
    // Transmitter waveform coordinates relative to the acquisition anchor.
    // The receive grid remains common to every channel. These describe the
    // source waveform on that grid, not a second resampling of the measurement.
    double source_offset_samples{},source_rate_error{};
};
struct TimingObservation {
    enum class Kind {reset,rate,offset};
    Kind kind;unsigned channel;
    double sample,value; // Corrected channel-grid coordinate; offset in channel samples.
    double variance=std::numeric_limits<double>::quiet_NaN(); // Conditional on the fixed pilot anchor.
};
struct ReceiverBatch {
    OfdmBatch carriers;
    std::vector<TrackedSymbol> symbols;
    std::vector<TimingObservation> timing;
};
// Reduced telemetry in corrected 8 MS/s channel coordinates. RMS values are
// exponentially weighted over 0.25 s, not estimates of absolute RF time.
struct ReceiverStatus {
    bool locked{},ready{};
    unsigned segment{},age{};
    std::uint64_t symbols{},reacquisitions{};
    double frequency_hz{},pilot_quality{},timing_samples{},timing_rms_samples{},
        residual_rate_ppm{},phase_rms_degrees{},pilot_snr_db{};
    double timing_sigma_samples=std::numeric_limits<double>::quiet_NaN();
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
    std::vector<ReceiverStatus> status() const;
    void reset();
    // Secondary data only: an earlier decoded frame supplies additive carrier
    // variance for future pilot likelihoods. A pre-acquisition frame is ignored.
    void pilot_variance(unsigned channel,std::uint64_t last_sample,std::span<const float>);
    std::vector<Symbol> download(const ReceiverBatch& batch) const;
};
}
