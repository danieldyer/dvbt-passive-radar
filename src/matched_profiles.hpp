#pragma once
#include "gpu_pipeline.hpp"

namespace radar {
inline constexpr unsigned matched_profile_size=1024;
inline constexpr unsigned matched_positive_context=928,matched_negative_context=64;
struct MatchedProfiles {
    std::vector<C> values; // [moment=0,1][row][sampled excess delay]
    std::size_t first{},count{};
    double information=1; // Inverse-noise template energy used by normalization.
    std::vector<double> cancellation_variance; // Per-carrier frozen transfer error; correlated across rows.
};
// Linear, finite-waveform correlation. The previous OFDM symbol supplies the
// reference before the guard; the first time moment retains within-symbol Doppler.
class GpuMatchedProfiles {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuMatchedProfiles(std::shared_ptr<gpu::Context>);
    ~GpuMatchedProfiles();
    MatchedProfiles calculate(const SpectralBatch&,gpu::BufferView noise,unsigned guard,
                              ComputeTimings* =nullptr,bool cancel_direct=true,std::span<const C> stationary={});
};
}
