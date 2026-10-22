#pragma once
#include "gpu_pipeline.hpp"

namespace radar {
// Keep the compact layout for existing 30 km processing and archives. Wider
// captures use a longer delay period, with the same 8 MS/s lag coordinates.
inline constexpr unsigned matched_profile_size=1024;
inline constexpr unsigned matched_positive_context=928,matched_negative_context=64;
inline constexpr unsigned matched_extended_profile_size=4096;
inline constexpr double matched_max_range_metres=100000;
inline unsigned matched_size_for_range(double metres){return metres>30000 ? matched_extended_profile_size : matched_profile_size;}
inline unsigned matched_context_for_size(unsigned size){return size-matched_negative_context-32;}
struct MatchedProfiles {
    std::vector<C> values; // [moment=0,1][row][sampled excess delay]
    std::size_t first{},count{};
    double information=1; // Inverse-noise template energy used by normalization.
    std::vector<double> cancellation_variance; // Per-carrier frozen transfer error; correlated across rows.
    unsigned size=matched_profile_size;
};
// Linear, finite-waveform correlation. The previous OFDM symbol supplies the
// reference before the guard; the first time moment retains within-symbol Doppler.
class GpuMatchedProfiles {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuMatchedProfiles(std::shared_ptr<gpu::Context>,unsigned size=matched_profile_size);
    ~GpuMatchedProfiles();
    MatchedProfiles calculate(const SpectralBatch&,gpu::BufferView noise,unsigned guard,
                              ComputeTimings* =nullptr,bool cancel_direct=true,std::span<const C> stationary={});
};
}
