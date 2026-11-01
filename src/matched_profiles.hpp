#pragma once
#include "gpu_pipeline.hpp"

namespace radar {
// Same 8 MS/s lag coordinates for every layout. The 3072-sample layout keeps
// 2976 positive lags (111.5 km), 64 negative lags and a zero seam. This covers
// 100 km plus the admitted migration context. Retain 4096 for legacy archives.
inline constexpr unsigned matched_profile_size = 1024;
inline constexpr unsigned matched_positive_context = 928, matched_negative_context = 64;
inline constexpr unsigned matched_extended_profile_size = 4096;
inline constexpr unsigned matched_compact_profile_size = 3072;
inline constexpr bool matched_profile_size_supported(std::size_t size) {
    return size == matched_profile_size || size == matched_compact_profile_size ||
           size == matched_extended_profile_size;
}
inline constexpr double matched_max_range_metres = 100000;
inline unsigned matched_size_for_range(double metres) {
    return metres > 30000 ? matched_compact_profile_size : matched_profile_size;
}
inline unsigned matched_context_for_size(unsigned size) {
    return size - matched_negative_context - 32;
}
struct MatchedProfiles {
    std::vector<C> values; // [moment=0,1][row][sampled excess delay]
    std::size_t first{}, count{};
    double information = 1; // Inverse-noise template energy used by normalization.
    unsigned size = matched_profile_size;
};
// Linear, finite-waveform correlation. The previous OFDM symbol supplies the
// reference before the guard; the first time moment retains within-symbol Doppler.
class GpuMatchedProfiles {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit GpuMatchedProfiles(std::shared_ptr<gpu::Context>, unsigned size = matched_profile_size,
                                unsigned tile = 64);
    ~GpuMatchedProfiles();
    MatchedProfiles calculate(const SpectralBatch &, gpu::BufferView noise, unsigned guard,
                              ComputeTimings * = nullptr);
    // Caller-owned output keeps its capacity across batches; no borrowed GPU
    // storage escapes the call; steady-size batches reuse the output allocation.
    void calculate_into(MatchedProfiles &, const SpectralBatch &, gpu::BufferView noise, unsigned guard,
                        ComputeTimings * = nullptr);
};
} // namespace radar
