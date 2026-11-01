#pragma once
#include "gpu_pipeline.hpp"

namespace radar {
// Source-conditioned diagonal residual covariance with one unknown, stationary
// complex transfer per carrier. Weighted least-squares residuals with the exact
// one-complex-mean degrees-of-freedom correction; primary weights lag one batch.
// Time-varying nuisance modes are not removed from training while still present
// in primary observations. No spectral loading, clipping or receive phase edit.
// Independent innovations and stationary, target-free secondary observations
// are conditional assumptions, not established by this estimator.
class GpuNoise {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit GpuNoise(std::shared_ptr<gpu::Context>);
    ~GpuNoise();
    gpu::BufferView estimate(const SpectralBatch &, ComputeTimings * = nullptr);
    void reset();
    bool ready() const;
};
} // namespace radar
