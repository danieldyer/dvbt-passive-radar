#pragma once
#include "gpu_pipeline.hpp"

namespace radar {
// Robust physical carrier variance, learned in source-divided coordinates.
// It includes coherent narrowband interferers and never changes receive phase.
class GpuNoise {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuNoise(std::shared_ptr<gpu::Context>);
    ~GpuNoise();
    gpu::BufferView estimate(const SpectralBatch&,ComputeTimings* =nullptr);
    void reset();
};
}
