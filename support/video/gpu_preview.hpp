#pragma once
#include "gpu_compute.hpp"
#include "range_doppler.hpp"

namespace radar::video {
class GpuPreview {
    gpu::Kernel kernel_;
public:
    unsigned width{},height{};
    std::vector<float> power;
    ComputeTimings timing;
    explicit GpuPreview(std::shared_ptr<gpu::Context>);
    void calculate(const DopplerMap&,const gpu::BufferView&,bool normalize);
};
}
