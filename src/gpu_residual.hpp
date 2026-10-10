#pragma once
#include "dsp.hpp"
namespace radar {
struct ResidualTimings { double upload{}, dispatch_wait{}, readback{}; };
class GpuResidual {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    GpuResidual();
    ~GpuResidual();
    std::vector<C> calculate(std::span<const C> measured,std::span<const C> reference,std::span<const C> response,
                             ResidualTimings* timings=nullptr);
};
}
