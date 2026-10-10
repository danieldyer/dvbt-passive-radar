#include "gpu_noise.hpp"
#include "gpu_compute.hpp"
#include <cmath>

namespace radar {
namespace {
constexpr std::uint32_t modes_code[] =
#include "noise_modes.inc"
;
constexpr std::uint32_t variance_code[] =
#include "noise_variance.inc"
;
}
struct GpuNoise::Impl {
    gpu::Kernel kernel;
    vk::UniquePipeline variance;
    bool ready=false;
    explicit Impl(std::shared_ptr<gpu::Context> context)
        :kernel(modes_code,16,{},std::move(context)),variance(kernel.pipeline(variance_code)) {
        kernel.storage(4,1020*2*sizeof(C));kernel.storage(5,6817*sizeof(float));
        kernel.storage(6,6817*sizeof(float));kernel.storage(7,6817*sizeof(float));
    }
};
GpuNoise::GpuNoise(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuNoise::~GpuNoise()=default;
void GpuNoise::reset(){impl_->kernel.wait();impl_->ready=false;}
gpu::BufferView GpuNoise::estimate(const SpectralBatch& batch,ComputeTimings* timing) {
    auto& p=*impl_;const auto& f=batch.frame;
    if(f.last<=f.first+2 || f.last>batch.residual.count || batch.residual.count>1020)
        throw std::invalid_argument("Noise estimation requires three new symbol rows");
    p.kernel.bind(0,f.carriers.measured);p.kernel.bind(1,batch.residual.reference);
    p.kernel.bind(2,batch.transmitted.valid);p.kernel.bind(3,batch.estimate.response);
    const std::array push{unsigned(f.first),unsigned(f.last),unsigned(p.ready),0U};
    const auto cmd=p.kernel.begin(timing);
    p.kernel.groups(cmd,p.kernel.primary(),push,unsigned(f.last));p.kernel.barrier(cmd);
    p.kernel.dispatch(cmd,*p.variance,push,6817);p.kernel.finish(cmd);p.ready=true;
    auto* applied=static_cast<float*>(p.kernel.view(5).mapped);
    const auto* current=static_cast<const float*>(p.kernel.view(7).mapped);
    std::vector<float> ratios;ratios.reserve(6817);
    for(unsigned k=0;k<6817;++k)if(std::isfinite(current[k]) && current[k]>1e-29F && std::isfinite(applied[k]) && applied[k]>1e-29F)
        ratios.push_back(current[k]/applied[k]);
    double level=1;
    if(!ratios.empty()){
        const auto middle=ratios.begin()+ratios.size()/2;std::nth_element(ratios.begin(),middle,ratios.end());
        level=*middle;
    }
    // Thousands of carriers constrain the common noise level without fitting
    // an individual echo. Keep earlier spectral shape, but never underestimate
    // a newly strong carrier after the common-level adjustment.
    for(unsigned k=0;k<6817;++k)applied[k]=std::max(float(applied[k]*level),current[k]);
    std::vector<float> finite;finite.reserve(6817);
    for(unsigned k=0;k<6817;++k)if(std::isfinite(applied[k]) && applied[k]>1e-29F)finite.push_back(applied[k]);
    if(finite.empty())throw std::runtime_error("No observed carrier noise for regularized whitening");
    const auto middle=finite.begin()+finite.size()/2;std::nth_element(finite.begin(),middle,finite.end());
    const float floor=std::max(*middle*.001F,1e-30F);
    for(unsigned k=0;k<6817;++k)applied[k]=std::isfinite(applied[k]) ? std::max(applied[k],floor) : floor;
    return p.kernel.view(5);
}
}
