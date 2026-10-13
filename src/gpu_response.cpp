#include "gpu_response.hpp"
#include "gpu_compute.hpp"
#include "dsp.hpp"
#include "pilot_phase.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>

namespace radar {
namespace {
constexpr double tau=2*std::numbers::pi;
constexpr std::uint32_t reduce_code[] =
#include "response_reduce.inc"
;
constexpr std::uint32_t fit_code[] =
#include "response_fit.inc"
;
constexpr std::uint32_t subgroup_code[] =
#include "response_fit_subgroup.inc"
;
constexpr std::uint32_t apply_code[] =
#include "response_apply.inc"
;
std::span<const std::uint32_t> solver_code(VkPhysicalDevice device) {
    const auto properties=vk::PhysicalDevice(device).getProperties2<vk::PhysicalDeviceProperties2,vk::PhysicalDeviceSubgroupProperties>();
    const auto& subgroup=properties.get<vk::PhysicalDeviceSubgroupProperties>();
    return (subgroup.supportedStages & vk::ShaderStageFlagBits::eCompute) &&
        (subgroup.supportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic)
        ? std::span<const std::uint32_t>{subgroup_code} : std::span<const std::uint32_t>{fit_code};
}
void validate(const gpu::BufferView& y,const ReferenceBatch& x,const std::shared_ptr<gpu::Context>& context) {
    if(!x.count || x.count>1020 || y.context!=context || x.carriers.context!=context || x.valid.context!=context ||
       y.bytes<x.count*6817*sizeof(C) || x.carriers.bytes<x.count*6817*sizeof(C) || x.valid.bytes<x.count*6817*sizeof(std::uint32_t))
        throw std::invalid_argument("Invalid resident response-fit batch");
}
}
struct GpuResponse::Impl {
    gpu::Kernel fit,apply;
    std::array<vk::UniquePipeline,33> solvers;
    unsigned cached_order=99;
    double cached_origin=std::numeric_limits<double>::quiet_NaN();
    Impl(std::shared_ptr<gpu::Context> context):fit(reduce_code,16,{},std::move(context)),apply(apply_code,16,{},fit.context()) {
        fit.storage(4,6817*sizeof(ResponseStatistics));fit.storage(5,6819*sizeof(C));fit.storage(8,6817*sizeof(float));
    }
    void rotation(double origin) {
        if(origin==cached_origin)return;
        std::vector<C> values(6817);
        for(unsigned k=0;k<6817;++k)values[k]=static_cast<C>(std::polar(1.0,-tau*(int(k)-3408)*origin/8192));
        fit.upload(3,std::span<const C>{values});cached_origin=origin;
    }
    void basis(unsigned order) {
        if(order==cached_order)return;
        std::vector<C> values((2*order+1)*6817);
        const auto orthogonal=minimum_phase_basis(6817,order);
        std::transform(orthogonal.begin(),orthogonal.end(),values.begin(),[](auto z){return static_cast<C>(z);});
        fit.upload(6,std::span<const C>{values});
        if(!solvers[order]){
            const auto limits=vk::PhysicalDevice(fit.physical()).getProperties().limits;
            const unsigned lanes=std::bit_floor(std::min({512U,limits.maxComputeWorkGroupInvocations,limits.maxComputeWorkGroupSize[0]}));
            solvers[order]=fit.pipeline(solver_code(fit.physical()),std::array{order,lanes});
        }
        cached_order=order;
    }
    void offsets(std::size_t count,std::span<const double> values) {
        if(!values.empty() && values.size()!=count)throw std::invalid_argument("Incomplete source-clock coordinates");
        std::vector<float> coordinates(count);
        for(std::size_t i=0;i<values.size();++i){
            if(!std::isfinite(values[i]))throw std::invalid_argument("Nonfinite source-clock coordinate");
            coordinates[i]=static_cast<float>(values[i]);
        }
        fit.upload(7,std::span<const float>{coordinates});
    }
};
GpuResponse::GpuResponse(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuResponse::~GpuResponse()=default;
ResponseEstimate GpuResponse::fit(const gpu::BufferView& y,const ReferenceBatch& x,double origin,unsigned order,unsigned iterations,ComputeTimings* timings,gpu::Commands* commands,std::span<const double> offsets) {
    auto& p=*impl_;validate(y,x,p.fit.context());
    if(!std::isfinite(origin) || order>32 || !iterations || iterations>128)throw std::invalid_argument("Invalid minimum-phase fit configuration");
    p.rotation(origin);p.basis(order);p.offsets(x.count,offsets);p.fit.bind(0,y);p.fit.bind(1,x.carriers);p.fit.bind(2,x.valid);
    const std::array push{static_cast<unsigned>(x.count),order,iterations,0U};
    const bool detailed=timings && timings->detailed;
    const auto cmd=p.fit.begin(timings,detailed ? 6 : 2,commands);
    p.fit.dispatch(cmd,p.fit.primary(),push,6817);p.fit.barrier(cmd);
    if(detailed)p.fit.stamp(cmd,1);
    p.fit.groups(cmd,*p.solvers[order],push,1);
    if(detailed)for(unsigned i=2;i<5;++i)p.fit.stamp(cmd,i);
    p.fit.finish(cmd);
    ResponseEstimate estimate{p.fit.view(5),p.fit.view(4),x.count,origin,0,p.fit.view(7),p.fit.view(8)};
    if(!commands)read_diagnostics(estimate);
    return estimate;
}
void GpuResponse::read_diagnostics(ResponseEstimate& estimate) {
    const auto* h=static_cast<const C*>(estimate.response.mapped);
    if(!h || estimate.response.bytes<6819*sizeof(C) || h[6818].imag()!=1 ||
       !std::isfinite(h[6817].real()) || !std::isfinite(h[6818].real()))
        throw std::runtime_error("No finite reference energy for response fit");
    estimate.relative_error=h[6817].real();
}

ResidualBatch GpuResponse::apply(const gpu::BufferView& y,const ReferenceBatch& x,const ResponseEstimate& estimate,ComputeTimings* timings,gpu::Commands* commands,bool residual_spectra) {
    auto& p=*impl_;validate(y,x,p.fit.context());
    if(estimate.count!=x.count || estimate.response.context!=p.fit.context() || estimate.response.bytes<6817*sizeof(C) || !std::isfinite(estimate.reference_origin) ||
       (estimate.source_offsets.bytes && (estimate.source_offsets.context!=p.fit.context() || estimate.source_offsets.bytes<x.count*sizeof(float))))
        throw std::invalid_argument("Invalid minimum-phase estimate for residual");
    p.rotation(estimate.reference_origin);
    p.apply.bind(0,y);p.apply.bind(1,x.carriers);p.apply.bind(2,x.valid);p.apply.bind(3,p.fit.view(3));p.apply.bind(4,estimate.response);
    if(!estimate.source_offsets.bytes)p.offsets(x.count,{});
    p.apply.bind(8,estimate.source_offsets.bytes ? estimate.source_offsets : p.fit.view(7));
    p.apply.storage(5,x.count*6817*sizeof(C));
    for(unsigned b=6;b<8;++b)p.apply.storage(b,residual_spectra ? x.count*6817*sizeof(C) : sizeof(C));
    const auto cmd=p.apply.begin(timings,2,commands);
    p.apply.dispatch(cmd,p.apply.primary(),std::array{static_cast<unsigned>(x.count),unsigned(residual_spectra),0U,0U},x.count*6817);p.apply.finish(cmd);
    return {p.apply.view(5),residual_spectra ? p.apply.view(6) : gpu::BufferView{},residual_spectra ? p.apply.view(7) : gpu::BufferView{},estimate.response,x.valid,x.count,estimate.reference_origin};
}
double GpuResponse::direct_origin_adjustment(const ResponseEstimate& estimate) {
    if(!estimate.response.mapped || !estimate.statistics.mapped || estimate.response.bytes<6817*sizeof(C) ||
       estimate.statistics.bytes<6817*sizeof(ResponseStatistics))throw std::invalid_argument("Invalid origin calibration statistics");
    const auto* h=static_cast<const C*>(estimate.response.mapped);
    const auto* stats=static_cast<const ResponseStatistics*>(estimate.statistics.mapped);
    // One-time direct-reference calibration. Source-conditioned Gaussian ML
    // weights are reference energy, not a sequential unwrapped phase chain.
    // A coarse Fourier likelihood search chooses the delay branch; the complex
    // likelihood fit then refines it. This never alters the receive time grid.
    std::array<C,6817> measured{},predicted{};
    std::array<std::uint32_t,6817> indices{};
    std::vector<C> likelihood(8192);double energy=0;
    for(unsigned k=0;k<6817;++k) {
        indices[k]=k;
        if(stats[k].reference_power<=0)continue;
        const float scale=std::sqrt(stats[k].reference_power);
        measured[k]=stats[k].cross/scale;predicted[k]=h[k]*scale;
        likelihood[(k+8192-3408)%8192]=measured[k]*std::conj(predicted[k]);
        energy+=std::norm(predicted[k]);
    }
    if(!(energy>1e-30))throw std::runtime_error("No timing baseline for direct reference origin");
    Fft fft(8192);fft.transform(likelihood,true);
    const auto peak=std::max_element(likelihood.begin(),likelihood.end(),[](C a,C b){return std::norm(a)<std::norm(b);});
    const auto index=peak-likelihood.begin();const double coarse=index<=4096 ? double(index) : double(index)-8192;
    return -pilot_phase(measured,predicted,indices,-tau*coarse/8192).slope*8192/tau;
}
}
