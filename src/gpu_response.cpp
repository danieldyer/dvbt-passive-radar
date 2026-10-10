#include "gpu_response.hpp"
#include "gpu_compute.hpp"
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
constexpr std::uint32_t apply_code[] =
#include "response_apply.inc"
;
void validate(const gpu::BufferView& y,const ReferenceBatch& x,const std::shared_ptr<gpu::Context>& context) {
    if(!x.count || x.count>1020 || y.context!=context || x.carriers.context!=context || x.valid.context!=context ||
       y.bytes<x.count*6817*sizeof(C) || x.carriers.bytes<x.count*6817*sizeof(C) || x.valid.bytes<x.count*6817*sizeof(std::uint32_t))
        throw std::invalid_argument("Invalid resident response-fit batch");
}
}
struct GpuResponse::Impl {
    gpu::Kernel fit,apply;
    vk::UniquePipeline solver;
    unsigned cached_order=99;
    double cached_origin=std::numeric_limits<double>::quiet_NaN();
    Impl(std::shared_ptr<gpu::Context> context):fit(reduce_code,16,{},std::move(context)),apply(apply_code,16,{},fit.context()),solver(fit.pipeline(fit_code)) {
        fit.storage(4,6817*sizeof(ResponseStatistics));fit.storage(5,6819*sizeof(C));
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
        for(unsigned d=0;d<=2*order;++d)for(unsigned k=0;k<6817;++k) {
            const auto value=std::polar(1.0,-tau*((d+1)/2)*(int(k)-3408)/8192)*
                (d && !(d&1) ? std::complex<double>{0,1} : std::complex<double>{1,0});
            values[d*6817+k]=static_cast<C>(value);
        }
        fit.upload(6,std::span<const C>{values});cached_order=order;
    }
};
GpuResponse::GpuResponse(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuResponse::~GpuResponse()=default;
ResponseEstimate GpuResponse::fit(const gpu::BufferView& y,const ReferenceBatch& x,double origin,unsigned order,unsigned iterations,ComputeTimings* timings,gpu::Commands* commands) {
    auto& p=*impl_;validate(y,x,p.fit.context());
    if(!std::isfinite(origin) || order>32 || !iterations || iterations>128)throw std::invalid_argument("Invalid minimum-phase fit configuration");
    p.rotation(origin);p.basis(order);p.fit.bind(0,y);p.fit.bind(1,x.carriers);p.fit.bind(2,x.valid);
    const std::array push{static_cast<unsigned>(x.count),order,iterations,0U};
    const auto cmd=p.fit.begin(timings,2,commands);
    p.fit.dispatch(cmd,p.fit.primary(),push,6817);p.fit.barrier(cmd);
    p.fit.groups(cmd,*p.solver,push,1);p.fit.finish(cmd);
    ResponseEstimate estimate{p.fit.view(5),p.fit.view(4),x.count,origin,0};
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

ResidualBatch GpuResponse::apply(const gpu::BufferView& y,const ReferenceBatch& x,const ResponseEstimate& estimate,ComputeTimings* timings,gpu::Commands* commands) {
    auto& p=*impl_;validate(y,x,p.fit.context());
    if(estimate.count!=x.count || estimate.response.context!=p.fit.context() || estimate.response.bytes<6817*sizeof(C) || !std::isfinite(estimate.reference_origin))
        throw std::invalid_argument("Invalid minimum-phase estimate for residual");
    p.rotation(estimate.reference_origin);
    p.apply.bind(0,y);p.apply.bind(1,x.carriers);p.apply.bind(2,x.valid);p.apply.bind(3,p.fit.view(3));p.apply.bind(4,estimate.response);
    for(unsigned b=5;b<8;++b)p.apply.storage(b,x.count*6817*sizeof(C));
    const auto cmd=p.apply.begin(timings,2,commands);
    p.apply.dispatch(cmd,p.apply.primary(),std::array{static_cast<unsigned>(x.count),0U,0U,0U},x.count*6817);p.apply.finish(cmd);
    return {p.apply.view(5),p.apply.view(6),p.apply.view(7),estimate.response,x.valid,x.count,estimate.reference_origin};
}
double GpuResponse::direct_origin_adjustment(const ResponseEstimate& estimate) {
    if(!estimate.response.mapped || !estimate.statistics.mapped || estimate.response.bytes<6817*sizeof(C) ||
       estimate.statistics.bytes<6817*sizeof(ResponseStatistics))throw std::invalid_argument("Invalid origin calibration statistics");
    const auto* h=static_cast<const C*>(estimate.response.mapped);
    const auto* stats=static_cast<const ResponseStatistics*>(estimate.statistics.mapped);
    double phase=0,previous=0,sw=0,sx=0,sy=0,sxx=0,sxy=0;
    for(unsigned k=0;k<6817;++k) {
        const auto cross=std::complex<double>(stats[k].cross);
        if(stats[k].reference_power<=0 || std::norm(h[k])<1e-30)continue;
        const double angle=std::arg(cross/std::complex<double>(h[k]));
        phase+=sw ? std::remainder(angle-previous,tau) : angle;previous=angle;
        const double w=std::abs(cross),x=double(k)-3408;
        sw+=w;sx+=w*x;sy+=w*phase;sxx+=w*x*x;sxy+=w*x*phase;
    }
    const double denominator=sw*sxx-sx*sx;
    if(denominator<=1e-30)throw std::runtime_error("No timing baseline for direct reference origin");
    return -(sw*sxy-sx*sy)/denominator*8192/tau;
}
}
