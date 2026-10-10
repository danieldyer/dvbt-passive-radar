#include "matched_profiles.hpp"
#include "gpu_fft_plan.hpp"
#include <cmath>
#include <cstring>
#include <numbers>

namespace radar {
namespace {
constexpr unsigned capacity=128,fft_size=8192;
constexpr std::uint32_t pack_code[] =
#include "matched_pack.inc"
;
constexpr std::uint32_t pad_code[] =
#include "matched_pad.inc"
;
constexpr std::uint32_t product_code[] =
#include "matched_product.inc"
;
constexpr std::uint32_t extract_code[] =
#include "matched_extract.inc"
;
struct Parameters {unsigned start,count,total,capacity,guard;int origin;float fraction,gain;unsigned cancel_direct,output_count,output_first;};
}
struct GpuMatchedProfiles::Impl {
    gpu::Kernel kernel;
    vk::UniquePipeline pad,product,extract;
    std::unique_ptr<gpu::FftPlan> samples,moment,inverse;
    double last_origin=-1;
    explicit Impl(std::shared_ptr<gpu::Context> context)
        :kernel(pack_code,sizeof(Parameters),{},std::move(context)),pad(kernel.pipeline(pad_code)),
         product(kernel.pipeline(product_code)),extract(kernel.pipeline(extract_code)) {
        kernel.storage(5,2*(capacity+1)*8192*sizeof(C));
        kernel.storage(6,2*capacity*fft_size*sizeof(C));
        kernel.storage(7,2*1020*matched_profile_size*sizeof(C));
        kernel.storage(9,6817*sizeof(C));
        samples=std::make_unique<gpu::FftPlan>(kernel,5,8192,2*(capacity+1));
        moment=std::make_unique<gpu::FftPlan>(kernel,6,fft_size,capacity,capacity*fft_size*sizeof(C));
        inverse=std::make_unique<gpu::FftPlan>(kernel,6,fft_size,2*capacity);
    }
    ~Impl(){kernel.wait();}
};
GpuMatchedProfiles::GpuMatchedProfiles(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuMatchedProfiles::~GpuMatchedProfiles()=default;
MatchedProfiles GpuMatchedProfiles::calculate(const SpectralBatch& batch,gpu::BufferView noise,unsigned guard,ComputeTimings* timing,bool cancel_direct,std::span<const C> stationary) {
    auto& p=*impl_;const auto& f=batch.frame;
    const double origin=batch.estimate.reference_origin;
    if((guard!=4 && guard!=8 && guard!=16 && guard!=32) || !std::isfinite(origin) || origin<0 || origin>256 ||
       f.last<f.first || f.last>batch.transmitted.count || batch.transmitted.count>1020 ||
       !batch.estimate.response.mapped || batch.estimate.response.bytes<6817*sizeof(C) || !noise.mapped || noise.bytes<6817*sizeof(float))
        throw std::invalid_argument("Invalid finite reference correlation coordinates");
    const auto first=std::min(f.last,std::max<std::size_t>(1,f.first)),count=f.last-first;
    MatchedProfiles output{{},first,count};output.values.resize(2*count*matched_profile_size);
    if(!count)return output;
    if(!stationary.empty() && stationary.size()!=6817)throw std::invalid_argument("Incomplete stationary background");
    static constexpr std::array<C,6817> zero{};
    p.kernel.upload<C>(9,stationary.empty() ? std::span<const C>{zero} : stationary);
    if(p.last_origin!=origin){
        std::array<C,2*6817> rotations;
        for(unsigned k=0;k<6817;++k){
            const auto phase=-2*std::numbers::pi*(int(k)-3408)/8192;
            rotations[k]=static_cast<C>(std::polar(1.0,phase*(origin-std::floor(origin))));
            rotations[6817+k]=static_cast<C>(std::polar(1.0,phase*origin));
        }
        p.kernel.upload<C>(8,rotations);p.last_origin=origin;
    }
    p.kernel.bind(0,f.carriers.measured);p.kernel.bind(1,batch.transmitted.carriers);
    p.kernel.bind(2,batch.transmitted.valid);p.kernel.bind(3,batch.estimate.response);p.kernel.bind(4,noise);
    const auto* h=static_cast<const C*>(batch.estimate.response.mapped);
    const auto* variance=static_cast<const float*>(noise.mapped);
    double gain=0;for(unsigned k=0;k<6817;++k)gain+=std::norm(h[k])/std::max(double(variance[k]),1e-30);
    gain/=8192;
    if(!std::isfinite(gain) || gain<=0)throw std::runtime_error("No finite matched-template energy");
    output.information=gain;
    const auto cmd=p.kernel.begin(timing);
    for(std::size_t offset=0;offset<count;offset+=capacity) {
        const auto n=std::min<std::size_t>(capacity,count-offset);
        const Parameters push{unsigned(first+offset),unsigned(n),unsigned(batch.transmitted.count),capacity,
                              guard,int(std::floor(origin)),float(origin-std::floor(origin)),float(gain),unsigned(cancel_direct),unsigned(count),unsigned(first)};
        p.kernel.dispatch(cmd,p.kernel.primary(),push,2*(capacity+1)*8192);p.kernel.barrier(cmd);
        p.samples->append(cmd,true);
        p.kernel.dispatch(cmd,*p.pad,push,capacity*fft_size);p.kernel.barrier(cmd);
        p.moment->append(cmd,false);
        p.kernel.dispatch(cmd,*p.product,push,capacity*fft_size);p.kernel.barrier(cmd);
        p.inverse->append(cmd,true);
        p.kernel.dispatch(cmd,*p.extract,push,capacity*matched_profile_size);p.kernel.barrier(cmd);
    }
    p.kernel.finish(cmd);
    const auto* result=static_cast<const C*>(p.kernel.view(7).mapped);
    std::copy_n(result,output.values.size(),output.values.data());
    return output;
}
}
