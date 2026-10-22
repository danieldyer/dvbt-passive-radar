#include "matched_profiles.hpp"
#include "gpu_fft_plan.hpp"
#include <bit>
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
constexpr std::uint32_t prefix_code[] =
#include "matched_prefix.inc"
;
constexpr std::uint32_t prefix_product_code[] =
#include "matched_prefix_product.inc"
;
struct Parameters {unsigned start,count,total,capacity,guard;int origin;float fraction,gain;unsigned cancel_direct,output_count,output_first,size,profile_size,positive_context;};
}
struct GpuMatchedProfiles::Impl {
    gpu::Kernel kernel;
    unsigned profile_size,prefix_size;
    vk::UniquePipeline pad,product,extract,prefix,prefix_product;
    struct Plans {
        gpu::FftPlan samples,moment,inverse,prefix_forward,prefix_inverse;
        Plans(gpu::Kernel& kernel,unsigned tile,unsigned prefix_size):samples(kernel,5,8192,2*(tile+1)),
            moment(kernel,6,fft_size,tile,tile*fft_size*sizeof(C)),inverse(kernel,6,fft_size,2*tile),
            prefix_forward(kernel,11,prefix_size,3*tile),prefix_inverse(kernel,11,prefix_size,2*tile){}
    };
    std::array<std::unique_ptr<Plans>,4> shapes;
    double last_origin=-1;
    explicit Impl(std::shared_ptr<gpu::Context> context,unsigned size)
        :kernel(pack_code,sizeof(Parameters),{},std::move(context)),profile_size(size),prefix_size(2*size),pad(kernel.pipeline(pad_code)),
         product(kernel.pipeline(product_code)),extract(kernel.pipeline(extract_code)),
         prefix(kernel.pipeline(prefix_code)),prefix_product(kernel.pipeline(prefix_product_code)) {
        if(size!=matched_profile_size && size!=matched_extended_profile_size)throw std::invalid_argument("Invalid matched profile size");
        kernel.storage(5,2*(capacity+1)*8192*sizeof(C));
        kernel.storage(6,2*capacity*fft_size*sizeof(C));
        kernel.storage(7,2*1020*profile_size*sizeof(C));
        kernel.storage(9,6817*sizeof(C));
        kernel.storage(11,3*capacity*prefix_size*sizeof(C));
        // All possible tail shapes are bounded and known before capture.
        // A discontinuity must not trigger compilation of an unseen FFT size.
        for(unsigned tile:{16U,32U,64U,128U})plans(tile);
    }
    Plans& plans(unsigned tile){
        auto& shape=shapes.at(std::bit_width(tile)-5);
        if(!shape)shape=std::make_unique<Plans>(kernel,tile,prefix_size);
        return *shape;
    }
    ~Impl(){kernel.wait();}
};
GpuMatchedProfiles::GpuMatchedProfiles(std::shared_ptr<gpu::Context> context,unsigned size):impl_(std::make_unique<Impl>(std::move(context),size)){}
GpuMatchedProfiles::~GpuMatchedProfiles()=default;
MatchedProfiles GpuMatchedProfiles::calculate(const SpectralBatch& batch,gpu::BufferView noise,unsigned guard,ComputeTimings* timing,bool cancel_direct,std::span<const C> stationary) {
    auto& p=*impl_;const auto& f=batch.frame;
    const double origin=batch.estimate.reference_origin;
    if((guard!=4 && guard!=8 && guard!=16 && guard!=32) || !std::isfinite(origin) || origin<0 || origin>256 ||
       f.last<f.first || f.last>batch.transmitted.count || batch.transmitted.count>1020 ||
       !batch.estimate.response.mapped || batch.estimate.response.bytes<6817*sizeof(C) || !noise.mapped || noise.bytes<6817*sizeof(float))
        throw std::invalid_argument("Invalid finite reference correlation coordinates");
    const auto first=std::min(f.last,std::max<std::size_t>(1,f.first)),count=f.last-first;
    MatchedProfiles output;output.size=p.profile_size;output.first=first;output.count=count;output.values.resize(2*count*output.size);
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
    const auto offsets=batch.estimate.source_offsets;
    if(offsets.bytes && (offsets.context!=p.kernel.context() || !offsets.mapped || offsets.bytes<batch.transmitted.count*sizeof(float)))
        throw std::invalid_argument("Incomplete matched-filter source clock");
    const auto* shifts=static_cast<const float*>(offsets.mapped);
    std::vector<std::array<float,4>> coordinates(batch.transmitted.count);
    for(std::size_t row=0;row<coordinates.size();++row){
        const double shift=shifts ? shifts[row] : 0,local=origin-shift;
        if(!std::isfinite(local) || local<0 || local>256)throw std::invalid_argument("Source clock moved outside finite reference support");
        coordinates[row]={float(std::floor(local)),float(local-std::floor(local)),float(shift),0};
    }
    p.kernel.upload(10,std::span<const std::array<float,4>>{coordinates});
    const auto* h=static_cast<const C*>(batch.estimate.response.mapped);
    const auto* variance=static_cast<const float*>(noise.mapped);
    double gain=0;for(unsigned k=0;k<6817;++k)gain+=std::norm(h[k])/std::max(double(variance[k]),1e-30);
    gain/=8192;
    if(!std::isfinite(gain) || gain<=0)throw std::runtime_error("No finite matched-template energy");
    output.information=gain;
    // Compile/cache only the four bounded batch shapes. The small tail must
    // not execute the transforms and scratch writes of a full 128-row tile.
    const auto tail=unsigned(std::max(std::size_t(16),std::bit_ceil(count%capacity)));
    if(count>=capacity)p.plans(capacity);
    if(count%capacity)p.plans(tail);
    const auto positive_context=matched_context_for_size(p.profile_size);
    const bool correct_prefix=std::ranges::any_of(coordinates,[&](const auto& c){return c[0]+positive_context>8192/guard;});
    const auto cmd=p.kernel.begin(timing);
    for(std::size_t offset=0;offset<count;offset+=capacity) {
        const auto n=std::min<std::size_t>(capacity,count-offset);
        const auto tile=n==capacity ? capacity : tail;auto& plans=p.plans(tile);
        const Parameters push{unsigned(first+offset),unsigned(n),unsigned(batch.transmitted.count),tile,
                              guard,int(std::floor(origin)),float(origin-std::floor(origin)),float(gain),unsigned(cancel_direct),unsigned(count),unsigned(first),p.prefix_size,p.profile_size,positive_context};
        p.kernel.dispatch(cmd,p.kernel.primary(),push,2*(tile+1)*8192);p.kernel.barrier(cmd);
        plans.samples.append(cmd,true);
        if(correct_prefix){
            p.kernel.dispatch(cmd,*p.prefix,push,3*tile*p.prefix_size);p.kernel.barrier(cmd);
            plans.prefix_forward.append(cmd,false);
            p.kernel.dispatch(cmd,*p.prefix_product,push,2*tile*p.prefix_size);p.kernel.barrier(cmd);
            plans.prefix_inverse.append(cmd,true);
        }
        p.kernel.dispatch(cmd,*p.pad,push,tile*fft_size);p.kernel.barrier(cmd);
        plans.moment.append(cmd,false);
        p.kernel.dispatch(cmd,*p.product,push,tile*fft_size);p.kernel.barrier(cmd);
        plans.inverse.append(cmd,true);
        p.kernel.dispatch(cmd,*p.extract,push,tile*p.profile_size);p.kernel.barrier(cmd);
    }
    p.kernel.finish(cmd);
    const auto* result=static_cast<const C*>(p.kernel.view(7).mapped);
    std::copy_n(result,output.values.size(),output.values.data());
    return output;
}
}
