#include "spectral_clock.hpp"
#include "gpu_compute.hpp"
#include "clock_interpolator.hpp"
#include "channel_clock_interpolator.hpp"
#include "small_clock_interpolator.hpp"
#include <vkFFT.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numbers>
#include <numeric>

namespace radar {
namespace {
constexpr double tau=2*std::numbers::pi;
constexpr std::uint32_t boundary_code[] =
#include "spectral_boundary.inc"
;
constexpr std::uint32_t synth_code[] =
#include "spectral_synth.inc"
;
struct Plan {
    VkFFTApplication app{};
    bool ready=false;
    std::array<VkBuffer,2> inputs{};unsigned input_slot=0;
    ~Plan(){if(ready)deleteVkFFT(&app);}
    void initialize(VkFFTConfiguration& config) {
        const auto result=initializeVkFFT(&app,config);
        if(result!=VKFFT_SUCCESS)throw std::runtime_error("Spectral retiming FFT initialization failed: "+std::to_string(result));
        ready=true;
    }
    void append(vk::CommandBuffer command,int direction,std::uint64_t input_offset=0,std::uint64_t buffer_offset=0,VkBuffer input={}) {
        VkCommandBuffer raw=command;VkFFTLaunchParams launch{};launch.commandBuffer=&raw;launch.inputBufferOffset=input_offset;launch.bufferOffset=buffer_offset;
        // VkFFT detects replacement by POINTER identity, not handle value.
        // Alternate persistent slots so descriptors update even when successive
        // BufferViews arrive at the same caller stack address.
        if(input){input_slot^=1;inputs[input_slot]=input;launch.inputBuffer=&inputs[input_slot];}
        const auto result=VkFFTAppend(&app,direction,&launch);
        if(result!=VKFFT_SUCCESS)throw std::runtime_error("Spectral retiming FFT dispatch failed: "+std::to_string(result));
    }
};
// Absolute phases remain in compensated double precision; only local angles
// enter the shader. This also works for seeks far beyond FP32 integer precision.
C oscillator(std::uint64_t n,double frequency,double fs) {
    const double rate=frequency/fs,product=double(n)*rate;
    const double correction=std::fma(double(n),rate,-product)+double(n)*std::fma(-rate,fs,frequency)/fs;
    return static_cast<C>(std::polar(1.0,-tau*(std::remainder(product,1.0)+correction)));
}
float integer_bits(std::int64_t value) {
    if(value<INT32_MIN || value>INT32_MAX)throw std::runtime_error("Retiming input coordinate exceeds buffer index range");
    return std::bit_cast<float>(std::int32_t(value));
}
}
struct SpectralClock::Impl {
    static constexpr unsigned blocks=800,inverse_size=1024,stride=262144,response_size=65536;
    unsigned rate,source_rate,decimation,fft_size,guard,hop,raw_stride;
    double source_ratio;
    bool touches_nyquist;
    std::vector<double> centres;
    ChannelClockInterpolator narrow;
    ClockInterpolator raw;
    gpu::Kernel kernel;
    vk::UniquePipeline synth;
    VkDevice device=kernel.device();VkPhysicalDevice physical=kernel.physical();
    VkQueue queue=kernel.queue();VkCommandPool pool=kernel.pool();VkFence fence=kernel.fence();
    VkBuffer forward_buffer{},input_buffer{};
    std::uint64_t forward_bytes{},input_bytes{};
    Plan forward,boundary_fft;
    struct Push {std::uint32_t fft_size,hop,decimation,count,channels,stride,input_count,guard;};
    Impl(std::uint32_t fs,std::span<const double> offsets,std::span<const float> prototype,
         std::shared_ptr<gpu::Context> context,std::uint32_t source)
        : rate(fs),source_rate(source ? source : fs),decimation(fs/8'000'000),fft_size(512*decimation),
          source_ratio(double(source_rate)/fs),
          touches_nyquist(std::ranges::any_of(offsets,[this](double f){return std::abs(f)+3671875>source_rate/2.0;})),
          centres(offsets.begin(),offsets.end()),
          raw((std::abs(*std::max_element(offsets.begin(),offsets.end(),[](double a,double b){return std::abs(a)<std::abs(b);}))+3328125)/source_rate),
          kernel(boundary_code,32,std::array<std::uint32_t,3>{raw.taps,fft_size,blocks},std::move(context)),
          synth(kernel.pipeline(synth_code,std::array<std::uint32_t,5>{fft_size,
              std::bit_cast<std::uint32_t>(float(SmallClockInterpolator::weights[0])),
              std::bit_cast<std::uint32_t>(float(SmallClockInterpolator::weights[1])),
              std::bit_cast<std::uint32_t>(float(SmallClockInterpolator::weights[2])),blocks})) {
        // Both interpolation support and the FIR halo are inside the FFT's
        // discarded edges. An extra eight ADC samples cover the 1000 ppm bound.
        guard=unsigned(((prototype.size()/2+(narrow.taps/2+1)*decimation/2+72+decimation-1)/decimation)*decimation);
        hop=(fft_size-2*guard)/decimation;
        if(hop>=16)hop-=hop%16;
        // Exact raw FFT stride for rational nominal rate conversion whenever
        // a useful block fits; otherwise boundary synthesis handles the residue.
        const auto alignment=std::lcm(16U,8'000'000/std::gcd(source_rate,8'000'000U));
        if(alignment<=hop)hop-=hop%alignment;
        raw_stride=unsigned(std::floor(double(hop)*decimation*source_ratio));
        if(2*guard>=fft_size || !hop)throw std::runtime_error("Insufficient FFT support for spectral clock correction");
        forward_bytes=std::uint64_t(fft_size)*blocks*sizeof(C);
        input_bytes=(std::max<std::uint64_t>(stride,std::uint64_t(blocks)*hop)*decimation+fft_size+8192)*sizeof(C);
        kernel.storage(0,input_bytes);input_buffer=kernel.buffer(0);
        kernel.storage(1,forward_bytes);
        forward_buffer=kernel.buffer(1);
        VkFFTConfiguration config{};config.FFTdim=1;config.size[0]=fft_size;config.numberBatches=blocks;
        config.device=&device;config.physicalDevice=&physical;config.queue=&queue;config.commandPool=&pool;config.fence=&fence;
        config.buffer=&forward_buffer;config.bufferSize=&forward_bytes;config.makeForwardPlanOnly=1;config.useLUT=1;
        config.specifyOffsetsAtLaunch=1;
        config.isInputFormatted=1;config.inputBuffer=&input_buffer;config.inputBufferSize=&input_bytes;
        config.inputBufferStride[0]=raw_stride;
        forward.initialize(config);
        config.isInputFormatted=0;config.numberBatches=touches_nyquist ? blocks : 1;
        boundary_fft.initialize(config);
        // One real, zero-phase prototype response for all arbitrary centres and
        // clock rates. Linear lookup preserves the measured stopband bound.
        Fft fft(response_size);std::vector<C> response(response_size);
        for(std::size_t j=0;j<prototype.size();++j)
            response[(j+response_size-prototype.size()/2)%response_size]=C{prototype[j],0};
        fft.transform(response);
        std::vector<float> table(response_size/2+1);
        for(std::size_t j=0;j<table.size();++j)table[j]=response[j].real();
        for(unsigned j=0;j<inverse_size;++j) {
            const auto value=std::polar(1.0,tau*j/inverse_size);table.push_back(float(value.real()));table.push_back(float(value.imag()));
        }
        kernel.upload<float>(2,table);
        auto coefficients=narrow.coefficients;
        coefficients.insert(coefficients.end(),raw.coefficients.begin(),raw.coefficients.end());
        kernel.upload<float>(6,coefficients);
        kernel.storage(4,std::uint64_t(stride)*centres.size()*sizeof(C));
    }
};
SpectralClock::SpectralClock(std::uint32_t rate,std::span<const double> centres,
    std::span<const float> prototype,std::shared_ptr<gpu::Context> context,std::uint32_t source_rate)
    : impl_(std::make_unique<Impl>(rate,centres,prototype,std::move(context),source_rate)) {}
SpectralClock::~SpectralClock()=default;
C* SpectralClock::input_data(std::size_t count) {
    if(count>impl_->input_bytes/sizeof(C))throw std::runtime_error("Spectral clock input exceeds batch support");
    return static_cast<C*>(impl_->kernel.storage(0,impl_->input_bytes));
}
gpu::BufferView SpectralClock::output_buffer() const {return impl_->kernel.view(4);}
std::size_t SpectralClock::capacity() const {return std::min<std::size_t>(impl_->hop*Impl::blocks,Impl::stride);}
unsigned SpectralClock::support() const {return impl_->guard;}
unsigned SpectralClock::raw_left() const {return impl_->raw.left;}
unsigned SpectralClock::raw_right() const {return impl_->raw.taps-impl_->raw.left;}
unsigned SpectralClock::interpolation_taps() const {return impl_->narrow.taps;}
std::size_t SpectralClock::input_capacity() const {return impl_->input_bytes/sizeof(C);}
void SpectralClock::process(std::span<const C> samples,std::uint64_t input_start,
    std::uint64_t first,std::size_t count,std::span<const ClockSegment> clock,ComputeTimings* timings) {
    auto& p=*impl_;
    if(samples.size()>input_capacity())throw std::runtime_error("Spectral clock input exceeds batch support");
    auto* raw=static_cast<C*>(p.kernel.storage(0,p.input_bytes));
    if(!samples.empty() && raw!=samples.data())std::memcpy(raw,samples.data(),samples.size_bytes());
    std::fill(raw+samples.size(),raw+input_capacity(),C{});
    process(p.kernel.view(0),samples.size(),input_start,first,count,clock,timings);
}
void SpectralClock::process(const gpu::BufferView& samples,std::size_t valid_samples,std::uint64_t input_start,
    std::uint64_t first,std::size_t count,std::span<const ClockSegment> clock,ComputeTimings* timings) {
    auto& p=*impl_;
    if(clock.empty() || first<p.guard+8 || !count || count>capacity() || valid_samples>input_capacity() ||
       samples.context!=p.kernel.context() || samples.bytes<p.input_bytes)
        throw std::runtime_error("Invalid spectral clock coordinates or resident input support");
    p.kernel.bind(0,samples);
    auto segment_at=[&](std::uint64_t n) {
        auto it=std::upper_bound(clock.begin(),clock.end(),n,[](auto value,const ClockSegment& s){return value<s.start;});
        return it==clock.begin() ? it : it-1;
    };
    auto source_at=[&](std::uint64_t n) {
        // Exact integer quotient/remainder before subtracting the buffer epoch:
        // nominal rational conversion must not discard fractional samples at a
        // large recording index, or introduce differential RF phase errors.
        const auto product=(n%p.rate)*std::uint64_t(p.source_rate);
        const auto integer=n/p.rate*p.source_rate+product/p.rate;
        return double(std::int64_t(integer)-std::int64_t(input_start))+
            double(product%p.rate)/p.rate+segment_at(n)->offset_at(double(n))*p.source_ratio;
    };
    if(source_at(first-p.guard)<p.raw.left ||
       std::floor(source_at(first+(count-1)*p.decimation+p.guard))+p.raw.taps-p.raw.left>double(valid_samples))
        throw std::runtime_error("Raw ADC support missing for spectral clock correction");
    std::vector<std::array<float,4>> metadata(Impl::blocks+p.centres.size()+Impl::blocks*p.centres.size());
    std::vector<std::array<float,4>> coordinates,boundary_blocks;
    const auto active=(count+p.hop-1)/p.hop;
    const auto middle=unsigned(active/2);
    const auto centre=first+std::uint64_t(middle)*p.hop*p.decimation;
    const auto unaligned_base=std::max<std::int64_t>(0,std::int64_t(std::floor(source_at(centre)))-
        p.guard-std::int64_t(middle)*p.raw_stride);
    const auto input_base=(unaligned_base+8)/16*16;
    const auto padded=std::size_t(input_base)+(Impl::blocks-1)*p.raw_stride+p.fft_size;
    if(padded*sizeof(C)>p.input_bytes)throw std::runtime_error("Spectral clock FFT input exceeds batch support");
    for(unsigned c=0;c<p.centres.size();++c) {
        const auto bin=std::int32_t(std::llround(p.centres[c]*p.fft_size/p.source_rate));
        const auto corrected_bin=std::int32_t(std::llround(p.centres[c]*p.fft_size/p.rate));
        metadata[Impl::blocks+c]={std::bit_cast<float>(bin),float(p.centres[c]*p.fft_size/p.rate),std::bit_cast<float>(corrected_bin),0};
    }
    for(unsigned b=0;b<active;++b) {
        const auto n=first+std::uint64_t(b)*p.hop*p.decimation;
        const auto begin=n-p.guard;
        const auto segment=segment_at(n);
        // Only windows containing a rate change need full-band interpolation.
        // The conservative interval includes fractional input support and drift.
        // A prototype transition cut by ADC Nyquist is not a band-limited
        // channel field. Preserve the original full-band interpolation there.
        bool affine=!p.touches_nyquist && segment_at(begin-8)==segment_at(begin+p.fft_size+8);
        const auto input_begin=input_base+std::int64_t(b)*p.raw_stride;
        const double shift=source_at(n)-double(input_begin)-p.guard;
        const double first_at=2.0*(p.guard+shift)/p.decimation;
        const double rate=segment->rate*p.source_ratio;
        const auto used=std::min<std::size_t>(p.hop,count-b*p.hop);
        const double last_at=first_at+2*(used-1)*rate;
        const double halo=double(p.guard-72);
        affine=affine && first_at*p.decimation/2>=halo && last_at*p.decimation/2<=p.fft_size-halo;
        std::uint32_t coordinate=UINT32_MAX;
        if(!affine) {
            boundary_blocks.push_back({std::bit_cast<float>(std::uint32_t(b)),0,0,0});
            coordinate=std::uint32_t(coordinates.size());
            for(unsigned j=0;j<p.fft_size;j+=64) {
                const auto t=begin+j;const auto s=segment_at(t);
                const auto next=s+1;
                const double position=source_at(t);
                const auto integer=std::int64_t(std::floor(position));
                const double split=next!=clock.end() && next->start<t+64 ? double(next->start-t) : 64;
                const double after=(split<64 ? next->rate : s->rate)*p.source_ratio;
                coordinates.push_back({integer_bits(integer),float(position-std::floor(position)),float(s->rate*p.source_ratio-1),float(after-1)});
                coordinates.push_back({float(split),0,0,0});
            }
        }
        const double at=affine ? first_at : 2.0*p.guard/p.decimation;
        const double applied_rate=affine ? rate : 1;
        const double midpoint=.5*(used-1);
        const double fractional=at-std::floor(at)+2*midpoint*(applied_rate-1);
        metadata[b]={integer_bits(std::int64_t(std::floor(at))+std::int64_t(std::floor(fractional))),
            float(fractional-std::floor(fractional)),float(applied_rate-1),std::bit_cast<float>(coordinate)};
        for(unsigned c=0;c<p.centres.size();++c) {
            const double bin=std::llround(p.centres[c]*p.fft_size/(affine ? p.source_rate : p.rate));
            const auto anchor=oscillator(n,p.centres[c],p.rate)*static_cast<C>(std::polar(1.0,
                tau*bin*(affine ? p.guard+shift : p.guard)/p.fft_size));
            const double step=tau*(bin*applied_rate/512-p.centres[c]/8e6);
            metadata[Impl::blocks+p.centres.size()+c*Impl::blocks+b]={anchor.real(),anchor.imag(),float(step),float(midpoint)};
        }
    }
    // Inactive FFTs are zero-filled, never dependent on the preceding batch.
    for(unsigned b=unsigned(active);b<Impl::blocks;++b)metadata[b][3]=std::bit_cast<float>(UINT32_MAX);
    metadata.insert(metadata.end(),boundary_blocks.begin(),boundary_blocks.end());
    p.kernel.upload<std::array<float,4>>(5,metadata);
    if(coordinates.empty())coordinates.resize(1);
    p.kernel.upload<std::array<float,4>>(7,coordinates);
    if(p.touches_nyquist) {
        auto* work=static_cast<C*>(p.kernel.storage(1,p.forward_bytes));
        std::fill(work+active*p.fft_size,work+Impl::blocks*p.fft_size,C{});
    }
    const Impl::Push push{p.fft_size,p.hop,p.decimation,std::uint32_t(count),std::uint32_t(p.centres.size()),Impl::stride,std::uint32_t(valid_samples),p.guard};
    const bool detailed=timings && timings->detailed;
    const auto cmd=p.kernel.begin(timings,detailed ? 6 : 2);
    if(!p.touches_nyquist) {
        p.forward.append(cmd,-1,std::uint64_t(input_base)*sizeof(C),0,samples.buffer);p.kernel.barrier(cmd);
    }
    if(detailed)p.kernel.stamp(cmd,1);
    if(!boundary_blocks.empty()) {
        p.kernel.dispatch(cmd,p.kernel.primary(),push,boundary_blocks.size()*p.fft_size);p.kernel.barrier(cmd);
        if(p.touches_nyquist)p.boundary_fft.append(cmd,-1);
        else for(const auto& record:boundary_blocks) {
                const auto block=std::bit_cast<std::uint32_t>(record[0]);
                p.boundary_fft.append(cmd,-1,0,std::uint64_t(block)*p.fft_size*sizeof(C));
            }
        p.kernel.barrier(cmd);
    }
    if(detailed)p.kernel.stamp(cmd,2);
    p.kernel.groups(cmd,*p.synth,push,Impl::blocks,unsigned(p.centres.size()));
    if(detailed) {p.kernel.stamp(cmd,3);p.kernel.stamp(cmd,4);}
    p.kernel.finish(cmd);
}
}
