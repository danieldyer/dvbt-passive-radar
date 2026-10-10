#include "gpu_frontend.hpp"
#include "gpu_compute.hpp"
#include "fractional_resampler.hpp"
#include "channel_filters.hpp"
#include "spectral_clock.hpp"
#include <vkFFT.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <numbers>

namespace radar {
namespace {
constexpr double tau=2*std::numbers::pi;
constexpr std::uint32_t fold_spirv[] =
#include "channel_fold.inc"
;
constexpr std::uint32_t unpack_spirv[] =
#include "channel_unpack.inc"
;
constexpr std::uint32_t resample_spirv[] =
#include "resample.inc"
;
using gpu::Kernel;
// Compensated product/division keeps absolute NCO phase accurate at large file
// indices. Do not form sin(float(absolute_sample)*float(frequency)).
C oscillator(std::uint64_t n,double frequency,double fs) {
    const double rate=frequency/fs, product=static_cast<double>(n)*rate;
    const double correction=std::fma(static_cast<double>(n),rate,-product)+
        static_cast<double>(n)*(std::fma(-rate,fs,frequency)/fs);
    return static_cast<C>(std::polar(1.0,-tau*(std::remainder(product,1.0)+correction)));
}
}
struct GpuChannelBank::Impl {
    // Keep the common 7:1 forward FFT within one workgroup's shared memory.
    static constexpr std::size_t inverse_size=512,batches=512;
    static constexpr std::size_t oscillator_size=256+batch_size/256;
    WaveReader* input=nullptr;
    std::uint32_t input_rate;
    std::vector<double> offsets;
    std::vector<float> prototype;
    std::size_t fft_size,hop;
    std::shared_ptr<SampleClock> clock;
    std::unique_ptr<SpectralClock> retiming;
    bool download=true;
    std::size_t valid_count=0;
    Kernel kernel;
    vk::UniquePipeline unpack;
    VkDevice device=kernel.device();VkPhysicalDevice physical=kernel.physical();
    VkQueue queue=kernel.queue();VkCommandPool pool=kernel.pool();VkFence fence=kernel.fence();
    VkBuffer forward_buffer{},inverse_buffer{},input_buffer{};
    std::uint64_t forward_bytes{},inverse_bytes{},input_bytes{};
    struct Plan {
        VkFFTApplication app{};bool ready=false;
        ~Plan(){if(ready)deleteVkFFT(&app);}
        void initialize(VkFFTConfiguration& config) {
            const auto result=initializeVkFFT(&app,config);
            if(result!=VKFFT_SUCCESS)throw std::runtime_error("Channel bank VkFFT initialization failed: "+std::to_string(result));
            ready=true;
        }
        void append(vk::CommandBuffer command,int direction) {
            VkCommandBuffer raw=command;VkFFTLaunchParams launch{};launch.commandBuffer=&raw;
            const auto result=VkFFTAppend(&app,direction,&launch);
            if(result!=VKFFT_SUCCESS)throw std::runtime_error("Channel bank VkFFT dispatch failed: "+std::to_string(result));
        }
    } forward,inverse;
    Impl(std::uint32_t rate,std::span<const double> centres) : input_rate(rate),offsets(centres.begin(),centres.end()),
        prototype(GpuChannelBank::prototype(rate)),fft_size(inverse_size*(rate/8'000'000)),
        hop((fft_size-prototype.size()+1)/(rate/8'000'000)),
        kernel(fold_spirv,32,std::array<std::uint32_t,3>{rate/8'000'000,inverse_size,static_cast<std::uint32_t>(centres.size())}),
        unpack(kernel.pipeline(unpack_spirv,std::array<std::uint32_t,3>{rate/8'000'000,inverse_size,static_cast<std::uint32_t>(centres.size())})) {
        const auto decimation=rate/8'000'000;
        input_bytes=((batch_size-1)*decimation+prototype.size())*sizeof(C);
        forward_bytes=fft_size*batches*sizeof(C);inverse_bytes=inverse_size*batches*centres.size()*sizeof(C);
    }
    void initialize() {
        if(forward.ready)return;
        kernel.storage(0,input_bytes);input_buffer=kernel.buffer(0);
        kernel.storage(1,forward_bytes);kernel.storage(3,inverse_bytes);
        forward_buffer=kernel.buffer(1);inverse_buffer=kernel.buffer(3);
        VkFFTConfiguration config{};config.FFTdim=1;config.size[0]=fft_size;config.numberBatches=batches;
        config.device=&device;config.physicalDevice=&physical;config.queue=&queue;config.commandPool=&pool;config.fence=&fence;
        config.buffer=&forward_buffer;config.bufferSize=&forward_bytes;config.makeForwardPlanOnly=1;
        config.useLUT=1;
        // Each FFT reads its overlapping window directly from the shared raw
        // buffer. VkFFT's formatted input stride eliminates a packing pass.
        config.isInputFormatted=1;config.inputBuffer=&input_buffer;config.inputBufferSize=&input_bytes;
        config.inputBufferStride[0]=hop*(input_rate/8'000'000);
        forward.initialize(config);
        config.size[0]=inverse_size;config.numberBatches=batches*offsets.size();config.buffer=&inverse_buffer;config.bufferSize=&inverse_bytes;
        config.makeForwardPlanOnly=0;config.makeInversePlanOnly=1;config.normalize=1;
        config.isInputFormatted=0;config.inputBuffer=nullptr;config.inputBufferSize=nullptr;
        inverse.initialize(config);
    }
};
std::vector<float> GpuChannelBank::prototype(double rate) {
    if(!std::isfinite(rate) || rate<8e6 || rate>128e6 || std::fmod(rate,8e6)!=0)
        throw std::runtime_error("Prototype requires an integer 8 MS/s input multiple, up to 128 MS/s");
    // Less common high input rates use the shared conservative Kaiser designer.
    if(rate>64e6)return Channelizer::prototype(rate,3.5e6,1024);
    const auto taps=filters::prototype(static_cast<unsigned>(rate/8e6));
    return {taps.begin(),taps.end()};
}
GpuChannelBank::GpuChannelBank(std::uint32_t input_rate,std::span<const double> offsets) {
    const double fs=input_rate;
    if(input_rate%8'000'000 || fs<8e6 || fs>128e6 || offsets.empty() || offsets.size()>64)
        throw std::runtime_error("Invalid GPU channel bank: use 1..64 channels, integer 8 MS/s input multiples up to 128 MS/s, and a positive interval");
    for(double f:offsets) if(!std::isfinite(f) || std::abs(f)+3.5e6>fs/2)
        throw std::runtime_error("Channel extends beyond input Nyquist interval");
    impl_=std::make_unique<Impl>(input_rate,offsets);
    auto& p=*impl_; const auto half=p.prototype.size()/2;
    decimation=input_rate/8'000'000;
    std::vector<C> coefficients(offsets.size()*(p.fft_size+p.oscillator_size+1)),response(p.fft_size);
    Fft fft(p.fft_size);
    for(std::size_t c=0;c<offsets.size();++c) {
        std::fill(response.begin(),response.end(),C{});
        for(std::size_t j=0;j<p.prototype.size();++j)
            response[j]=static_cast<C>(static_cast<double>(p.prototype[j])*
                std::polar(1.0,tau*offsets[c]*(static_cast<double>(j)-half)/fs));
        fft.transform(response);
        for(std::size_t k=0;k<p.fft_size;++k)
            coefficients[c*p.fft_size+k]=response[k]*static_cast<C>(std::polar(1.0/decimation,
                tau*k*((p.prototype.size()-1)%decimation)/p.fft_size));
        // Two small phase tables replace one full output-sized NCO array.
        // Their product has no recurrence drift and stays in the GPU cache.
        for(std::size_t j=0;j<p.oscillator_size;++j)
            coefficients[offsets.size()*p.fft_size+c*p.oscillator_size+j]=
                oscillator((j<256 ? j : (j-256)*256)*decimation,offsets[c],fs);
    }
    p.kernel.upload<C>(2,coefficients);
}
GpuChannelBank::GpuChannelBank(WaveReader& wave,std::span<const double> offsets,double start,double seconds)
    : GpuChannelBank(wave.info.sample_rate,offsets) {
    if(!std::isfinite(start) || start<0 || !std::isfinite(seconds) || seconds<=0 || wave.info.samples>(std::uint64_t{1}<<53))
        throw std::runtime_error("Invalid channel bank input interval");
    impl_->input=&wave;const auto half=taps()/2;const double fs=wave.info.sample_rate;
    if(start*fs>=static_cast<double>(wave.info.samples))throw std::runtime_error("Start is beyond recording");
    origin=std::max<std::uint64_t>(half,static_cast<std::uint64_t>(start*fs));
    origin=(origin+decimation-1)/decimation*decimation;
    if(origin+half>=wave.info.samples)throw std::runtime_error("Recording shorter than FIR support");
    output_count=std::min<std::uint64_t>((wave.info.samples-half-1-origin)/decimation+1,
        static_cast<std::uint64_t>(std::min(seconds*8e6,static_cast<double>(wave.info.samples))));
}
GpuChannelBank::~GpuChannelBank()=default;
std::size_t GpuChannelBank::taps() const {return impl_->prototype.size();}
std::size_t GpuChannelBank::capacity() const {return impl_->retiming ? impl_->retiming->capacity() : impl_->hop*impl_->batches;}
unsigned GpuChannelBank::interpolation_taps() const {return impl_->retiming ? impl_->retiming->interpolation_taps() : 0;}
void GpuChannelBank::set_clock(std::shared_ptr<SampleClock> clock) {
    auto& p=*impl_;
    if(produced || p.valid_count || p.clock || !clock)throw std::runtime_error("Attach the common clock before processing");
    p.retiming=std::make_unique<SpectralClock>(p.input_rate,p.offsets,p.prototype,p.kernel.context());
    p.clock=std::move(clock);
    if(p.input) {
        const auto old=origin;
        origin=std::max<std::uint64_t>(origin,p.retiming->support()+p.retiming->raw_left()+8);
        origin=(origin+decimation-1)/decimation*decimation;
        output_count-=std::min(output_count,(origin-old+decimation-1)/decimation);
    }
}
std::vector<std::vector<C>> GpuChannelBank::next(FrontendTimings* timings) {
    auto& p=*impl_;
    p.valid_count=0;
    if(!p.input)throw std::runtime_error("No WAV source attached; use process() for in-memory samples");
    auto count=static_cast<std::size_t>(std::min<std::uint64_t>(capacity(),output_count-produced));
    if(!count)return {};
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    auto measure=[&](double FrontendTimings::* member){if(timings){const auto now=Clock::now();timings->*member+=std::chrono::duration<double>(now-mark).count();mark=now;}};
    const auto centre=origin+produced*decimation;
    if(p.clock) {
        // Previous output used the right FIR halo. Activate the next immutable
        // segment only beyond that halo, so overlapping windows see one warp.
        const auto support=p.retiming->support();
        const auto first=centre-support;
        p.clock->update(centre+((support+63)/64)*64);
        auto original=[&](std::uint64_t n) {const auto s=p.clock->at(double(n));return std::int64_t(n)+std::int64_t(std::floor(s.offset_at(double(n))));};
        const auto begin=original(first)-p.retiming->raw_left();
        if(begin<0)throw std::runtime_error("Clock warp starts before available ADC support");
        std::int64_t end=0;
        while(count) {
            end=original(centre+(count-1)*decimation+support)+p.retiming->raw_right();
            if(end<=std::int64_t(p.input->info.samples))break;
            const auto missing=std::size_t(end-p.input->info.samples);
            count-=std::min(count,(missing+decimation-1)/decimation+1);
        }
        if(!count) {produced=output_count;return {};}
        auto* raw=p.retiming->input_data(std::size_t(end-begin));
        p.input->read(begin,{raw,std::size_t(end-begin)});measure(&FrontendTimings::read);
        auto result=process_clocked({raw,std::size_t(end-begin)},begin,centre,count,timings);
        produced+=count;return result;
    }
    const auto required=(count-1)*decimation+p.prototype.size();
    auto* input=static_cast<C*>(p.kernel.storage(0,((batch_size-1)*decimation+p.prototype.size())*sizeof(C)));
    p.input->read(origin+produced*decimation-p.prototype.size()/2,{input,required});
    measure(&FrontendTimings::read);
    auto result=process({input,required},origin+produced*decimation,count,timings);
    produced+=count;return result;
}
ChannelBatch GpuChannelBank::next_device(FrontendTimings* timings) {
    auto& p=*impl_;const auto first=produced;
    struct Restore {bool& flag;~Restore(){flag=true;}} restore{p.download};
    p.download=false;next(timings);
    if(!p.valid_count)return {};
    return {output_buffer(),first,p.valid_count,p.offsets.size(),batch_size};
}
gpu::BufferView GpuChannelBank::output_buffer() const {return impl_->retiming ? impl_->retiming->output_buffer() : impl_->kernel.view(4);}
std::shared_ptr<gpu::Context> GpuChannelBank::context() const {return impl_->kernel.context();}
ChannelBatch GpuChannelBank::process_clocked_device(std::span<const C> samples,std::uint64_t input_start,
    std::uint64_t first_center,std::size_t count,FrontendTimings* timings) {
    if(first_center<origin || (first_center-origin)%decimation)throw std::invalid_argument("Unaligned corrected channel origin");
    auto& p=*impl_;
    struct Restore {bool& flag;bool value;~Restore(){flag=value;}} restore{p.download,p.download};
    p.download=false;process_clocked(samples,input_start,first_center,count,timings);
    return {output_buffer(),(first_center-origin)/decimation,count,p.offsets.size(),batch_size};
}
std::size_t GpuChannelBank::input_capacity() const {
    if(!impl_->retiming)throw std::logic_error("Attach the common clock before requesting resident input capacity");
    return impl_->retiming->input_capacity();
}
ChannelBatch GpuChannelBank::process_clocked_device(const gpu::BufferView& samples,std::size_t valid_samples,
    std::uint64_t input_start,std::uint64_t first_center,std::size_t count,FrontendTimings* timings) {
    auto& p=*impl_;
    if(!p.clock || !count || count>capacity() || first_center<origin || (first_center-origin)%decimation ||
       first_center<taps()/2 || first_center>(std::uint64_t{1}<<53)-(count-1)*decimation)
        throw std::invalid_argument("Invalid resident common-clock channel coordinates");
    using Timer=std::chrono::steady_clock;
    const auto mark=timings ? Timer::now() : Timer::time_point{};
    p.retiming->process(samples,valid_samples,input_start,first_center,count,p.clock->segments(),timings);p.valid_count=count;
    if(timings)timings->dispatch_wait+=std::chrono::duration<double>(Timer::now()-mark).count();
    return {output_buffer(),(first_center-origin)/decimation,count,p.offsets.size(),batch_size};
}
std::vector<std::vector<C>> GpuChannelBank::process_clocked(std::span<const C> samples,std::uint64_t input_start,
    std::uint64_t first_center,std::size_t count,FrontendTimings* timings) {
    auto& p=*impl_;
    if(!p.clock || !count || count>capacity() || first_center<taps()/2 || first_center>(std::uint64_t{1}<<53)-(count-1)*decimation)
        throw std::runtime_error("Invalid common-clock channel bank coordinates");
    using Timer=std::chrono::steady_clock;
    const auto mark=timings ? Timer::now() : Timer::time_point{};
    p.retiming->process(samples,input_start,first_center,count,p.clock->segments(),timings);
    p.valid_count=count;
    if(timings)timings->dispatch_wait+=std::chrono::duration<double>(Timer::now()-mark).count();
    if(!p.download)return {};
    const auto* output=static_cast<const C*>(p.retiming->output_buffer().mapped);
    std::vector<std::vector<C>> result(p.offsets.size());
    for(std::size_t c=0;c<result.size();++c)result[c].assign(output+c*batch_size,output+c*batch_size+count);
    return result;
}
std::vector<std::vector<C>> GpuChannelBank::process(std::span<const C> samples,std::uint64_t first_center,std::size_t count,FrontendTimings* timings) {
    auto& p=*impl_;
    if(p.clock)throw std::runtime_error("Use process_clocked() with an attached sample clock");
    p.initialize();
    if(!count || count>capacity() || samples.size()<(count-1)*decimation+taps() || first_center<taps()/2 ||
       first_center>(std::uint64_t{1}<<53)-(count-1)*decimation)
        throw std::runtime_error("Invalid channel bank input support or coordinates");
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    auto measure=[&](double FrontendTimings::* member){if(timings){const auto now=Clock::now();timings->*member+=std::chrono::duration<double>(now-mark).count();mark=now;}};
    auto* input=static_cast<C*>(p.kernel.storage(0,((batch_size-1)*decimation+taps())*sizeof(C)));
    if(input!=samples.data())std::memcpy(input,samples.data(),((count-1)*decimation+taps())*sizeof(C));
    std::fill(input+(count-1)*decimation+taps(),input+(p.batches-1)*p.hop*decimation+p.fft_size,C{});
    auto* coefficients=static_cast<C*>(p.kernel.storage(2,p.offsets.size()*(p.fft_size+p.oscillator_size+1)*sizeof(C)));
    for(std::size_t c=0;c<p.offsets.size();++c)
        coefficients[p.offsets.size()*(p.fft_size+p.oscillator_size)+c]=oscillator(first_center,p.offsets[c],p.input_rate);
    auto* output=static_cast<C*>(p.kernel.storage(4,p.offsets.size()*batch_size*sizeof(C)));
    struct Push { std::uint32_t fft_size,hop,decimation,count,channels,stride,taps,batches; };
    const Push push{static_cast<std::uint32_t>(p.fft_size),static_cast<std::uint32_t>(p.hop),decimation,
        static_cast<std::uint32_t>(count),static_cast<std::uint32_t>(p.offsets.size()),batch_size,static_cast<std::uint32_t>(p.prototype.size()),p.batches};
    const bool detailed=timings && timings->detailed;
    const auto cmd=p.kernel.begin(timings,detailed ? 6 : 2);
    if(detailed)p.kernel.stamp(cmd,1);
    p.forward.append(cmd,-1);p.kernel.barrier(cmd);
    if(detailed)p.kernel.stamp(cmd,2);
    p.kernel.dispatch(cmd,p.kernel.primary(),push,p.inverse_size*p.batches);p.kernel.barrier(cmd);
    if(detailed)p.kernel.stamp(cmd,3);
    p.inverse.append(cmd,1);p.kernel.barrier(cmd);
    if(detailed)p.kernel.stamp(cmd,4);
    p.kernel.dispatch(cmd,*p.unpack,push,p.batches*64,p.offsets.size());p.kernel.finish(cmd);
    measure(&FrontendTimings::dispatch_wait);
    p.valid_count=count;
    if(!p.download)return {};
    std::vector<std::vector<C>> result(p.offsets.size());
    for(std::size_t c=0;c<result.size();++c)result[c].assign(output+c*batch_size,output+c*batch_size+count);
    measure(&FrontendTimings::readback);
    return result;
}
struct GpuResampler::Impl {
    Kernel kernel{resample_spirv,8};
    Impl() {const FractionalResampler filter;kernel.upload<float>(1,filter.coefficients());}
};
GpuResampler::GpuResampler() : impl_(std::make_unique<Impl>()) {}
GpuResampler::~GpuResampler()=default;
std::vector<C> GpuResampler::process(std::span<const C> input,double first,double step,std::size_t count,FrontendTimings* timings) {
    if(!count || count>1<<20 || input.size()>INT32_MAX || !std::isfinite(first) || !std::isfinite(step) ||
        std::abs(step-1)>0.001 || first<47 || std::floor(first+(count-1)*step)+48>=static_cast<double>(input.size()))
        throw std::runtime_error("Invalid GPU resampling coordinates/support (maximum clock error 1000 ppm)");
    auto& kernel=impl_->kernel;
    kernel.upload<C>(0,input);
    std::vector<C> anchors((count+63)/64);
    for(std::size_t g=0;g<anchors.size();++g) {
        const double t=first+(g*64)*(step-1), integral=std::floor(t);
        anchors[g]={std::bit_cast<float>(static_cast<std::int32_t>(integral)),static_cast<float>(t-integral)};
    }
    kernel.upload<C>(2,anchors);
    auto* output=static_cast<C*>(kernel.storage(3,count*sizeof(C)));
    struct Push { std::uint32_t count;float delta; };
    kernel.run(Push{static_cast<std::uint32_t>(count),static_cast<float>(step-1)},count,timings);
    return {output,output+count};
}
}
