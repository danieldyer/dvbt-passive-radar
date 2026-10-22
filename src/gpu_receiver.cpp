#include "gpu_receiver.hpp"
#include "gpu_compute.hpp"
#include "pilot_phase.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>

namespace radar {
namespace {
constexpr std::uint32_t copy_code[] =
#include "stream_copy.inc"
;
constexpr double fs=8e6,tau=2*std::numbers::pi;
unsigned batch_capacity(std::size_t samples,std::span<const unsigned> guards) {
    std::size_t n=0;
    for(auto guard:guards){
        if(guard!=4 && guard!=8 && guard!=16 && guard!=32)throw std::invalid_argument("Invalid guard interval");
        // Acquisition retains at most three complete symbols. Locked tracking
        // retains less; count the actual channel guards rather than padding
        // every transform for eight extra shortest-guard symbols.
        n+=samples/(8192+8192/guard)+3;
    }
    if(!samples || guards.empty() || n>4096)throw std::invalid_argument("Invalid streaming OFDM dimensions");
    return static_cast<unsigned>(n);
}
}
struct GpuReceiver::Impl {
    struct Channel {
        unsigned guard{},segment=0,age=0,last_mod=0;
        bool locked=false,previous=false,announced=false;
        std::uint64_t start=0,next_cp=0,phase_sample=0,previous_sample=0;
        std::size_t count=0,keep_offset=0,keep_count=0;
        double phase=0,frequency=0,estimated_frequency=0,previous_centre_phase=0;
        std::vector<C> previous_pilots,anchor;
        std::vector<double> pilot_precision;
        std::uint64_t acquisition_sample{};
        ReceiverStatus status;
        double timing_square{},phase_square{};
        double anchor_slope{};
        std::int64_t window_shift=0;
        double source_rate_error=0;
        unsigned length() const {return 8192+8192/guard;}
    };
    std::size_t max_samples,stride;
    gpu::Kernel copy;
    GpuOfdm ofdm;
    std::array<gpu::BufferView,2> buffers;
    unsigned current=0;
    std::uint64_t expected=0;
    bool have_input=false;
    std::vector<Channel> channels;
    PilotPattern pattern;
    std::vector<TimingObservation> observations;
    Impl(const PilotPattern& pilots,std::span<const unsigned> guards,std::size_t maximum,
         std::shared_ptr<gpu::Context> context):max_samples(maximum),stride(maximum+4*10240),
        copy(copy_code,16,{},std::move(context)),ofdm(pilots,batch_capacity(maximum,guards),copy.context()),
        pattern(pilots) {
        if(stride*guards.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("Invalid streaming OFDM sample coordinates");
        for(unsigned guard:guards) {
            if(guard!=4 && guard!=8 && guard!=16 && guard!=32)throw std::invalid_argument("Invalid guard interval");
            Channel channel;channel.guard=guard;channels.push_back(std::move(channel));
        }
        copy.storage(2,stride*channels.size()*sizeof(C));buffers[0]=copy.view(2);
        copy.storage(4,stride*channels.size()*sizeof(C));buffers[1]=copy.view(4);
    }
    void reset(unsigned c) {
        auto& s=channels[c];s.locked=false;s.previous=false;s.announced=false;s.age=0;
        s.previous_pilots.clear();s.anchor.clear();++s.segment;
        s.pilot_precision.clear();
        s.window_shift=0;s.source_rate_error=0;
        observations.push_back({TimingObservation::Kind::reset,c,0,0});
        ++s.status.reacquisitions;s.status.ready=false;
    }
    gpu::BufferView append(const ChannelBatch& input,ComputeTimings* timing,gpu::Commands& commands) {
        if(input.channels!=channels.size() || input.count>max_samples || input.stride<input.count ||
           (input.channels-1)*input.stride+input.count>input.samples.bytes/sizeof(C) ||
           input.first>std::numeric_limits<std::uint64_t>::max()-input.count)
            throw std::invalid_argument("Invalid channel batch for streaming OFDM");
        std::vector<std::array<std::uint32_t,4>> regions(channels.size());
        std::size_t count=0;
        for(unsigned c=0;c<channels.size();++c) {
            auto& s=channels[c];
            if(have_input && input.first!=expected){reset(c);s.keep_count=0;}
            const auto previous_offset=c*stride+s.keep_offset;
            s.start=s.keep_count ? s.start+s.keep_offset : input.first;
            s.count=s.keep_count+input.count;
            if(s.count>stride)throw std::logic_error("OFDM history exceeds streaming capacity");
            regions[c]={static_cast<std::uint32_t>(c*input.stride),static_cast<std::uint32_t>(input.count),
                static_cast<std::uint32_t>(previous_offset),static_cast<std::uint32_t>(s.keep_count)};
            count=std::max(count,s.count);
        }
        copy.bind(0,input.samples);copy.bind(1,buffers[current]);current^=1;copy.bind(2,buffers[current]);
        copy.upload(3,std::span<const std::array<std::uint32_t,4>>{regions});
        const auto cmd=copy.begin(timing,2,&commands);
        copy.dispatch(cmd,copy.primary(),std::array{static_cast<unsigned>(stride),static_cast<unsigned>(count),
            static_cast<unsigned>(channels.size()),0U},count,channels.size());
        copy.finish(cmd);have_input=true;expected=input.first+input.count;
        return buffers[current];
    }
    void acquire(const gpu::BufferView& input,ComputeTimings* timing) {
        std::vector<unsigned> candidates;
        std::vector<OfdmWindow> probes;
        for(unsigned c=0;c<channels.size();++c) {
            auto& s=channels[c];const unsigned length=s.length();
            if(s.locked || s.count<2*length)continue;
            const auto base=static_cast<std::uint32_t>(c*stride);
            std::vector<std::uint32_t> positions;
            for(unsigned i=0;i<=length;i+=4)positions.push_back(base+i);
            const auto scores=ofdm.correlate(input,positions,s.guard,timing);
            const auto best=std::max_element(scores.begin(),scores.end(),[](auto a,auto b){return a.quality<b.quality;});
            if(best->quality<.3F)continue;
            const unsigned peak=positions[best-scores.begin()]-base;
            positions.clear();
            for(unsigned i=peak>4 ? peak-4 : 0;i<=std::min<unsigned>(peak+4,s.count-length);++i)positions.push_back(base+i);
            const auto refined=ofdm.correlate(input,positions,s.guard,timing);
            const auto match=std::max_element(refined.begin(),refined.end(),[](auto a,auto b){return a.quality<b.quality;});
            s.next_cp=s.start+positions[match-refined.begin()]-base;
            s.frequency=std::arg(match->correlation)*fs/(tau*8192);
            s.phase_sample=s.next_cp+8192/s.guard-fft_window_margin;s.phase=0;
            probes.push_back({base+static_cast<std::uint32_t>(s.phase_sample-s.start),s.phase_sample,0,s.frequency});
            candidates.push_back(c);
        }
        if(probes.empty())return;
        ofdm.transform(input,probes,timing);
        const auto frequencies=ofdm.integer_frequency(128,timing);
        for(unsigned j=0;j<candidates.size();++j) {
            auto& s=channels[candidates[j]];
            if(frequencies[j].quality<.3F)continue;
            s.frequency+=frequencies[j].bins*fs/8192;s.estimated_frequency=s.frequency;s.locked=true;s.acquisition_sample=s.phase_sample;
        }
    }
    ReceiverBatch process(const ChannelBatch& input,ComputeTimings* timing,gpu::Commands* output_commands) {
        if(!input.count)return {};
        observations.clear();gpu::Commands commands(copy.context());
        const auto samples=append(input,timing,commands);
        if(std::any_of(channels.begin(),channels.end(),[](const Channel& c){return !c.locked;}))commands.wait();
        acquire(samples,timing);
        ReceiverBatch result;std::vector<OfdmWindow> windows;
        for(unsigned c=0;c<channels.size();++c) {
            auto& s=channels[c];
            if(!s.locked)continue;
            while(s.next_cp+s.length()<=s.start+s.count) {
                const auto at=s.next_cp+8192/s.guard-fft_window_margin;
                const double phase=s.phase-tau*s.frequency*(double(at)-double(s.phase_sample))/fs;
                OfdmWindow window{static_cast<std::uint32_t>(c*stride+at-s.start),at,phase,s.frequency};
                windows.push_back(window);result.symbols.push_back({window,c,s.segment});
                s.next_cp+=s.length();
            }
            const auto next=s.next_cp+8192/s.guard-fft_window_margin;
            s.phase-=tau*s.frequency*(double(next)-double(s.phase_sample))/fs;s.phase_sample=next;
        }
        if(!windows.empty()) {
            result.carriers=ofdm.transform(samples,windows,timing,&commands);commands.wait();
            const auto* stats=static_cast<const C*>(result.carriers.pilots.mapped);
            std::vector<double> rotations(windows.size());
            for(unsigned row=0;row<windows.size();++row) {
                auto& meta=result.symbols[row];auto& s=channels[meta.channel];
                if(!s.locked)continue;
                const auto* v=stats+row*result.carriers.pilot_stride;
                const auto mod=std::bit_cast<std::uint32_t>(v[0].real());meta.modulo4=mod;
                s.status.pilot_quality=v[0].imag();
                s.status.pilot_snr_db=10*std::log10(std::max(double(v[1].imag()),1e-30)/std::max(double(v[1].real()),1e-30));
                meta.tps=v[result.carriers.pilot_stride-1];
                if(v[0].imag()<.3F || (s.previous && mod!=(s.last_mod+1)%4)) {reset(meta.channel);continue;}
                const std::span<const C> pilots{v+2,pattern.continual.size()};
                const double centre_phase=meta.window.phase-tau*meta.window.frequency_hz*4096/fs;
                const double n=double(meta.window.sample)+4096;
                if(s.previous) {
                    const double window_motion=double(meta.window.sample-s.previous_sample)-s.length();
                    const auto fit=pilot_phase(pilots,s.previous_pilots,pattern.continual,tau*window_motion/8192,s.pilot_precision);
                    const double dt=double(meta.window.sample-s.previous_sample)/fs;
                    const double applied=-(centre_phase-s.previous_centre_phase)/(tau*dt);
                    const double measured=applied+std::remainder(fit.phase,tau)/(tau*dt);
                    s.estimated_frequency+=(1-std::exp(-dt/.02))*(measured-s.estimated_frequency);
                    // Moving the FFT window is not an ADC clock observation.
                    // Remove that known integer motion from the phase slope.
                    const double rate_error=(fit.slope*8192/tau-window_motion)/(dt*fs);
                    observations.push_back({TimingObservation::Kind::rate,meta.channel,n,rate_error});
                    s.source_rate_error+=(-std::expm1(-dt/.05))*(rate_error-s.source_rate_error);
                    s.status.residual_rate_ppm=rate_error*1e6;
                    const double alpha=-std::expm1(-dt/.25);
                    const double phase=std::remainder(fit.phase,tau)*180/std::numbers::pi;
                    s.phase_square+=alpha*(phase*phase-s.phase_square);
                    s.status.phase_rms_degrees=std::sqrt(s.phase_square);
                }
                s.previous_pilots.assign(pilots.begin(),pilots.end());s.previous=true;s.last_mod=mod;
                s.previous_sample=meta.window.sample;s.previous_centre_phase=centre_phase;
                if(s.age==128){s.anchor.assign(pilots.size(),C{});s.anchor_slope=0;}
                if(s.age>=128 && s.age<160)for(unsigned j=0;j<pilots.size();++j)s.anchor[j]+=pilots[j]/32.0F;
                if(s.age>=160) {
                    const auto fit=pilot_phase(pilots,s.anchor,pattern.continual,s.anchor_slope,s.pilot_precision,std::max(double(v[1].real()),1e-30));
                    s.anchor_slope=fit.slope;
                    const double local_offset=fit.slope*8192/tau;
                    observations.push_back({TimingObservation::Kind::offset,meta.channel,n,local_offset-s.window_shift,fit.delay_variance});
                    // Carrier phase is a demodulation coordinate, not a clock
                    // observation. Export exactly the rotation applied to Y;
                    // the response and these coordinates retain its reference.
                    rotations[row]=-fit.phase;meta.window.phase+=rotations[row];
                    meta.ready=true;meta.discontinuity=!s.announced;s.announced=true;
                    meta.source_offset_samples=local_offset;
                    meta.source_rate_error=s.source_rate_error;
                    s.status.timing_samples=local_offset-s.window_shift;
                    s.status.timing_sigma_samples=std::sqrt(fit.delay_variance);
                    const double alpha=-std::expm1(-double(s.length())/(fs*.25));
                    s.timing_square+=alpha*(s.status.timing_samples*s.status.timing_samples-s.timing_square);
                    s.status.timing_rms_samples=std::sqrt(s.timing_square);
                    s.status.ready=true;
                }
                ++s.age;
                ++s.status.symbols;
            }
            ofdm.rotate(rotations,timing,output_commands);
        }
        for(unsigned c=0;c<channels.size();++c) {
            auto& s=channels[c];
            s.frequency=s.estimated_frequency;
            if(s.locked && s.age>=160){
                // Keep the useful FFT interval inside its own cyclic prefix
                // even when transmitter clocks differ. Absolute window times
                // and the applied integer motion remain explicit coordinates.
                const auto shift=-std::llround(s.anchor_slope*8192/tau);
                if(shift){
                    const auto next=double(s.next_cp)+double(shift);
                    if(std::abs(shift)>32 || next<double(s.start) || next>double(s.start+s.count))reset(c);
                    else {s.next_cp=std::uint64_t(std::int64_t(s.next_cp)+shift);
                        s.window_shift+=shift;s.anchor_slope+=tau*shift/8192;}
                }
            }
            const auto keep=s.locked ? std::max(s.start,s.next_cp>32 ? s.next_cp-32 : 0) :
                s.start+s.count-std::min<std::size_t>(s.count,3*s.length());
            s.keep_offset=static_cast<std::size_t>(keep-s.start);s.keep_count=s.count-s.keep_offset;
        }
        result.timing=std::move(observations);return result;
    }
};
GpuReceiver::GpuReceiver(const PilotPattern& pilots,std::span<const unsigned> guards,std::size_t maximum,
    std::shared_ptr<gpu::Context> context):
    impl_(std::make_unique<Impl>(pilots,guards,maximum,std::move(context))){}
GpuReceiver::~GpuReceiver()=default;
void GpuReceiver::wait(){impl_->ofdm.wait();impl_->copy.wait();}
void GpuReceiver::reset(){
    wait();auto& p=*impl_;p.have_input=false;p.expected=0;p.observations.clear();
    for(auto& channel:p.channels){const auto guard=channel.guard;channel={};channel.guard=guard;}
}
std::vector<ReceiverStatus> GpuReceiver::status() const {
    std::vector<ReceiverStatus> result;
    for(const auto& s:impl_->channels){auto value=s.status;value.locked=s.locked;value.segment=s.segment;
        value.age=s.age;value.frequency_hz=s.frequency;result.push_back(value);}
    return result;
}
void GpuReceiver::pilot_variance(unsigned channel,std::uint64_t last_sample,std::span<const float> variance){
    auto& p=*impl_;auto& s=p.channels.at(channel);
    if(variance.size()!=6817)throw std::invalid_argument("Incomplete pilot covariance");
    if(!s.locked || last_sample<s.acquisition_sample)return;
    std::vector<double> precision;precision.reserve(p.pattern.continual.size());
    for(auto k:p.pattern.continual){
        if(!std::isfinite(variance[k]) || variance[k]<=0)return;
        precision.push_back(1/double(variance[k]));
    }
    s.pilot_precision=std::move(precision);
}
ReceiverBatch GpuReceiver::process(const ChannelBatch& input,ComputeTimings* timing,gpu::Commands* commands){return impl_->process(input,timing,commands);}
std::vector<Symbol> GpuReceiver::download(const ReceiverBatch& batch) const {
    if(batch.symbols.empty())return {};
    std::vector<OfdmWindow> windows;windows.reserve(batch.symbols.size());
    for(const auto& s:batch.symbols)windows.push_back(s.window);
    auto result=impl_->ofdm.download(batch.carriers,windows);
    for(unsigned i=0;i<result.size();++i)result[i].discontinuity=batch.symbols[i].discontinuity;
    return result;
}
}
