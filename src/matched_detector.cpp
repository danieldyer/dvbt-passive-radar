#include "matched_detector.hpp"
#include "stationary_background.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <numeric>
#include <time.h>

namespace radar {
namespace {
double thread_cpu(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
float median(std::vector<float>& values){
    if(values.empty())throw std::runtime_error("No background cells for detection calibration");
    auto middle=values.begin()+values.size()/2;std::nth_element(values.begin(),middle,values.end());return *middle;
}
float robust_scale(std::vector<float>& values){return std::max(median(values)/float(std::log(2.0)),1e-30F);}
void regularize_scales(std::vector<float>& scales,std::size_t first,std::size_t observations){
    // An exponential-power median has log-variance approximately 1/(N log(2)^2).
    // Shrink fluctuations below that sampling uncertainty toward a common scale;
    // preserve strong deviations so narrow interference is never averaged away.
    std::vector<float> logs;logs.reserve(scales.size()-first);
    for(std::size_t i=first;i<scales.size();++i)logs.push_back(std::log(scales[i]));
    const float centre=median(logs);
    const double variance=1/(std::max<std::size_t>(1,observations)*std::pow(std::log(2.0),2));
    double observed=0;for(float value:logs)observed+=std::min(std::pow(double(value-centre),2),16*variance);
    observed/=logs.size();const double alpha=observed>variance ? 1-variance/observed : 0;
    for(std::size_t i=first;i<scales.size();++i){
        const double deviation=std::log(scales[i])-centre;
        if(deviation*deviation<16*variance)scales[i]=float(std::exp(centre+alpha*deviation));
    }
}
}
void estimate_map_noise(DopplerMap& map){
    const auto first=std::size_t(std::ceil(600/map.range_step));
    if(map.ranges<=first+32 || map.frequencies<32 || map.values.size()!=map.ranges*map.frequencies)
        throw std::invalid_argument("Insufficient map dimensions for robust noise calibration");
    // Use separated resolution cells for the robust scalar calibration. This
    // is not part of the matched filter and cannot alter its physical SNR.
    const auto stride=std::max<std::size_t>(1,std::size_t(std::round(4*(299792458.0/8e6)/map.range_step)));
    constexpr std::size_t frequency_stride=4;
    const auto nr=(map.ranges+stride-1)/stride,nf=(map.frequencies+frequency_stride-1)/frequency_stride;
    // Read the large complex map once in row order. The independent-cell power
    // grid fits in CPU cache, avoiding repeated strided scans of the full map.
    std::vector<float> power(nr*nf),values;values.reserve(std::max(nr,nf));
    for(std::size_t r=0;r<nr;++r)for(std::size_t f=0;f<nf;++f)
        power[r*nf+f]=std::norm(map.values[(r*stride)*map.frequencies+f*frequency_stride]);
    map.noise_doppler.resize(map.frequencies);std::vector<float> scales(nf);
    for(std::size_t f=0;f<nf;++f){
        values.clear();for(std::size_t r=(first+stride-1)/stride;r<nr;++r)values.push_back(power[r*nf+f]);
        scales[f]=robust_scale(values);
    }
    regularize_scales(scales,0,nr-(first+stride-1)/stride);
    for(std::size_t f=0;f<nf;++f)map.noise_doppler[f*frequency_stride]=scales[f];
    for(std::size_t f=0;f<map.frequencies;++f){
        const auto left=f/frequency_stride*frequency_stride,right=std::min(left+frequency_stride,(map.frequencies-1)/frequency_stride*frequency_stride);
        const float fraction=float(f-left)/frequency_stride;
        map.noise_doppler[f]=(1-fraction)*map.noise_doppler[left]+fraction*map.noise_doppler[right];
    }
    // Range dependence is estimated after Doppler calibration. Exclude the
    // deliberately suppressed near-DC region and sample independent resolution cells.
    map.noise_range.assign(map.ranges,1);scales.resize(nr);std::size_t observations=0;
    for(std::size_t r=0;r<nr;++r){
        values.clear();for(std::size_t f=0;f<nf;f+=2){
            if(std::abs(map.frequency_start+f*frequency_stride*map.frequency_step)<3/map.integration)continue;
            values.push_back(power[r*nf+f]/map.noise_doppler[f*frequency_stride]);
        }
        scales[r]=robust_scale(values);observations=values.size();
    }
    regularize_scales(scales,(first+stride-1)/stride,observations);
    for(std::size_t r=0;r<nr;++r)map.noise_range[r*stride]=scales[r];
    for(std::size_t r=0;r<map.ranges;++r){
        const auto left=r/stride*stride,right=std::min(left+stride,(map.ranges-1)/stride*stride);
        const float fraction=float(r-left)/stride;
        map.noise_range[r]=(1-fraction)*map.noise_range[left]+fraction*map.noise_range[right];
    }
}
void standardize_map(DopplerMap& map){
    if(map.amplitude_units!="complex_reflectivity" || map.noise_range.size()!=map.ranges || map.noise_doppler.size()!=map.frequencies)
        throw std::invalid_argument("Map must have physical amplitudes and a background calibration");
    std::vector<float> gain(map.frequencies);
    std::ranges::transform(map.noise_doppler,gain.begin(),[](float power){return 1/std::sqrt(power);});
    for(std::size_t r=0;r<map.ranges;++r){
        const float range_gain=1/std::sqrt(map.noise_range[r]);
        for(std::size_t f=0;f<map.frequencies;++f)map.values[r*map.frequencies+f]*=range_gain*gain[f];
    }
    map.amplitude_units="noise_standard_deviation";
}
struct MatchedDetector::Impl {
    struct Row {std::array<C,2048> values;double time,information;};
    struct Completed {DopplerMap map;ComputeTimings focus;double calibration{},standardization{};};
    struct Channel {
        GpuNoise noise;GpuFocus focus;StationaryBackground background;std::deque<Row> history;std::uint64_t segment=UINT64_MAX,sequence{};
        double last_publish=-1e30;
        double background_end=-1e30,background_start{};
        // Declared last: its destructor joins before any referenced GPU owner
        // is destroyed, including during exception unwinding.
        std::future<Completed> pending;
        explicit Channel(std::shared_ptr<gpu::Context> context):noise(context),focus(std::move(context)){}
    };
    static void collect(Channel& channel,const Consumer& consumer,DetectionTimings* timing,bool wait){
        if(!channel.pending.valid() || (!wait && channel.pending.wait_for(std::chrono::seconds(0))!=std::future_status::ready))return;
        auto completed=channel.pending.get();
        if(timing){
            timing->focus.gpu+=completed.focus.gpu;timing->focus.batches+=completed.focus.batches;
            timing->focus.gpu_timestamps_available|=completed.focus.gpu_timestamps_available;
            for(unsigned i=0;i<5;++i)timing->focus.stages[i]+=completed.focus.stages[i];
            timing->calibration_cpu+=completed.calibration;
            timing->standardization_cpu+=completed.standardization;
        }
        consumer(completed.map.channel,std::move(completed.map));
    }
    DetectionOptions options;GpuMatchedProfiles profiles;
    std::vector<std::unique_ptr<Channel>> channels;
    Impl(std::shared_ptr<gpu::Context> context,std::size_t count,DetectionOptions opts)
        :options(opts),profiles(context){
        if(!count || !std::isfinite(opts.integration_seconds) || opts.integration_seconds<.5 || opts.integration_seconds>10 ||
           !std::isfinite(opts.update_seconds) || opts.update_seconds<=0 ||
           !std::isfinite(opts.focus.range_metres) || opts.focus.range_metres<1000 || opts.focus.range_metres>30000 ||
           !std::isfinite(opts.focus.doppler_hz) || opts.focus.doppler_hz<4/opts.integration_seconds || opts.focus.doppler_hz>200)
            throw std::invalid_argument("Detection requires CPI 0.5..10 s, range 1..30 km, Doppler limit 4/CPI..200 Hz and a positive update interval");
        for(std::size_t i=0;i<count;++i)channels.push_back(std::make_unique<Channel>(context));
    }
};
MatchedDetector::MatchedDetector(std::shared_ptr<gpu::Context> context,std::size_t channels,DetectionOptions options)
    :impl_(std::make_unique<Impl>(std::move(context),channels,options)){}
MatchedDetector::~MatchedDetector()=default;
void MatchedDetector::append(const SarBatch& batch,const Consumer& consume,DetectionTimings* timing){
    auto& p=*impl_;const auto& frame=batch.spectra.frame;auto& c=*p.channels.at(frame.channel);
    for(auto& channel:p.channels)Impl::collect(*channel,consume,timing,false);
    const unsigned guard=batch.capture.channels.at(frame.channel).guard;
    const double dt=(8192.0+8192.0/guard)/8e6;
    const auto required=std::size_t(std::ceil(p.options.integration_seconds/dt));
    if(c.segment!=frame.segment){Impl::collect(c,consume,timing,true);c.history.clear();c.noise.reset();c.background.reset();c.segment=frame.segment;c.last_publish=-1e30;c.background_end=-1e30;c.background_start=batch.seconds(frame.first);c.focus.prepare(required);}
    if(frame.last<=frame.first+2)return;
    const auto variance=c.noise.estimate(batch.spectra,timing ? &timing->noise : nullptr);
    if(p.options.focus.remove_dc && !c.background.ready()){
        c.background.observe(batch.spectra,dt);
        // Statistics include decoder context. No detection CPI may reuse those
        // observations as held-out data after the frozen estimate is established.
        c.background_end=batch.seconds(frame.symbols.size()-1);return;
    }
    const auto profiles=p.profiles.calculate(batch.spectra,variance,guard,timing ? &timing->profiles : nullptr,true,c.background.response());
    const auto* valid=static_cast<const std::uint32_t*>(batch.spectra.transmitted.valid.mapped);
    for(std::size_t n=0;n<profiles.count;++n){
        const auto row=profiles.first+n;const double time=batch.seconds(row);
        if(time<=c.background_end)continue;
        const auto flags=std::span{valid+(row-1)*6817,2*std::size_t(6817)};
        if(std::ranges::any_of(flags,[](auto v){return !v;}) ||
           (!c.history.empty() && std::abs(time-c.history.back().time-dt)>dt*.01)){
            c.history.clear();if(std::ranges::any_of(flags,[](auto v){return !v;}))continue;
        }
        auto& stored=c.history.emplace_back();stored.time=time;stored.information=profiles.information;
        for(unsigned moment=0;moment<2;++moment)
            std::copy_n(profiles.values.data()+(moment*profiles.count+n)*1024,1024,stored.values.data()+moment*1024);
        if(c.history.size()>required)c.history.pop_front();
        if(c.history.size()!=required || time-c.last_publish+dt*1e-8<p.options.update_seconds)continue;
        std::vector<C> input(2*required*1024);
        std::vector<double> information(required);
        for(std::size_t r=0;r<required;++r)information[r]=c.history[r].information;
        for(std::size_t r=0;r<required;++r)for(unsigned moment=0;moment<2;++moment)
            std::copy_n(c.history[r].values.data()+moment*1024,1024,input.data()+(moment*required+r)*1024);
        // One outstanding job per channel bounds both memory and scheduling.
        // The job owns all numerical inputs; it cannot retain receiver buffers,
        // consumers, timing pointers or the shared clock's mutable history.
        Impl::collect(c,consume,timing,true);
        DopplerMap metadata;metadata.sequence=++c.sequence;metadata.segment=c.segment;
        metadata.requested_integration=p.options.integration_seconds;
        metadata.channel=frame.channel;metadata.reference_origin_samples=batch.spectra.estimate.reference_origin;
        if(c.background.ready()){
            metadata.stationary_prediction="frozen_source_divided_transfer";
            metadata.background_start=c.background_start;metadata.background_end=c.background_end;
            metadata.dc_method="frozen_stationary_prediction_and_constant_projection";
        }
        c.pending=std::async(std::launch::async,[focus=&c.focus,input=std::move(input),information=std::move(information),required,dt,
            rf=batch.capture.channels.at(frame.channel).rf_hz,first=c.history.front().time,options=p.options.focus,
            normalize=p.options.standardize_noise,metadata=std::move(metadata)]()mutable{
            Impl::Completed completed;
            auto& map=completed.map;map=focus->calculate(input,required,dt,rf,first,options,&completed.focus,information);
            map.sequence=metadata.sequence;map.segment=metadata.segment;map.channel=metadata.channel;
            map.requested_integration=metadata.requested_integration;map.reference_origin_samples=metadata.reference_origin_samples;
            if(!metadata.stationary_prediction.empty()){
                map.stationary_prediction=std::move(metadata.stationary_prediction);map.dc_method=std::move(metadata.dc_method);
                map.background_start=metadata.background_start;map.background_end=metadata.background_end;
            }
            const auto start=thread_cpu();estimate_map_noise(map);
            completed.calibration=thread_cpu()-start;
            if(normalize){const auto mark=thread_cpu();standardize_map(map);completed.standardization=thread_cpu()-mark;}
            return completed;
        });
        c.last_publish=time;
    }
}
void MatchedDetector::finish(const Consumer& consume,DetectionTimings* timing){
    for(auto& channel:impl_->channels)Impl::collect(*channel,consume,timing,true);
}
}
