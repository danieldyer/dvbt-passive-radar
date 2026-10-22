#include "sample_clock.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace radar {
SampleClock::SampleClock(double fs,std::span<const double> centres,double degrees,ClockControlSettings settings)
    :fs_(fs),degrees_(degrees),settings_(settings),
     estimator_(centres.size(),fs,settings.receiver,std::vector<ClockNoise>(centres.size(),settings.transmitter)),observers_(centres.size()) {
    if(centres.empty() || !std::isfinite(fs) || fs<=0 || !std::isfinite(degrees) || degrees<=0)
        throw std::runtime_error("Invalid common sample-clock specification");
    for(double f:centres)if(!std::isfinite(f))throw std::runtime_error("Non-finite channel centre");
    for(double x:{settings.settling_seconds,settings.innovation_huber_sigma,settings.unknown_sigma_samples})
        if(!std::isfinite(x) || x<=0)throw std::invalid_argument("Invalid clock controller settings");
    const auto [lo,hi]=std::minmax_element(centres.begin(),centres.end());
    budget_=degrees/360*fs/(*hi-*lo+6656250);
}
ClockSegment SampleClock::at_unlocked(double n) const {
    auto it=std::upper_bound(segments_.begin(),segments_.end(),n,
        [](double n,const ClockSegment& s){return n<double(s.start);});
    return it==segments_.begin() ? segments_.front() : *--it;
}
ClockSegment SampleClock::at(double n) const {std::lock_guard lock(mutex_);return at_unlocked(n);}
std::vector<ClockSegment> SampleClock::segments() const {std::lock_guard lock(mutex_);return segments_;}
std::vector<ClockObserverStatus> SampleClock::status(double n) const {
    std::lock_guard lock(mutex_);std::vector<ClockObserverStatus> result;
    for(const auto& o:observers_)result.push_back({o.rate_time>=0 && n-o.rate_time<fs_*.5,o.anchored,o.points.size(),
        (o.rate-1)*1e6,o.points.empty() ? 0 : (o.points.back().n-o.points.front().n)/fs_,o.fit_rms,
        o.points.empty() ? std::numeric_limits<double>::quiet_NaN() : std::sqrt(o.points.back().variance),
        o.innovation.value,o.innovation.variance,o.innovation.normalized_square,o.downweighted,o.late});
    return result;
}
void SampleClock::schedule_unlocked(std::uint64_t start,double rate) {
    if(start<segments_.back().start+64 || !std::isfinite(rate) || rate<.999 || rate>1.001)
        throw std::runtime_error("Invalid or retroactive sample-clock segment");
    segments_.push_back({start,segments_.back().offset_at(double(start)),rate});
}
void SampleClock::schedule(std::uint64_t start,double rate){std::lock_guard lock(mutex_);schedule_unlocked(start,rate);}
void SampleClock::enable_diagnostics(bool enabled){std::lock_guard lock(mutex_);diagnostics_enabled_=enabled;diagnostics_.clear();}
std::vector<ClockObservation> SampleClock::take_observations(){std::lock_guard lock(mutex_);std::vector<ClockObservation> out;out.swap(diagnostics_);return out;}
ClockEstimate SampleClock::estimate(double n) const {
    std::lock_guard lock(mutex_);
    if(estimator_.ready())return estimator_.estimate(std::max(n/fs_,estimator_.epoch()));
    const double unknown=std::numeric_limits<double>::quiet_NaN();return {0,0,0,unknown,unknown,unknown};
}
void SampleClock::reset_observer(std::size_t channel){
    std::lock_guard lock(mutex_);auto& o=observers_.at(channel);
    if(diagnostics_enabled_)diagnostics_.push_back({ClockObservation::Kind::reset,channel,o.rate_time});
    if(diagnostics_enabled_)for(auto observation:pending_)if(observation.channel==channel){
        observation.innovation.gaussian_inlier=false;observation.innovation.robust_weight=0;diagnostics_.push_back(observation);
    }
    o=Observer{};
    std::erase_if(pending_,[&](const auto& p){return p.channel==channel;});
    // Keep the common and differential clock states: reacquisition changes a
    // pilot anchor, not the ADC time reference or the physical TX oscillator.
}
void SampleClock::observe_rate(std::size_t channel,double n,double error){
    if(!std::isfinite(n) || !std::isfinite(error) || std::abs(error)>.001)return;
    std::lock_guard lock(mutex_);auto& o=observers_.at(channel);if(n<=o.rate_time)return;
    const auto applied=at_unlocked(n);
    const double rate=o.rate_time<0 ? applied.rate : 1+
        (applied.offset_at(n)-at_unlocked(o.rate_time).offset_at(o.rate_time))/(n-o.rate_time);
    // Differential CP timing is acquisition-only. It is correlated with
    // adjacent pilot offsets and must not be counted as independent evidence.
    const double alpha=o.rate_time<0 ? 1 : -std::expm1(-(n-o.rate_time)/(fs_*.03));
    o.rate+=alpha*(rate/(1+error)-o.rate);o.rate_time=n;
    if(diagnostics_enabled_)diagnostics_.push_back({ClockObservation::Kind::rate,channel,n,error,0,o.rate,applied.offset_at(n),applied.rate});
}
void SampleClock::observe_offset(std::size_t channel,double n,double error,double variance){
    if(!std::isfinite(n) || !std::isfinite(error))return;
    std::lock_guard lock(mutex_);observers_.at(channel);const auto s=at_unlocked(n);
    ClockObservation observation{ClockObservation::Kind::offset,channel,n,error,variance,
        s.offset_at(n)-error*s.rate,s.offset_at(n),s.rate};
    if(std::isinf(variance) || variance<=0){
        observation.innovation.gaussian_inlier=false;observation.innovation.robust_weight=0;
        if(diagnostics_enabled_)diagnostics_.push_back(observation);return;
    }
    pending_.push_back(observation);
}
void SampleClock::update(std::uint64_t start){
    std::lock_guard lock(mutex_);if(start<=segments_.back().start)return;
    // Receiver batches contain channel-major observations. Assimilate them in
    // actual epoch order, never in decoder-completion/channel order.
    std::stable_sort(pending_.begin(),pending_.end(),[](const auto& a,const auto& b){return a.sample<b.sample;});
    for(auto observation:pending_){
        auto& o=observers_[observation.channel];const double t=observation.sample/fs_;
        if(estimator_.ready() && t<estimator_.epoch()){
            observation.late=true;++o.late;
            if(diagnostics_enabled_)diagnostics_.push_back(observation);
            continue;
        }
        if(!estimator_.ready()){
            double rate=0;unsigned count=0;
            for(const auto& source:observers_)if(source.rate_time>=0){rate+=source.rate-1;++count;}
            estimator_.initialize(t,observation.applied_offset,count ? fs_*rate/count : 0);
        }
        if(!o.anchored){o.bias=estimator_.channel_offset(observation.channel,t)-observation.feedback_removed;o.anchored=true;}
        const bool known=std::isfinite(observation.variance);
        const double variance=known ? observation.variance*observation.applied_rate*observation.applied_rate :
            settings_.unknown_sigma_samples*settings_.unknown_sigma_samples;
        const auto innovation=estimator_.observe(observation.channel,t,observation.feedback_removed+o.bias,variance,
            known ? settings_.innovation_huber_sigma*settings_.innovation_huber_sigma : std::numeric_limits<double>::infinity());
        observation.innovation=innovation;o.innovation=innovation;
        if(!innovation.gaussian_inlier)++o.downweighted;
        // Telemetry includes bounded observations; excluding them would hide
        // the very model mismatch the innovation history is meant to expose.
        o.points.push_back({observation.sample,observation.feedback_removed+o.bias,variance});
        while(o.points.size()>1 && observation.sample-o.points.front().n>fs_)o.points.pop_front();
        o.fit_rms=std::sqrt(.99*o.fit_rms*o.fit_rms+.01*innovation.value*innovation.value);
        if(diagnostics_enabled_)diagnostics_.push_back(observation);
    }
    pending_.clear();
    if(estimator_.ready()){
        const double n=double(start);const auto state=estimator_.estimate(n/fs_);
        // A fixed first-order phase servo follows the conditional clock rate.
        // No channel-dependent horizon/precision can redefine the reference.
        const double velocity=state.velocity+(state.offset-at_unlocked(n).offset_at(n))/settings_.settling_seconds;
        schedule_unlocked(start,1+velocity/fs_);
    }else{
        double rate=0;unsigned count=0;
        for(const auto& o:observers_)if(o.rate_time>=0 && double(start)-o.rate_time<fs_*.5){rate+=o.rate;++count;}
        if(count)schedule_unlocked(start,rate/count);
    }
}
}
