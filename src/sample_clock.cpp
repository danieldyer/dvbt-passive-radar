#include "sample_clock.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace radar {
SampleClock::SampleClock(double fs,std::span<const double> centres,double degrees)
    : fs_(fs),degrees_(degrees),observers_(centres.size()) {
    if(centres.empty() || !std::isfinite(fs) || fs<=0 || !std::isfinite(degrees) || degrees<=0)
        throw std::runtime_error("Invalid common sample-clock specification");
    for(double f:centres)if(!std::isfinite(f))throw std::runtime_error("Non-finite channel centre");
    const auto [lo,hi]=std::minmax_element(centres.begin(),centres.end());
    // Bound differential phase between the most widely separated occupied
    // carriers. A common phase rotation does not hide a relative timing error.
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
        (o.rate-1)*1e6,o.points.empty() ? 0 : (o.points.back().n-o.points.front().n)/fs_,o.fit_rms});
    return result;
}
void SampleClock::schedule_unlocked(std::uint64_t start,double rate) {
    if(start<segments_.back().start+64 || !std::isfinite(rate) || rate<.999 || rate>1.001)
        throw std::runtime_error("Invalid or retroactive sample-clock segment");
    segments_.push_back({start,segments_.back().offset_at(double(start)),rate});
}
void SampleClock::schedule(std::uint64_t start,double rate) {std::lock_guard lock(mutex_);schedule_unlocked(start,rate);}
void SampleClock::reset_observer(std::size_t channel) {std::lock_guard lock(mutex_);observers_.at(channel)=Observer{};}
void SampleClock::observe_rate(std::size_t channel,double n,double error) {
    if(!std::isfinite(n) || !std::isfinite(error) || std::abs(error)>.001)return;
    std::lock_guard lock(mutex_);auto& o=observers_.at(channel);
    if(n<=o.rate_time)return;
    const double applied=o.rate_time<0 ? at_unlocked(n).rate : 1+
        (at_unlocked(n).offset_at(n)-at_unlocked(o.rate_time).offset_at(o.rate_time))/(n-o.rate_time);
    const double rate=applied/(1+error);
    const double alpha=o.rate_time<0 ? 1 : -std::expm1(-(n-o.rate_time)/(fs_*.03));
    o.rate+=alpha*(rate-o.rate);o.rate_time=n;
}
void SampleClock::observe_offset(std::size_t channel,double n,double error) {
    if(!std::isfinite(n) || !std::isfinite(error))return;
    std::lock_guard lock(mutex_);auto& o=observers_.at(channel);
    if(!o.points.empty() && n<=o.points.back().n)return;
    if(!o.anchored) {
        const double t=(n-reference_epoch_)/fs_;
        const double reference=reference_ready_ ? reference_offset_+t*(reference_velocity_+t*reference_acceleration_) : at_unlocked(n).offset_at(n);
        o.bias=reference-at_unlocked(n).offset_at(n)+error*at_unlocked(n).rate;
        o.anchored=true;
    }
    // Remove the correction already applied to this observation. Otherwise a
    // feedback loop would mistake its own resampling for new oscillator drift.
    const double y=at_unlocked(n).offset_at(n)-error*at_unlocked(n).rate+o.bias;
    o.points.push_back({n,y});
    while(o.points.size()>1 && n-o.points.front().n>fs_)o.points.pop_front();
}
void SampleClock::update(std::uint64_t start) {
    std::lock_guard lock(mutex_);
    if(start<=segments_.back().start)return;
    const double n=double(start),current=segments_.back().offset_at(n);
    double rate_sum=0,weights=0;
    double acquisition_sum=0,acquisition_count=0;
    double position_sum=0,velocity_sum=0,acceleration_sum=0,reference_weights=0;
    for(auto& o:observers_) {
        if(o.rate_time<0 || n-o.rate_time>fs_*.5)continue;
        // Quadratic clock-offset fit, centred on the newest measurement. This
        // separates each transmitter's constant delay from a shared clock rate.
        // Fall back to differential acquisition until a useful baseline exists.
        if(o.points.size()<32 || o.points.back().n-o.points.front().n<fs_*.08) {
            acquisition_sum+=o.rate;acquisition_count+=1;continue;
        }
        const double epoch=o.points.back().n;
        std::array<std::array<double,4>,3> a{};
        for(auto p:o.points) {
            const double t=(p.n-epoch)/fs_;const std::array v{1.0,t,t*t};
            for(unsigned i=0;i<3;++i) {for(unsigned j=0;j<3;++j)a[i][j]+=v[i]*v[j];a[i][3]+=v[i]*p.y;}
        }
        for(unsigned i=0;i<3;++i) {
            const double pivot=a[i][i];
            for(unsigned j=i;j<4;++j)a[i][j]/=pivot;
            for(unsigned k=0;k<3;++k)if(k!=i) {const double q=a[k][i];for(unsigned j=i;j<4;++j)a[k][j]-=q*a[i][j];}
        }
        const double c=a[0][3],v=a[1][3],acc=a[2][3];
        double residual=0;
        for(auto p:o.points) {const double t=(p.n-epoch)/fs_,e=p.y-(c+t*(v+t*acc));residual+=e*e;}
        const double sigma=std::sqrt(residual/std::max<std::size_t>(1,o.points.size()-3));
        o.fit_rms=sigma;
        // A linear map approximating a quadratic has peak error |a|T^2/4.
        // Reserve half the phase budget for this extrapolation and cap updates
        // at 0.1 s; processing batches may update more frequently.
        const double horizon=std::clamp(std::sqrt(2*budget_/std::max(1e-12,std::abs(acc))),.025,.1);
        const double t=(n-epoch)/fs_+horizon;
        const double target=c+t*(v+t*acc);
        const double rate=1+(target-current)/(fs_*horizon);
        if(!std::isfinite(rate) || std::abs(rate-1)>.001)continue;
        // This reference is the arithmetic ensemble of transmitter clocks.
        // Time-varying precision weights would redefine its phase origin when
        // real transmitter rates differ, converting SNR changes into ADC warp.
        // Fit precision remains telemetry, not a change to the clock standard.
        constexpr double w=1;
        rate_sum+=w*rate;weights+=w;
        const double now=(n-epoch)/fs_;
        position_sum+=w*(c+now*(v+now*acc));velocity_sum+=w*(v+2*now*acc);
        acceleration_sum+=w*acc;reference_weights+=w;
    }
    if(reference_weights) {
        reference_ready_=true;reference_epoch_=n;reference_offset_=position_sum/reference_weights;
        reference_velocity_=velocity_sum/reference_weights;reference_acceleration_=acceleration_sum/reference_weights;
    }
    // Differential acquisition supplies a clock only until anchored fits are
    // available. It has no phase reference and cannot join a phase servo.
    if(weights) schedule_unlocked(start,rate_sum/weights);
    else if(acquisition_count) schedule_unlocked(start,acquisition_sum/acquisition_count);
}
}
