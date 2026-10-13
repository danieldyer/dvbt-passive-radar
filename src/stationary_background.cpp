#include "stationary_background.hpp"
#include <cmath>
#include <stdexcept>

namespace radar {
void StationaryBackground::observe(const SpectralBatch& batch,double interval,std::span<const CarrierTraining> training){
    if(ready())return;
    const auto& estimate=batch.estimate;
    if(!estimate.count || batch.frame.last<batch.frame.first || batch.frame.last>estimate.count ||
       training.size()!=6817 ||
       !estimate.response.mapped || estimate.response.bytes<6817*sizeof(C) || interval<=0 || !std::isfinite(interval))
        throw std::invalid_argument("Stationary prediction needs complete response statistics");
    const auto* h=static_cast<const C*>(estimate.response.mapped);
    for(unsigned k=0;k<6817;++k){
        const auto& t=training[k];
        if(!std::isfinite(t.energy) || t.energy<=0 || !std::isfinite(t.variance) || t.variance<=0 ||
           !std::isfinite(std::norm(t.transfer)) || !std::isfinite(std::norm(h[k])) || std::norm(h[k])<1e-30)continue;
        const double precision=t.energy*double(std::norm(h[k]))/t.variance;
        cross_[k]+=precision*std::complex<double>(t.transfer)/std::complex<double>(h[k]);
        power_[k]+=precision;
    }
    seconds_+=(batch.frame.last-batch.frame.first)*interval;
    // Known pilots alone are not enough: a frozen prediction needs decoded
    // source energy on every occupied carrier before it can be released.
    if(seconds_<3 || std::ranges::any_of(power_,[](double power){return power<=0;}))return;
    response_.resize(6817);variance_.resize(6817);
    for(unsigned k=0;k<6817;++k){response_[k]=static_cast<C>(cross_[k]/power_[k]-1.0);variance_[k]=1/power_[k];}
}
}
