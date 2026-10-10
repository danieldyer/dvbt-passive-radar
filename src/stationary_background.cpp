#include "stationary_background.hpp"
#include <cmath>
#include <stdexcept>

namespace radar {
void StationaryBackground::observe(const SpectralBatch& batch,double interval){
    if(ready())return;
    const auto& estimate=batch.estimate;
    if(!estimate.count || batch.frame.last<batch.frame.first || batch.frame.last>estimate.count ||
       !estimate.statistics.mapped || estimate.statistics.bytes<6817*sizeof(ResponseStatistics) ||
       !estimate.response.mapped || estimate.response.bytes<6817*sizeof(C) || interval<=0 || !std::isfinite(interval))
        throw std::invalid_argument("Stationary prediction needs complete response statistics");
    const auto* h=static_cast<const C*>(estimate.response.mapped);
    const auto* statistics=static_cast<const ResponseStatistics*>(estimate.statistics.mapped);
    const double fraction=double(batch.frame.last-batch.frame.first)/estimate.count;
    for(unsigned k=0;k<6817;++k){
        if(!std::isfinite(statistics[k].reference_power) || statistics[k].reference_power<=0 ||
           !std::isfinite(std::norm(statistics[k].cross)) || !std::isfinite(std::norm(h[k])) || std::norm(h[k])<1e-30)continue;
        cross_[k]+=fraction*std::complex<double>(statistics[k].cross)/std::complex<double>(h[k]);
        power_[k]+=fraction*statistics[k].reference_power;
    }
    seconds_+=(batch.frame.last-batch.frame.first)*interval;
    // Known pilots alone are not enough: a frozen prediction needs decoded
    // source energy on every occupied carrier before it can be released.
    if(seconds_<3 || std::ranges::any_of(power_,[](double power){return power<1000;}))return;
    response_.resize(6817);
    for(unsigned k=0;k<6817;++k)response_[k]=power_[k]>0 ? static_cast<C>(cross_[k]/power_[k]-1.0) : C{};
}
}
