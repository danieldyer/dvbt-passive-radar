#pragma once
// Historical empirical display calibration, retained only for diagnostic replay.
// Not covariance whitening or a proven CFAR statistic. Never used by production.
#include "gpu_focus.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
namespace radar::validation {
namespace {
inline float median(std::vector<float>& values){
    if(values.empty())throw std::runtime_error("No background cells for detection calibration");
    auto middle=values.begin()+values.size()/2;std::nth_element(values.begin(),middle,values.end());return *middle;
}
inline float robust_scale(std::vector<float>& values){return std::max(median(values)/float(std::log(2.0)),1e-30F);}
inline void regularize_scales(std::vector<float>& scales,std::size_t first,std::size_t observations){
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
inline void estimate_map_noise(DopplerMap&,std::span<const C>);
inline void estimate_map_noise(DopplerMap& map){
    estimate_map_noise(map,map.values);
}
inline void estimate_map_noise(DopplerMap& map,std::span<const C> input){
    const auto first=std::size_t(std::ceil(600/map.range_step));
    if(map.ranges<=first+32 || map.frequencies<32 || input.size()!=map.ranges*map.frequencies)
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
        power[r*nf+f]=std::norm(input[(r*stride)*map.frequencies+f*frequency_stride]);
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
inline void standardize_map(DopplerMap& map){
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
}
