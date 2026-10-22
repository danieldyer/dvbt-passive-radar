#pragma once
#include "complex_sample.hpp"
#include <cmath>
#include <cstdint>
#include <numbers>
#include <limits>
#include <span>
#include <stdexcept>

namespace radar {
struct PilotPhase {
    double phase{},slope{};
    // Conditional Fisher covariance after profiling complex amplitude.
    // Delay is in 8 MS/s samples. The supplied anchor is fixed; a finite
    // anchor's shared uncertainty is a separate nuisance, not fresh white
    // measurement noise at each future symbol.
    double delay_variance{},phase_variance{},phase_delay_covariance{};
};

// Conditional complex-Gaussian ML: now[k] = a * before[k] * exp(i*b*k) + e[k].
// Eliminating a leaves max_b |sum(now*conj(before)*exp(-i*b*k))|^2. The
// Optional precision is the inverse additive variance of independent secondary
// observations, giving the diagonal-covariance GLS likelihood. The
// previous timing observation selects the local branch; no sequential unwrap,
// amplitude clipping or empirical phase weights are used. The supplied anchor
// is treated as known; its estimation uncertainty is not a CRLB certificate.
inline PilotPhase pilot_phase(std::span<const C> now,std::span<const C> before,
                             std::span<const std::uint32_t> indices,double slope=0,std::span<const double> precision={},double fallback_variance=1.) {
    if(now.size()!=before.size() || now.size()!=indices.size() || (!precision.empty() && precision.size()!=indices.size()))
        throw std::invalid_argument("Pilot phase dimensions disagree");
    for(double weight:precision)if(!std::isfinite(weight) || weight<0)throw std::invalid_argument("Invalid pilot precision");
    if(!std::isfinite(fallback_variance) || fallback_variance<=0)throw std::invalid_argument("Invalid pilot noise variance");
    struct Sum {std::complex<double> value{},first{},second{};};
    auto sum=[&](double b){
        Sum result;
        for(std::size_t j=0;j<indices.size();++j){
            const double x=(double(indices[j])-3408)/8192;
            const auto z=(precision.empty() ? 1. : precision[j])*std::complex<double>(now[j])*std::conj(std::complex<double>(before[j]))*std::polar(1.0,-b*x);
            result.value+=z;result.first+=x*z;result.second+=x*x*z;
        }
        return result;
    };
    double b=slope*8192;auto current=sum(b);
    // Bounded local Newton solve with likelihood-increasing backtracking.
    // Frequency coordinates are scaled to avoid a poorly conditioned Hessian.
    for(unsigned iteration=0;iteration<4;++iteration){
        const double gradient=2*std::imag(current.first*std::conj(current.value));
        const double curvature=2*(std::norm(current.first)-std::real(current.second*std::conj(current.value)));
        if(!(curvature<0) || std::abs(gradient)<1e-12*std::norm(current.value))break;
        double step=-gradient/curvature;
        // The predicted likelihood increase is -g^2/(2*h). Once it is
        // below double precision, backtracking cannot resolve an improvement.
        // Avoid eight redundant trigonometric sweeps at that numerical floor.
        if(gradient*step<=4*std::numeric_limits<double>::epsilon()*std::norm(current.value))break;
        bool accepted=false;
        for(unsigned trial=0;trial<8;++trial){
            auto next=sum(b+step);
            if(std::norm(next.value)>std::norm(current.value)){b+=step;current=next;accepted=true;break;}
            step*=.5;
        }
        if(!accepted)break;
    }
    double energy=0,first=0,second=0;
    for(std::size_t j=0;j<indices.size();++j){
        const double weight=(precision.empty() ? 1. : precision[j])*std::norm(std::complex<double>(before[j]));
        const double omega=2*std::numbers::pi*(double(indices[j])-3408)/8192;
        energy+=weight;first+=weight*omega;second+=weight*omega*omega;
    }
    const double amplitude=energy>0 ? std::norm(current.value)/(energy*energy) : 0;
    const double determinant=energy*second-first*first;
    if(!(amplitude>0) || !(determinant>0))
        return {std::arg(current.value),b/8192,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),0};
    const double scale=(precision.empty() ? fallback_variance : 1.)/(2*amplitude*determinant);
    return {std::arg(current.value),b/8192,scale*energy,scale*second,-scale*first};
}
}
