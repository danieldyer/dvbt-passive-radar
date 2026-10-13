#pragma once
#include "gpu_pipeline.hpp"
#include "gpu_noise.hpp"
#include <algorithm>
#include <complex>

namespace radar {
// A frozen nuisance prediction, relative to Hmin. It is only for the optional
// stationary-suppression branch; it never becomes the channel/template response.
class StationaryBackground {
    std::vector<std::complex<double>> cross_=std::vector<std::complex<double>>(6817);
    std::vector<double> power_=std::vector<double>(6817);
    std::vector<C> response_;
    std::vector<double> variance_;
    double seconds_{};
public:
    bool ready() const {return !response_.empty();}
    double seconds() const {return seconds_;}
    std::span<const C> response() const {return response_;}
    // Conditional covariance of the frozen complex coefficients. A coefficient
    // error is shared by all subsequent rows: it is NOT extra white noise.
    std::span<const double> variance() const {return variance_;}
    void reset(){std::fill(cross_.begin(),cross_.end(),std::complex<double>{});std::fill(power_.begin(),power_.end(),0);response_.clear();variance_.clear();seconds_=0;}
    void observe(const SpectralBatch&,double symbol_interval,std::span<const CarrierTraining>);
};
}
