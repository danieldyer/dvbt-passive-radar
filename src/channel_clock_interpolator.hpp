#pragma once
#include "clock_filter.hpp"
#include "dsp.hpp"
#include <algorithm>
#include <cmath>

namespace radar {
// Scalar oracle and coefficient source for the twice-oversampled channel grid.
// Compact polynomial coefficients are designed offline, not fitted at runtime.
struct ChannelClockInterpolator {
    static constexpr unsigned phases=2048,taps=16,left=7;
    std::vector<float> coefficients;
    ChannelClockInterpolator() : coefficients((phases+1)*taps) {
        for(unsigned p=0;p<=phases;++p)for(unsigned j=0;j<taps;++j) {
            const double mu=double(p)/phases-.5;
            double value=0;
            for(auto k=filters::clock_polynomial[j].rbegin();k!=filters::clock_polynomial[j].rend();++k)value=value*mu+*k;
            coefficients[p*taps+j]=float(value);
        }
    }
    C interpolate(std::span<const C> input,double at) const {
        const auto integer=std::int64_t(std::floor(at));
        if(integer<left || std::uint64_t(integer-left+taps)>input.size())throw std::runtime_error("Channel clock interpolation support missing");
        const double phase=(at-double(integer))*phases;
        const auto p=std::min(phases-1,unsigned(phase));const double alpha=phase-p;
        std::complex<double> value{};
        for(unsigned j=0;j<taps;++j)value+=std::complex<double>(input[integer-left+j])*
            ((1-alpha)*coefficients[p*taps+j]+alpha*coefficients[(p+1)*taps+j]);
        return static_cast<C>(value);
    }
};
}
