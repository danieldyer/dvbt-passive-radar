#pragma once
#include "signal_constants.hpp"
#include <volk/volk.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>
#include <span>

namespace radar {
// Centred 96-tap Kaiser-windowed sinc, 2048 phases with linear phase-table interpolation.
// Input points to sample floor(t)-47; fractional coordinate mu is in [0,1).
// Timing latency is represented by the caller's absolute sample coordinate.
class FractionalResampler {
    static constexpr unsigned phases=2048, taps=96;
    std::vector<float> table_;
    static double i0(double x) {
        double term=1,sum=1;
        for(unsigned k=1;k<80;++k) { term*=x*x/(4.0*k*k);sum+=term;if(term<sum*1e-16)break; }
        return sum;
    }
public:
    FractionalResampler() : table_((phases+1)*taps) {
        for(unsigned p=0;p<=phases;++p) {
            const double mu=static_cast<double>(p)/phases;
            double sum=0;
            for(unsigned j=0;j<taps;++j) {
                const double x=static_cast<double>(j)-47-mu, z=x/48;
                const double h=(std::abs(x)<1e-14 ? 0.98 : std::sin(0.98*std::numbers::pi*x)/(std::numbers::pi*x))*
                    i0(10*std::sqrt(std::max(0.0,1-z*z)))/i0(10);
                table_[p*taps+j]=static_cast<float>(h);sum+=h;
            }
            for(unsigned j=0;j<taps;++j)table_[p*taps+j]/=static_cast<float>(sum);
        }
    }
    std::complex<float> interpolate(const std::complex<float>* in,double mu) const {
        const double phase=mu*phases;
        const auto p=std::min(phases-1,static_cast<unsigned>(phase));
        std::complex<float> a,b;
#if defined(__ARM_NEON)
        // Evaluate the adjacent phase filters together: each input vector is
        // loaded once, with independent accumulators to hide FMA latency.
        const auto zero=vdupq_n_f32(0);
        std::array<float32x4_t,2> ar{zero,zero},ai{zero,zero},br{zero,zero},bi{zero,zero};
        const auto* first=table_.data()+p*taps;
        const auto* second=first+taps;
        for(unsigned j=0;j<taps;j+=8)for(unsigned half=0;half<2;++half) {
            const auto k=j+4*half;
            const auto value=vld2q_f32(reinterpret_cast<const float*>(in+k));
            const auto wa=vld1q_f32(first+k),wb=vld1q_f32(second+k);
            ar[half]=vfmaq_f32(ar[half],wa,value.val[0]);
            ai[half]=vfmaq_f32(ai[half],wa,value.val[1]);
            br[half]=vfmaq_f32(br[half],wb,value.val[0]);
            bi[half]=vfmaq_f32(bi[half],wb,value.val[1]);
        }
        a={vaddvq_f32(vaddq_f32(ar[0],ar[1])),vaddvq_f32(vaddq_f32(ai[0],ai[1]))};
        b={vaddvq_f32(vaddq_f32(br[0],br[1])),vaddvq_f32(vaddq_f32(bi[0],bi[1]))};
#else
        volk_32fc_32f_dot_prod_32fc(&a,in,table_.data()+p*taps,taps);
        volk_32fc_32f_dot_prod_32fc(&b,in,table_.data()+(p+1)*taps,taps);
#endif
        return a+static_cast<float>(phase-p)*(b-a);
    }
    unsigned ntaps() const { return taps; }
    std::span<const float> coefficients() const { return table_; }
};
}
