#include "gpu_frontend.hpp"
#include "clock_interpolator.hpp"
#include "frontend_reference.hpp"
#include <algorithm>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double fs=56e6,tau=2*std::numbers::pi;
void require(bool test,const char* message) {if(!test)throw std::runtime_error(message);}
}
int main(int argc,char**) {
    try {
        const std::array offsets{-21e6,137123.25,21e6};
        // Verify the dynamically sized raw interpolator against an analytic
        // fractional delay over its entire declared passband and phase table.
        for(double band:{.1420201,.4344844,.4785157}) {
            ClockInterpolator filter(band);double maximum=0;
            for(unsigned p=0;p<65;++p) {
                const double mu=(p+.371)/66,phase=mu*filter.phases;
                const auto index=unsigned(phase);const double alpha=phase-index;
                for(unsigned k=0;k<=256;++k) {
                    const double f=band*k/256;std::complex<double> response{};
                    for(unsigned j=0;j<filter.taps;++j) {
                        const double h=(1-alpha)*filter.coefficients[index*filter.taps+j]+alpha*filter.coefficients[(index+1)*filter.taps+j];
                        response+=h*std::polar(1.0,tau*f*(double(j)-filter.left-mu));
                    }
                    maximum=std::max(maximum,std::abs(response-1.0));
                }
            }
            std::cout<<"Raw interpolator: band="<<band<<", taps="<<filter.taps<<", complex error="<<maximum<<'\n';
            require(maximum<5e-6,"Raw clock interpolator does not preserve its declared passband");
        }
        auto clock=std::make_shared<SampleClock>(fs,offsets,3);
        require(std::abs(clock->timing_budget_samples()-3.0/360*fs/48656250)<1e-12,"Incorrect full-span phase budget");
        clock->schedule(1024,1.00003);clock->schedule(13237,.999985);
        const auto segments=clock->segments();
        for(std::size_t i=1;i<segments.size();++i)
            require(std::abs(segments[i].offset-segments[i-1].offset_at(double(segments[i].start)))<1e-12,"Clock update jumps sample position");
        // Independent receivers have different fixed propagation delays, but
        // observe one ADC oscillator. Neither delay may become a clock rate.
        SampleClock ensemble(fs,offsets,3);
        std::mt19937 random(199);std::normal_distribution<double> noise(0,.0005);
        constexpr double ppm=4e-6,drift=.15e-6;
        auto actual_offset=[](double n){return ppm*n+.5*drift*n*n/fs;};
        double worst=0;
        for(unsigned i=1;i<=10000;++i) {
            const double n=fs*i*.001;
            const double actual=actual_offset(n);
            for(std::size_t c=0;c<offsets.size();++c) {
                if((c==2 && i<5000) || (c==0 && i>7000))continue;
                const auto s=ensemble.at(n);
                const auto previous=n-fs*.001;
                const double applied=1+(s.offset_at(n)-ensemble.at(previous).offset_at(previous))/(fs*.001);
                const double true_rate=1+(actual-actual_offset(previous))/(fs*.001);
                ensemble.observe_rate(c,n,applied/true_rate-1);
                if(i>=200)ensemble.observe_offset(c,n,(s.offset_at(n)-actual+17*c)/s.rate+noise(random));
            }
            if(i%20==0)ensemble.update(std::uint64_t(n+fs*.03));
            if(i>2000) {
                // Absolute epoch is arbitrary; compare a 1-second interval.
                const double old=n-fs;
                const double error=(ensemble.at(n).offset_at(n)-ensemble.at(old).offset_at(old))-(actual-actual_offset(old));
                worst=std::max(worst,std::abs(error));
            }
        }
        std::cout<<"Common clock: maximum 1 s timing error="<<worst<<" ADC samples; budget="<<clock->timing_budget_samples()<<'\n';
        require(worst<clock->timing_budget_samples(),"Ensemble clock fails 3 degree relative-phase budget");
        if(argc==1)return 0;

        GpuChannelBank bank(56000000,offsets);bank.set_clock(clock);
        constexpr std::uint64_t first=7000,input_start=6000;
        constexpr std::size_t count=2048;
        std::vector<C> input(18000);
        auto unwarp=[&](double u) {
            auto s=segments.front();
            for(auto next:segments)if(u>=double(next.start)+next.offset)s=next;
            return double(s.start)+(u-double(s.start)-s.offset)/s.rate;
        };
        for(std::size_t j=0;j<input.size();++j) {
            const double n=unwarp(double(input_start+j));std::complex<double> value{};
            for(std::size_t c=0;c<offsets.size();++c)
                value+=std::polar(.2,tau*(offsets[c]+437500)*n/fs+.31*c);
            input[j]=static_cast<C>(value);
        }
        const auto output=bank.process_clocked(input,input_start,first,count);
        double phase_error=0,amplitude_error=0;
        for(std::size_t c=0;c<offsets.size();++c)for(std::size_t m=0;m<count;++m) {
            const auto expected=std::polar(.2,tau*437500*double(first+7*m)/fs+.31*c);
            const auto ratio=std::complex<double>(output[c][m])/expected;
            phase_error=std::max(phase_error,std::abs(std::arg(ratio)));
            amplitude_error=std::max(amplitude_error,std::abs(std::abs(ratio)-1));
        }
        // A separate scalar interpolation/convolution reference checks random
        // samples, not just tones that could conceal matching phase mistakes.
        for(auto& z:input)z={float(noise(random)*100),float(noise(random)*100)};
        const auto random_output=bank.process_clocked(input,input_start,first,count);
        ClockInterpolator reference((21e6+3328125)/fs);
        const auto taps=GpuChannelBank::prototype(fs);
        std::vector<C> corrected(63*7+taps.size());
        for(std::size_t j=0;j<corrected.size();++j) {
            const auto n=first-taps.size()/2+j;
            corrected[j]=reference.interpolate(input,double(n-input_start)+clock->at(double(n)).offset_at(double(n)));
        }
        double scalar_error=0;
        for(std::size_t c=0;c<offsets.size();++c) {
            const auto expected=channel_reference(corrected,taps,offsets[c],fs,first,7,64);
            for(std::size_t m=0;m<64;++m)scalar_error=std::max(scalar_error,double(std::abs(expected[m]-random_output[c][m])));
        }
        // Repeat with a different chunk boundary. Historical segments and
        // overlapping FIR supports must yield identical common-grid samples.
        const auto overlapping=bank.process_clocked(input,input_start,first+7*777,count-777);
        double boundary_error=0;
        for(std::size_t c=0;c<offsets.size();++c)for(std::size_t m=0;m<count-777;++m)
            boundary_error=std::max(boundary_error,double(std::abs(overlapping[c][m]-random_output[c][m+777])));
        std::cout<<"Three-channel raw clock warp: phase="<<phase_error*180/std::numbers::pi<<" degrees, gain="<<amplitude_error
            <<", scalar reference="<<scalar_error<<", chunk boundary="<<boundary_error<<'\n';
        require(phase_error<2e-4 && amplitude_error<2e-4,"Shared raw-IQ retiming loses channel phase or gain");
        require(scalar_error<2e-6 && boundary_error<2e-6,"Common-clock GPU disagrees with scalar reference or chunk coordinates");
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
