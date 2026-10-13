#include "pilot_phase.hpp"
#include "modem.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
// Previous sequential unwrap, retained only to demonstrate the regression.
PilotPhase sequential(std::span<const C> now,std::span<const C> anchor,std::span<const std::uint32_t> indices){
    double sw=0,sx=0,sy=0,sxx=0,sxy=0,previous=0;
    for(std::size_t j=0;j<indices.size();++j){
        const auto z=std::complex<double>(now[j])*std::conj(std::complex<double>(anchor[j]));
        const double x=double(indices[j])-3408,w=std::abs(z);
        previous+=std::remainder(std::arg(z)-previous,tau);
        sw+=w;sx+=w*x;sy+=w*previous;sxx+=w*x*x;sxy+=w*x*previous;
    }
    const double slope=(sw*sxy-sx*sy)/(sw*sxx-sx*sx);
    return {(sy-slope*sx)/sw,slope};
}
}
int main(){try{
    Modem modem;const auto indices=modem.pilot_pattern().continual;
    std::mt19937 random(91573);std::normal_distribution<float> noise(0,.025F);
    std::vector<C> anchor(indices.size()),now(indices.size());
    double old_square=0,new_square=0,worst=0,seed=0;unsigned slips=0;
    const auto start=std::chrono::steady_clock::now();
    for(unsigned n=0;n<3000;++n){
        const double delay=.008*n,phase=std::remainder(.731+.271*n,tau);
        for(std::size_t j=0;j<indices.size();++j){
            const double k=double(indices[j])-3408;
            const auto h=1.0+std::polar(.95,-tau*k*17.5/8192)+std::polar(.65,-tau*k*71.25/8192+.3);
            anchor[j]=static_cast<C>(h);
            now[j]=static_cast<C>(h*std::polar(1.0,phase+tau*k*delay/8192))+C{noise(random),noise(random)};
        }
        const auto before=sequential(now,anchor,indices),after=pilot_phase(now,anchor,indices,seed);seed=after.slope;
        const double old_error=before.slope*8192/tau-delay,new_error=after.slope*8192/tau-delay;
        slips+=std::abs(old_error)>.5;old_square+=old_error*old_error;new_square+=new_error*new_error;worst=std::max(worst,std::abs(new_error));
        if(n==1700){
            auto reverse_indices=indices;auto reverse_now=now,reverse_anchor=anchor;
            std::ranges::reverse(reverse_indices);std::ranges::reverse(reverse_now);std::ranges::reverse(reverse_anchor);
            const auto reversed=pilot_phase(reverse_now,reverse_anchor,reverse_indices,seed);
            check(std::abs(reversed.slope-after.slope)<1e-9,"Pilot fit depends on traversal order");
        }
    }
    std::cout<<"Fading pilot regression: old timing RMS="<<std::sqrt(old_square/3000)<<" samples, branch slips="<<slips
        <<"; circular RMS="<<std::sqrt(new_square/3000)<<", maximum="<<worst
        <<"; combined comparison CPU="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<" s\n";
    check(worst<.01,"Faded pilots corrupted the timing branch");
    check(slips>0 && old_square>100*new_square,"Regression did not exercise sequential unwrap failure");
    for(double phase:{-3.14,-.7,.731,3.14})for(double delay:{-.4,0.,.4,32.25}){
        for(std::size_t j=0;j<indices.size();++j){anchor[j]={.8F,.3F};now[j]=anchor[j]*static_cast<C>(std::polar(1.0,phase+tau*(int(indices[j])-3408)*delay/8192));}
        const auto fit=pilot_phase(now,anchor,indices,tau*delay/8192);
        check(std::abs(std::remainder(fit.phase-phase,tau))<1e-7 && std::abs(fit.slope*8192/tau-delay)<1e-7,"Pilot fit lost analytic phase/timing coordinates");
    }
    std::cout<<"Pilot timing branch, traversal invariance and analytic phase: PASS\n";
    std::vector<double> precision(indices.size());
    std::vector<C> white_now(indices.size()),white_anchor(indices.size());
    double unweighted_square=0,weighted_square=0;
    for(unsigned trial=0;trial<2000;++trial){
        for(std::size_t j=0;j<indices.size();++j){
            const double sigma=j%5==0 ? 10 : 1;
            anchor[j]={.8F,.3F};now[j]=anchor[j]*static_cast<C>(std::polar(1.,.731+tau*(int(indices[j])-3408)*.173/8192))+
                float(sigma)*C{noise(random),noise(random)};
            precision[j]=1/(sigma*sigma);white_now[j]=now[j]/float(sigma);white_anchor[j]=anchor[j]/float(sigma);
        }
        const auto fitted=pilot_phase(now,anchor,indices,0,precision),whitened=pilot_phase(white_now,white_anchor,indices);
        check(std::abs(fitted.slope-whitened.slope)*8192/tau<1e-7,"Pilot precision differs from explicit whitening");
        unweighted_square+=std::pow(pilot_phase(now,anchor,indices).slope*8192/tau-.173,2);
        weighted_square+=std::pow(fitted.slope*8192/tau-.173,2);
    }
    check(weighted_square<unweighted_square/10,"Known coloured pilot noise was not downweighted by its precision");
    std::cout<<"Coloured pilot noise: timing MSE improvement="<<unweighted_square/weighted_square<<" times\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
