#include "clock_estimator.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <random>

using namespace radar;
namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
using M=std::array<std::array<double,3>,3>;
void invalid_dimensions(){
    for(const auto channels:std::array<std::size_t,3>{0,33,std::numeric_limits<std::size_t>::max()}){
        bool rejected=false;
        try{ClockEstimator invalid(channels,64e6);}catch(const std::invalid_argument&){rejected=true;}
        check(rejected,"Invalid clock dimensions must be rejected before allocation");
    }
}
M inverse(M a){M out{};for(unsigned i=0;i<3;++i)out[i][i]=1;
    for(unsigned j=0;j<3;++j){const double pivot=a[j][j];for(unsigned k=0;k<3;++k){a[j][k]/=pivot;out[j][k]/=pivot;}
        for(unsigned i=0;i<3;++i)if(i!=j){const double f=a[i][j];for(unsigned k=0;k<3;++k){a[i][k]-=f*a[j][k];out[i][k]-=f*out[j][k];}}}return out;}
void polynomial_bound(){
    constexpr double fs=64e6,dt=.02,variance=4e-6;constexpr unsigned count=1000;
    ClockEstimator estimator(1,fs,{0,0,0});estimator.initialize(0,0);
    std::mt19937_64 random(37115);std::normal_distribution<double> normal;M information{};std::array<double,3> score{};
    const double mid=.5*(count+1)*dt,half=.5*(count-1)*dt;
    for(unsigned j=1;j<=count;++j){const double t=j*dt,u=(t-mid)/half;const std::array h{1.,u,u*u};
        const double y=.31+.013*t-.0002*t*t+std::sqrt(variance)*normal(random);
        estimator.observe(0,t,y,variance);
        for(unsigned i=0;i<3;++i){score[i]+=h[i]*y/variance;for(unsigned k=0;k<3;++k)information[i][k]+=h[i]*h[k]/variance;}
    }
    const auto covariance=inverse(information);std::array<double,3> coefficient{};
    for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)coefficient[i]+=covariance[i][j]*score[j];
    const auto estimate=estimator.estimate(count*dt);const std::array prediction{1.,1.,1.},rate{0.,1/half,2/half};
    auto bound=[&](auto h){double v=0;for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)v+=h[i]*covariance[i][j]*h[j];return v;};
    const double expected=coefficient[0]+coefficient[1]+coefficient[2],velocity=(coefficient[1]+2*coefficient[2])/half;
    std::cout<<"Constant-drift timing CRLB ratio="<<estimate.offset_variance/bound(prediction)
        <<", rate CRLB ratio="<<estimate.velocity_variance/bound(rate)<<", batch mean discrepancy="<<estimate.offset-expected<<'\n';
    check(std::abs(estimate.offset_variance/bound(prediction)-1)<2e-5 && std::abs(estimate.velocity_variance/bound(rate)-1)<2e-5,"Clock covariance misses independent polynomial Fisher bound");
    check(std::abs(estimate.offset-expected)<2e-7 && std::abs(estimate.velocity-velocity)<2e-7,"Clock mean differs from independent batch likelihood");
}
void stochastic_covariance(bool independent_transmitters=false){
    constexpr double fs=64e6,dt=.02,sigma=.002;constexpr ClockNoise noise{2e-11,1e-11,0};
    constexpr unsigned trials=2000,count=300;std::mt19937_64 random(199741);std::normal_distribution<double> normal;
    const double qx=std::pow(fs*noise.white_fm,2),qv=3*std::pow(fs*noise.random_walk_fm,2);
    const double l00=std::sqrt(qx*dt+qv*dt*dt*dt/3),l10=(qv*dt*dt/2)/l00,l11=std::sqrt(qv*dt-l10*l10);
    const std::array transmitter{ClockNoise{1e-11,8e-11,0},ClockNoise{2e-11,2e-11,0},ClockNoise{1e-11,4e-11,0}};
    double phase=0,frequency=0,forecast=0,nis=0;unsigned innovations=0;
    for(unsigned trial=0;trial<trials;++trial){
        ClockEstimator estimator(3,fs,noise,independent_transmitters ? std::span<const ClockNoise>{transmitter} : std::span<const ClockNoise>{});estimator.initialize(0,0);
        double x=normal(random),v=fs*1e-4*normal(random),acc=fs*1e-6*normal(random);
        std::array<double,3> bias{},rate{},wander{},wander_rate{};double mean_b=0,mean_r=0;
        auto advance_transmitter=[&](unsigned c,double dt){
            if(!independent_transmitters)return;
            const double qp=std::pow(fs*transmitter[c].white_fm,2),qf=3*std::pow(fs*transmitter[c].random_walk_fm,2);
            const double a=std::sqrt(qp*dt+qf*dt*dt*dt/3),b=(qf*dt*dt/2)/a,d=std::sqrt(qf*dt-b*b),z=normal(random);
            wander[c]+=dt*wander_rate[c]+a*z;wander_rate[c]+=b*z+d*normal(random);
        };
        for(unsigned c=0;c<3;++c){bias[c]=4096*normal(random);rate[c]=fs*1e-6*normal(random);mean_b+=bias[c]/3;mean_r+=rate[c]/3;}
        for(unsigned c=0;c<3;++c){bias[c]-=mean_b;rate[c]-=mean_r;}
        for(unsigned j=1;j<=count;++j){
            const double a=normal(random),b=normal(random);x+=v*dt+.5*acc*dt*dt+l00*a;v+=acc*dt+l10*a+l11*b;
            for(unsigned c=0;c<3;++c){const double sd=sigma*(c==j/50%3 ? 4 : 1);bias[c]+=rate[c]*dt;
                advance_transmitter(c,dt);
                const auto innovation=estimator.observe(c,j*dt,x+bias[c]+wander[c]+sd*normal(random),sd*sd);
                if(j>100){nis+=innovation.normalized_square;++innovations;}
            }
        }
        const auto out=estimator.estimate(count*dt);const double mean=(wander[0]+wander[1]+wander[2])/3,mean_rate=(wander_rate[0]+wander_rate[1]+wander_rate[2])/3;
        phase+=std::pow(out.offset-x-mean,2)/out.offset_variance;
        frequency+=std::pow(out.velocity-v-mean_rate,2)/out.velocity_variance;
        constexpr double horizon=.2;const auto future=estimator.estimate(count*dt+horizon);
        const double a=std::sqrt(qx*horizon+qv*horizon*horizon*horizon/3),b=(qv*horizon*horizon/2)/a,z=normal(random);
        x+=v*horizon+.5*acc*horizon*horizon+a*z;v+=acc*horizon+b*z+std::sqrt(qv*horizon-b*b)*normal(random);
        for(unsigned c=0;c<3;++c)advance_transmitter(c,horizon);
        forecast+=std::pow(future.offset-x-(wander[0]+wander[1]+wander[2])/3,2)/future.offset_variance;
    }
    phase/=trials;frequency/=trials;forecast/=trials;nis/=innovations;
    std::cout<<"Matched stochastic model (independent TX="<<independent_transmitters<<"): timing MSE/posterior="<<phase<<", rate="<<frequency<<", 200 ms forecast="<<forecast<<", innovation chi-square="<<nis<<'\n';
    check(phase>.9 && phase<1.1 && frequency>.9 && frequency<1.1 && forecast>.9 && forecast<1.1 && nis>.97 && nis<1.03,"Clock posterior uncertainty is not calibrated under its declared stochastic model");
}
}
int main(){try{invalid_dimensions();polynomial_bound();stochastic_covariance();stochastic_covariance(true);}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
