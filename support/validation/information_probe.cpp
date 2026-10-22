// Independent statistical probes of production estimators. Diagnostic results
// are conditional on the injected model, not a real-recording certification.
#include "modem.hpp"
#include "pilot_phase.hpp"
#include "sample_clock.hpp"
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <vector>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi;
struct Errors {
    double sum{},square{},maximum{};
    unsigned count{};
    void add(double error){sum+=error;square+=error*error;maximum=std::max(maximum,std::abs(error));++count;}
    boost::json::object json(double bound) const {
        return {{"trials",count},{"bias_samples",sum/count},{"rms_samples",std::sqrt(square/count)},
            {"maximum_samples",maximum},{"mse_over_crlb",square/count/bound}};
    }
};
double delay_information(std::span<const C> anchor,std::span<const std::uint32_t> indices,
                         std::span<const double> variance){
    double energy=0,first=0,second=0;
    for(std::size_t j=0;j<indices.size();++j){
        const double w=std::norm(std::complex<double>(anchor[j]))/variance[j];
        const double frequency=tau*(double(indices[j])-3408)/8192;
        energy+=w;first+=w*frequency;second+=w*frequency*frequency;
    }
    // Schur complement of unknown complex gain, including its phase.
    return 2*(second-first*first/energy);
}
boost::json::object pilots(double snr_db,bool coloured,unsigned anchor_count){
    Modem modem;const auto indices=modem.pilot_pattern().continual;
    std::mt19937 random(61781);std::normal_distribution<double> normal;
    std::vector<C> truth(indices.size()),anchor(indices.size()),now(indices.size()),white_before(indices.size()),white_now(indices.size());
    std::vector<double> variance(indices.size()),precision(indices.size()),estimated_precision(indices.size());
    constexpr double delay=.173,phase=.731;
    for(std::size_t j=0;j<indices.size();++j){
        const double k=double(indices[j])-3408;
        truth[j]=static_cast<C>((1.0+std::polar(.7,-tau*k*17.5/8192))*(4.0/3));
        variance[j]=std::pow(10.0,-snr_db/10)*(coloured && j%5==0 ? 100.0 : 1.0);
    }
    // With finite training and unknown per-carrier complex response, eliminating
    // that nuisance increases the bound by (1+1/M) for these equal-gain pairs.
    const double crlb=(1+(anchor_count ? 1.0/anchor_count : 0))/delay_information(truth,indices,variance);
    Errors unweighted,production,weighted;double predicted=0;
    std::gamma_distribution<double> secondary(407,1./407); // 408 independent complex observations, one fitted mean.
    constexpr unsigned trials=6000;
    for(unsigned trial=0;trial<trials;++trial){
        for(std::size_t j=0;j<indices.size();++j){
            const double sigma=std::sqrt(variance[j]/2),k=double(indices[j])-3408;
            now[j]=static_cast<C>(std::complex<double>(truth[j])*std::polar(1.0,phase+tau*k*delay/8192)+
                                  sigma*std::complex<double>{normal(random),normal(random)});
            anchor[j]=truth[j];
            if(anchor_count)anchor[j]+=static_cast<C>(sigma/std::sqrt(double(anchor_count))*std::complex<double>{normal(random),normal(random)});
            // Independent whitening comparator. Production uses secondary
            // covariance estimates, not these known true variances.
            white_now[j]=now[j]/float(std::sqrt(variance[j]));
            white_before[j]=anchor[j]/float(std::sqrt(variance[j]));
            precision[j]=1/variance[j];estimated_precision[j]=1/(variance[j]*secondary(random));
        }
        unweighted.add(pilot_phase(now,anchor,indices).slope*8192/tau-delay);
        const auto estimated=pilot_phase(now,anchor,indices,0,estimated_precision);
        production.add(estimated.slope*8192/tau-delay);predicted+=estimated.delay_variance;
        const auto gls=pilot_phase(now,anchor,indices,0,precision);
        const auto whitened=pilot_phase(white_now,white_before,indices);
        if(std::abs(gls.slope-whitened.slope)*8192/tau>1e-6)throw std::runtime_error("Pilot GLS differs from independent prewhitening");
        weighted.add(gls.slope*8192/tau-delay);
    }
    return {{"nominal_snr_db",snr_db},{"coloured_noise",coloured},{"anchor_average_count",anchor_count},
        {"continual_pilots",indices.size()},{"crlb_variance_samples",crlb},
        {"unweighted",unweighted.json(crlb)},{"production",production.json(crlb)},{"known_covariance_comparator",weighted.json(crlb)},
        {"secondary_variance_observations",408},
        {"mean_reported_conditional_delay_variance",predicted/trials},
        {"reported_variance_excludes_shared_anchor_nuisance",true},
        {"variance_ratio_95pct_relative_half_width_approx",1.96*std::sqrt(2.0/(trials-1))}};
}
boost::json::object rate_information(){
    Modem modem;const auto indices=modem.pilot_pattern().continual;constexpr unsigned epochs=64,trials=2000;
    constexpr double dt=8704/8e6,snr=100,rate=.02,delay=.173;
    std::mt19937 random(19001);std::normal_distribution<double> normal;
    std::vector<C> truth(indices.size()),anchor(indices.size()),now(indices.size());std::vector<double> variance(indices.size()),precision(indices.size());
    for(unsigned j=0;j<indices.size();++j){const double k=double(indices[j])-3408;
        truth[j]=static_cast<C>((1.+std::polar(.7,-tau*k*17.5/8192))*(4./3));variance[j]=(j%5==0 ? 100. : 1.)/snr;precision[j]=1/variance[j];}
    const double centre=(epochs-1)*dt/2;double time_information=0;for(unsigned i=0;i<epochs;++i)time_information+=std::pow(i*dt-centre,2);
    const double bound=1/(delay_information(truth,indices,variance)*time_information);
    Errors fitted;double cross=0,left=0,right=0;
    for(unsigned trial=0;trial<trials;++trial){
        for(unsigned j=0;j<indices.size();++j){const double sigma=std::sqrt(variance[j]/64);
            anchor[j]=truth[j]+static_cast<C>(sigma*std::complex<double>{normal(random),normal(random)});}
        double estimate=0,seed=0;std::array<double,epochs> error{};
        for(unsigned i=0;i<epochs;++i){
            for(unsigned j=0;j<indices.size();++j){const double k=double(indices[j])-3408,sigma=std::sqrt(variance[j]/2);
                now[j]=static_cast<C>(std::complex<double>(truth[j])*std::polar(1.,.731+.13*i+tau*k*(delay+rate*i*dt)/8192)
                    +sigma*std::complex<double>{normal(random),normal(random)});}
            const auto value=pilot_phase(now,anchor,indices,seed,precision);seed=value.slope;
            const double offset=value.slope*8192/tau;estimate+=(i*dt-centre)*offset;error[i]=offset-delay-rate*i*dt;
        }
        fitted.add(estimate/time_information-rate);
        for(unsigned i=2;i<epochs;++i){const double a=error[i-1]-error[i-2],b=error[i]-error[i-1];cross+=a*b;left+=a*a;right+=b*b;}
    }
    const auto result=fitted.json(bound);
    if(result.at("mse_over_crlb").as_double()<.9 || result.at("mse_over_crlb").as_double()>1.1)
        throw std::runtime_error("Pilot-derived rate misses its channel/intercept-nuisance Fisher bound");
    return {{"epochs",epochs},{"trials",trials},{"interval_s",(epochs-1)*dt},{"units","delay samples per second"},
        {"model","Static frequency-selective channel; unknown complex gain each epoch; one noisy 32-symbol anchor shared by every epoch; known diagonal noise covariance"},
        {"rate_crlb_variance",bound},{"rate_fit",result},{"adjacent_differential_rate_error_correlation",cross/std::sqrt(left*right)},
        {"anchor_bias_profiled_as_unknown_intercept",true},{"instantaneous_rate_variance_to_64_epoch_crlb",2*time_information/(dt*dt)}};
}
boost::json::object all_carriers(){
    Modem modem;const auto pattern=modem.pilot_pattern();
    std::vector<std::uint32_t> all(6817);std::vector<C> source(6817);std::vector<double> variance(6817,1);
    for(unsigned k=0;k<6817;++k){
        all[k]=k;
        const bool boosted=k%12==0 || std::ranges::find(pattern.continual,k)!=pattern.continual.end();
        const bool tps=std::ranges::find(pattern.tps,k)!=pattern.tps.end();
        source[k]={tps || !boosted ? 1.0F : 4.0F/3,0};
    }
    const auto full=delay_information(source,all,variance);
    std::vector<C> selected(pattern.continual.size(),C{4.F/3,0});
    const auto pilot=delay_information(selected,pattern.continual,std::span{variance}.first(selected.size()));
    return {{"model","flat known channel; unit expected data/TPS power; boosted pilots; one symbol"},
        {"continual_pilot_information",pilot},{"all_known_carrier_information",full},
        {"all_to_pilot_information_ratio",full/pilot},{"information_ratio_db",10*std::log10(full/pilot)}};
}
boost::json::object ensemble(double budget){
    constexpr double fs=64e6,occupied_span=55.65625e6,dt=.001;
    constexpr std::array centres{-24.5e6,-17.5e6,-10.375e6,17.5e6,24.5e6};
    constexpr std::array biases{-.04e-6,-.02e-6,0.0,.02e-6,.04e-6};
    SampleClock clock(fs,centres,budget);
    std::mt19937 random(49781);std::normal_distribution<double> normal;
    auto truth=[](double n){return 4e-6*n+.5*.15e-6*n*n/fs;};
    Errors error;
    for(unsigned step=1;step<=15000;++step){
        const double n=fs*step*dt,previous=n-fs*dt;
        for(unsigned c=0;c<centres.size();++c){
            const auto applied=clock.at(n);
            const double actual=truth(n)+biases[c]*n,old=truth(previous)+biases[c]*previous;
            const double applied_rate=1+(applied.offset_at(n)-clock.at(previous).offset_at(previous))/(fs*dt);
            clock.observe_rate(c,n,applied_rate/(1+(actual-old)/(fs*dt))-1);
            const double sigma=(step/700+c)%3==0 ? .03 : .002;
            if(step>=200)clock.observe_offset(c,n,(applied.offset_at(n)-actual+17*c)/applied.rate+sigma*normal(random),sigma*sigma);
        }
        if(step%20==0)clock.update(std::uint64_t(n+fs*.03));
        if(step>4000){
            const double old=n-fs*3;
            error.add((clock.at(n).offset_at(n)-clock.at(old).offset_at(old))-(truth(n)-truth(old)));
        }
    }
    return {{"requested_budget_degrees",budget},{"comparison_interval_s",3},
        {"timing_budget_adc_samples",clock.timing_budget_samples()},
        {"phase_rms_degrees",std::sqrt(error.square/error.count)*360*occupied_span/fs},
        {"phase_maximum_degrees",error.maximum*360*occupied_span/fs},
        {"timing_rms_adc_samples",std::sqrt(error.square/error.count)}};
}
}
int main(){try{
    boost::json::array timing,clock;
    for(double snr:{10.,20.,40.})timing.emplace_back(pilots(snr,false,0));
    timing.emplace_back(pilots(20,true,0));timing.emplace_back(pilots(20,true,32));
    timing.emplace_back(pilots(10,false,1));timing.emplace_back(pilots(20,false,1));timing.emplace_back(pilots(40,false,1));
    clock.emplace_back(ensemble(3));clock.emplace_back(ensemble(.1));
    std::cout<<boost::json::serialize(boost::json::object{{"pilot_likelihood",std::move(timing)},
        {"decision_directed_information",all_carriers()},{"common_clock",std::move(clock)},{"pilot_rate_information",rate_information()}})<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
