#include "gpu_noise.hpp"
#include "matched_profiles.hpp"
#include "gpu_compute.hpp"
#include "dsp.hpp"
#include "gpu_focus.hpp"
#include "matched_detector.hpp"
#include "stationary_background.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>

using namespace radar;
namespace {
constexpr unsigned bins=6817,nfft=8192;
constexpr double tau=2*std::numbers::pi;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
struct Plane {
    std::shared_ptr<gpu::Context> context;
    gpu::Buffer storage;
    gpu::BufferView view;
    Plane(std::shared_ptr<gpu::Context> c,std::size_t bytes):context(std::move(c)),storage(gpu::make_buffer(context->gpu,bytes)),
        view{context,*storage.handle,bytes,context->gpu.device->mapMemory(*storage.memory,0,bytes)}{}
    ~Plane(){context->gpu.device->unmapMemory(*storage.memory);}
    template<class T> void set(std::span<const T> values){std::copy(values.begin(),values.end(),static_cast<T*>(view.mapped));}
};
void noise(std::shared_ptr<gpu::Context> context,float scale=1,unsigned bits=2,bool fixed_pilot=false,float tone=4) {
    constexpr unsigned rows=512;const float variance=.01F*scale*scale;
    std::mt19937 rng(9854);std::normal_distribution<float> normal(0,std::sqrt(variance/2));
    std::vector<C> x(rows*bins),y(x.size()),h(bins,scale*C{.8F,.3F});std::vector<std::uint32_t> v(x.size(),1);
    const auto points=qam_constellation(bits);
    for(unsigned i=0;i<x.size();++i){x[i]=points[rng()%points.size()];y[i]=h[i%bins]*x[i]+C{normal(rng),normal(rng)};}
    for(unsigned r=0;r<rows;++r){
        const auto at=r*bins+5840;
        if(fixed_pilot){x[at]=C{4.F/3,0};y[at]=h[5840]*x[at]+C{normal(rng),normal(rng)};}
        y[at]+=scale*std::polar(tone,fixed_pilot ? float(tau*1.17*r*.001088) : 0.F);
    }
    Plane xp(context,x.size()*sizeof(C)),yp(context,y.size()*sizeof(C)),hp(context,h.size()*sizeof(C)),vp(context,v.size()*sizeof(v[0]));
    xp.set<C>(x);yp.set<C>(y);hp.set<C>(h);vp.set<std::uint32_t>(v);
    FrameBatch frame;frame.first=0;frame.last=rows;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {xp.view,vp.view,rows},{hp.view,{},rows,0,0}, {xp.view,{},{},hp.view,vp.view,rows,0}};
    GpuNoise estimator(context);const auto result=estimator.estimate(batch);
    std::vector<float> measured(static_cast<const float*>(result.mapped),static_cast<const float*>(result.mapped)+bins);
    auto ordered=measured;std::ranges::sort(ordered);
    check(std::abs(ordered[bins/2]/variance-1)<.04,"White carrier variance is biased");
    const float relative=tone/4;
    check(measured[5840]>8*scale*scale*relative*relative && measured[5840]<24*scale*scale*relative*relative,"Coherent narrowband interference was lost by noise differencing");
    std::cout<<"Noise: median="<<ordered[bins/2]<<" coherent carrier="<<measured[5840]<<" fixed_pilot="<<fixed_pilot<<'\n';
}
void finite(std::shared_ptr<gpu::Context> context,unsigned guard=16) {
    constexpr unsigned rows=259,delay=700;const unsigned cp=8192/guard;constexpr double origin=128.375,frequency=200;
    const C channel{.8F,.3F},amplitude=std::polar(.02F,.73F);
    std::mt19937 rng(63214);const auto points=qam_constellation(2);
    std::vector<C> s(rows*bins),aligned(s.size()),y(s.size()),h(bins,channel),stationary(bins),useful(rows*nfft);
    for(unsigned k=0;k<bins;++k)stationary[k]=static_cast<C>(std::polar(.4,-tau*(int(k)-3408)*180.25/nfft+.31));
    std::vector<std::uint32_t> v(s.size(),1);std::vector<float> d(bins,1);
    Fft fft(nfft);std::vector<C> work(nfft);
    for(unsigned r=0;r<rows;++r) {
        std::fill(work.begin(),work.end(),C{});
        for(unsigned k=0;k<bins;++k) {
            s[r*bins+k]=points[rng()%4];
            aligned[r*bins+k]=s[r*bins+k]*static_cast<C>(std::polar(1.0,-tau*(int(k)-3408)*origin/nfft));
            work[(k+nfft-3408)%nfft]=s[r*bins+k]*static_cast<C>(std::polar(1.0,-tau*(int(k)-3408)*.375/nfft));
        }
        fft.transform(work,true);std::copy(work.begin(),work.end(),useful.begin()+r*nfft);
    }
    auto source=[&](unsigned r,int coordinate) {
        if(coordinate < -int(cp)){--r;coordinate+=nfft+cp;}
        return useful[r*nfft+(coordinate+int(nfft))%nfft];
    };
    for(unsigned r=1;r<rows;++r) {
        for(unsigned j=0;j<nfft;++j)work[j]=amplitude*source(r,int(j)-128-delay)*
            static_cast<C>(std::polar(1.0,tau*frequency*(double(j)-4095.5)/8e6));
        fft.transform(work);
        for(unsigned k=0;k<bins;++k)y[r*bins+k]=channel*((C{1,0}+stationary[k])*aligned[r*bins+k]+work[(k+nfft-3408)%nfft]);
    }
    std::copy_n(aligned.begin(),bins,y.begin());for(unsigned k=0;k<bins;++k)y[k]*=channel;
    Plane sp(context,s.size()*sizeof(C)),xp(context,s.size()*sizeof(C)),yp(context,s.size()*sizeof(C)),
        hp(context,h.size()*sizeof(C)),vp(context,v.size()*sizeof(v[0])),dp(context,d.size()*sizeof(float));
    sp.set<C>(s);xp.set<C>(aligned);yp.set<C>(y);hp.set<C>(h);vp.set<std::uint32_t>(v);dp.set<float>(d);
    FrameBatch frame;frame.first=1;frame.last=rows;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {sp.view,vp.view,rows},{hp.view,{},rows,origin,0},
        {xp.view,{},{},hp.view,vp.view,rows,origin}};
    GpuMatchedProfiles processor(context);ComputeTimings timing;
    const auto result=processor.calculate(batch,dp.view,guard,&timing,true,stationary);
    const double gain=std::norm(channel)*bins/nfft;
    for(unsigned r=1;r<rows;++r) {
        std::fill(work.begin(),work.end(),C{});
        for(unsigned k=0;k<bins;++k)work[(k+nfft-3408)%nfft]=(y[r*bins+k]-channel*(C{1,0}+stationary[k])*aligned[r*bins+k])*std::conj(channel);
        fft.transform(work,true);
        for(int lag:{-64,-32,0,16,128,384,600,700,800,928})for(unsigned moment=0;moment<2;++moment) {
            std::complex<double> expected{};
            for(unsigned j=0;j<nfft;++j)expected+=std::complex<double>(work[j])*std::conj(std::complex<double>(source(r,int(j)-128-lag)))*
                (moment ? (double(j)-4095.5)/nfft : 1);
            expected/=gain;
            const auto index=lag<0 ? matched_profile_size+lag : lag;
            const auto actual=result.values[(moment*result.count+r-1)*matched_profile_size+index];
            check(std::abs(std::complex<double>(actual)-expected)<2e-6,"Finite correlation, fractional reference phase or Doppler moment wrong");
        }
    }
    std::cout<<"Finite CP context, frozen stationary subtraction, fractional origin, complex phase and Doppler moments: PASS; GPU "<<timing.gpu*1000<<" ms\n";
}
void stationary() {
    for(float scale:{.01F,1.F,100.F}){
        std::vector<C> h(bins,scale*C{.8F,.3F}),expected(bins);
        std::vector<ResponseStatistics> statistics(bins);
        for(unsigned k=0;k<bins;++k){
            expected[k]=static_cast<C>(std::polar(.25,-tau*(int(k)-3408)*200.5/nfft));
            statistics[k]={1000.F*h[k]*(C{1,0}+expected[k]),1000,1000.F*std::norm(h[k])};
        }
        FrameBatch frame;frame.first=100;frame.last=1000;
        SpectralBatch batch{frame,{}, {}, {{},{},1020,0,0},{}};
        batch.estimate.response={{},{},h.size()*sizeof(C),h.data()};
        batch.estimate.statistics={{},{},statistics.size()*sizeof(ResponseStatistics),statistics.data()};
        StationaryBackground predictor;
        for(unsigned i=0;i<3;++i)predictor.observe(batch,.001088);
        check(!predictor.ready(),"Stationary estimate was released before independent training completed");
        predictor.observe(batch,.001088);check(predictor.ready(),"Stationary estimate did not complete");
        for(unsigned k=0;k<bins;++k)check(std::abs(predictor.response()[k]-expected[k])<3e-7F,"Stationary transfer lost phase or magnitude scaling");
        const std::vector<C> frozen(predictor.response().begin(),predictor.response().end());
        for(auto& statistic:statistics)statistic.cross*=C{0,2};
        predictor.observe(batch,.001088);
        check(std::ranges::equal(frozen,predictor.response()),"Future targets changed the frozen stationary estimate");
        predictor.reset();check(!predictor.ready(),"Stationary estimate survived an acquisition reset");
        statistics.back().reference_power=0;
        for(unsigned i=0;i<4;++i)predictor.observe(batch,.001088);
        check(!predictor.ready(),"Incomplete decoded source completed stationary training");
        statistics.back().reference_power=1000;
        for(unsigned i=0;i<2;++i)predictor.observe(batch,.001088);
        check(predictor.ready(),"Stationary training did not recover when decoded source became available");
    }
    std::cout<<"Independent stationary training, freeze, reset and 80 dB magnitude span: PASS\n";
}
void calibration() {
    for(bool colored:{false,true}){
        DopplerMap map;map.ranges=802;map.frequencies=257;map.range_step=299792458.0/8e6;
        map.frequency_start=-200;map.frequency_step=400.0/256;map.integration=3;
        map.values.resize(map.ranges*map.frequencies);
        std::mt19937 rng(68495);std::normal_distribution<float> normal(0,std::sqrt(.5F));
        for(std::size_t r=0;r<map.ranges;++r)for(std::size_t f=0;f<map.frequencies;++f){
            const double power=colored ? std::exp(.7*std::sin(tau*f/256)+.15*std::cos(tau*r/802)) : 1;
            auto& value=map.values[r*map.frequencies+f];value=std::sqrt(float(power))*C{normal(rng),normal(rng)};
            if(rng()%50==0)value+=std::polar(20.F,float(rng()%1000));
        }
        estimate_map_noise(map);
        std::vector<double> errors;double mean=0;
        for(std::size_t f=0;f<map.frequencies;++f){
            double expected=0;std::size_t count=0;
            for(std::size_t r=17;r<map.ranges;++r){
                const double power=colored ? std::exp(.7*std::sin(tau*f/256)+.15*std::cos(tau*r/802)) : 1;
                expected+=power/(map.noise_range[r]*map.noise_doppler[f]);++count;
            }
            errors.push_back(10*std::log10(expected/count));mean+=errors.back()/map.frequencies;
        }
        double rms=0;for(double error:errors)rms+=(error-mean)*(error-mean)/errors.size();rms=std::sqrt(rms);
        check(rms<(colored ? .65 : .35) && std::abs(mean)<.75,"Background calibration overfit sampling noise or sparse targets");
        std::cout<<"Independent expected-noise calibration RMS dB, colored="<<colored<<": "<<rms<<'\n';
    }
}
void focus(std::shared_ptr<gpu::Context> context,double rf=557625000,bool full_cpi=false,bool remove_dc=false,bool weighted=false) {
    const unsigned rows=full_cpi ? 2758 : 512,size=full_cpi ? 11520 : 2048,tone=full_cpi ? 2305 : 401;
    constexpr double dt=.001088,delay=167.375;
    const double fd=tone/(size*dt),mid=(rows-1)*dt*.5,x=fd*.001024;
    const double sinc=std::sin(std::numbers::pi*x)/(std::numbers::pi*x);
    const double derivative=(std::numbers::pi*x*std::cos(std::numbers::pi*x)-std::sin(std::numbers::pi*x))/(std::numbers::pi*x*x);
    const C amplitude=std::polar(.17F,1.23F);
    std::vector<double> information(rows,1);double total=0;
    if(weighted)for(unsigned n=0;n<rows;++n)information[n]=n<rows/3 ? 1 : n<2*rows/3 ? .1 : 10;
    for(double value:information)total+=value;
    std::vector<double> weights(rows);for(unsigned n=0;n<rows;++n)weights[n]=information[n]/total*rows;
    std::vector<C> moments(2*rows*1024),work(1024);Fft fft(1024);
    for(unsigned r=0;r<rows;++r){
        const double t=r*dt-mid,d=delay-fd*8e6/rf*t;
        std::fill(work.begin(),work.end(),C{});
        for(int k=-400;k<=400;++k)work[(k+1024)%1024]=amplitude*static_cast<C>(std::polar(1024.0/801,tau*fd*t-tau*k*d/1024));
        fft.transform(work,true);
        for(unsigned k=0;k<1024;++k){
            moments[r*1024+k]=work[k]*float(sinc);
            moments[(rows+r)*1024+k]=work[k]*C{0,float(-derivative/tau)};
            if(remove_dc){
                const float static_shape=std::exp(-std::pow((float(k)-20)/10,2));
                moments[r*1024+k]+=C{2,-3}*static_shape;
                moments[(rows+r)*1024+k]+=C{.2F,.3F}*static_shape;
            }
        }
    }
    GpuFocus processor(context);ComputeTimings timing;
    auto result=processor.calculate(moments,rows,dt,rf,5,{8000,200,remove_dc},&timing,information);
    const int half=int((result.frequencies-1)/2);
    double worst=0;
    for(int bin:{-400,-37,int(tone),int(tone+29)})for(unsigned range:{660,669,670,679}){
        std::complex<double> expected{};const double f=bin/(size*dt),rr=range*.25;
        for(int k=-400;k<=400;++k){
            std::complex<double> temporal{},source_mean{},template_sum{};
            for(unsigned n=0;n<rows;++n){
                const double t=n*dt-mid;
                temporal+=weights[n]*std::polar(1.0,tau*((fd-f)+k*8e6/(1024*rf)*(fd-f))*t);
                if(remove_dc){
                    const double scale=1+k*8e6/(1024*rf);
                    source_mean+=weights[n]*std::polar(1.0,tau*fd*scale*t);template_sum+=weights[n]*std::polar(1.0,-tau*f*scale*t);
                }
            }
            if(remove_dc)temporal-=source_mean*template_sum/double(rows);
            expected+=temporal*std::polar(1.0,tau*k*(rr-delay)/1024);
        }
        expected*=std::complex<double>(amplitude)/double(rows*801);
        const double fx=f*.001024,a=std::numbers::pi*fx;
        const double gain=std::abs(a)<1e-8 ? 1 : (2*std::sin(a)-a*std::cos(a))/a;
        expected*=(sinc-f*.001024*derivative)/gain;
        const auto actual=result.values[range*result.frequencies+bin+half];
        worst=std::max(worst,std::abs(std::complex<double>(actual)-expected));
    }
    std::cout<<"Constant-velocity focus complex error="<<worst<<" weighted="<<weighted<<" GPU="<<timing.gpu*1000<<" ms\n";
    check(worst<8e-5,"Range migration interpolation or centred phase epoch is wrong");
    check(std::abs(result.phase_epoch-(5+mid+4095.5/8e6))<1e-12,"Focus epoch omits within-symbol timing");
    // Scalar background calibration must be reversible without changing phase.
    estimate_map_noise(result);const auto original=result.values;
    standardize_map(result);double roundtrip=0,phase_error=0;
    for(std::size_t r=0;r<result.ranges;++r)for(std::size_t f=0;f<result.frequencies;++f){
        const auto i=r*result.frequencies+f;
        const auto recovered=result.values[i]*std::sqrt(result.noise_range[r]*result.noise_doppler[f]);
        roundtrip=std::max(roundtrip,double(std::abs(recovered-original[i])));
        if(std::abs(original[i])>1e-6F)phase_error=std::max(phase_error,std::abs(double(std::arg(recovered*std::conj(original[i])))));
    }
    check(roundtrip<1e-7 && phase_error<1e-6,"Noise standardization changed complex phase or is not reversible to float precision");
    check(result.json().find("noise_standard_deviation")!=std::string::npos,"Noise units missing from serialized metadata");
}
void noise_steps(std::shared_ptr<gpu::Context> context){
    constexpr unsigned rows=512;std::mt19937 random(7854);const auto points=qam_constellation(2);
    std::vector<C> source(rows*bins),measured(source.size()),response(bins,C{.8F,.3F});std::vector<std::uint32_t> valid(source.size(),1);
    for(auto& value:source)value=points[random()%4];
    Plane sp(context,source.size()*sizeof(C)),yp(context,source.size()*sizeof(C)),hp(context,response.size()*sizeof(C)),vp(context,valid.size()*sizeof(valid[0]));
    sp.set<C>(source);hp.set<C>(response);vp.set<std::uint32_t>(valid);
    FrameBatch frame;frame.last=rows;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {sp.view,vp.view,rows},{hp.view,{},rows,0,0},{sp.view,{},{},hp.view,vp.view,rows,0}};
    GpuNoise estimator(context);
    for(float variance:{.01F,.1F,.001F}){
        std::normal_distribution<float> normal(0,std::sqrt(variance/2));
        for(unsigned i=0;i<source.size();++i)measured[i]=response[i%bins]*source[i]+C{normal(random),normal(random)};
        for(unsigned r=0;r<rows;++r)measured[r*bins+5840]+=C{4,0};yp.set<C>(measured);
        const auto result=estimator.estimate(batch);const auto* values=static_cast<const float*>(result.mapped);
        std::vector<float> ordered(values,values+bins);std::ranges::sort(ordered);
        check(std::abs(ordered[bins/2]/variance-1)<.12,"Carrier weights failed to track a common noise-level step");
        check(values[5840]>8,"Noise-level adaptation underestimated a stable narrow interferer");
        std::cout<<"Noise step true/median="<<variance<<" / "<<ordered[bins/2]<<'\n';
    }
}
void owned_focus(std::shared_ptr<gpu::Context> context){
    constexpr unsigned rows=512,delay=100;
    std::mt19937 random(46059);std::normal_distribution<float> normal(0,.001F);
    const auto points=qam_constellation(2);
    std::vector<C> source(rows*bins),thermal(source.size()),measured(source.size()),response(bins,C{1,0});
    std::vector<std::uint32_t> valid(source.size(),1);
    for(unsigned i=0;i<source.size();++i){source[i]=points[random()%4];thermal[i]={normal(random),normal(random)};}
    Plane sp(context,source.size()*sizeof(C)),yp(context,source.size()*sizeof(C)),hp(context,response.size()*sizeof(C)),vp(context,valid.size()*sizeof(valid[0]));
    FrameBatch frame;frame.first=1;frame.last=rows;frame.carriers.measured=yp.view;frame.carriers.count=rows;frame.symbols.resize(rows);
    SpectralBatch batch{frame,{}, {sp.view,vp.view,rows},{hp.view,{},rows,0,0},{sp.view,{},{},hp.view,vp.view,rows,0}};
    SarCapture capture{56'000'000,7,0,553000000,3,{{557625000,4625000,16},{564625000,11625000,16}}};
    auto fill=[&](unsigned channel){
        frame.channel=channel;
        for(unsigned r=0;r<rows;++r){frame.symbols[r].channel=channel;frame.symbols[r].segment=frame.segment;frame.symbols[r].window.sample=r*8704;}
        for(unsigned i=0;i<source.size();++i){
            const auto echo=std::polar(.02F,float(.73+1.12*channel-tau*(int(i%bins)-3408)*delay/nfft));
            measured[i]=source[i]*(C{1,0}+echo)+thermal[i];
        }
        sp.set<C>(source);yp.set<C>(measured);hp.set<C>(response);vp.set<std::uint32_t>(valid);
    };
    auto poison=[&]{
        std::fill_n(static_cast<C*>(sp.view.mapped),source.size(),C{});
        std::fill_n(static_cast<C*>(yp.view.mapped),source.size(),C{});
        std::fill_n(static_cast<C*>(hp.view.mapped),response.size(),C{});
        std::fill_n(static_cast<std::uint32_t*>(vp.view.mapped),valid.size(),0U);
    };
    DetectionOptions options;options.integration_seconds=.5;options.focus={8000,200,false};
    MatchedDetector detector(context,2,options);DetectionTimings timing;
    const auto owner=std::this_thread::get_id();std::vector<DopplerMap> maps(2);unsigned count=0;
    auto consume=[&](std::size_t channel,DopplerMap&& map){
        check(std::this_thread::get_id()==owner,"Async focus invoked a consumer on its worker thread");
        check(map.sequence==1 && map.channel==channel,"Async focus reordered map metadata");
        maps[channel]=std::move(map);++count;
    };
    for(unsigned channel=0;channel<2;++channel){fill(channel);detector.append(SarBatch(capture,batch,{{0,0,1}}),consume,&timing);poison();}
    detector.finish(consume,&timing);detector.finish(consume,&timing);
    check(count==2 && timing.focus.batches==2,"Async finish lost, duplicated or mistimed a map");
    const auto zero=maps[0].frequencies/2;
    const auto a=maps[0].values[4*delay*maps[0].frequencies+zero],b=maps[1].values[4*delay*maps[1].frequencies+zero];
    check(std::abs(a)> .019 && std::abs(b)>.019,"Async focus retained borrowed buffers instead of owned profiles");
    check(std::abs(maps[0].phase_epoch-maps[1].phase_epoch)<1e-12,"Async channels acquired different phase epochs");
    check(std::abs(std::arg(b*std::conj(a))-1.12)<.003,"Async scheduling changed interchannel phase");
    ++frame.segment;fill(0);detector.append(SarBatch(capture,batch,{{0,0,1}}),consume);
    bool failed=false;
    try{detector.finish([](std::size_t,DopplerMap&&){throw std::runtime_error("sink failed");});}
    catch(const std::runtime_error& error){failed=std::string_view(error.what())=="sink failed";}
    check(failed,"Async map consumer failure was swallowed");
    capture.channels[0].rf_hz=50e6;++frame.segment;fill(0);
    detector.append(SarBatch(capture,batch,{{0,0,1}}),consume);
    failed=false;try{detector.finish(consume);}catch(const std::invalid_argument&){failed=true;}
    check(failed,"Async numerical failure was swallowed");capture.channels[0].rf_hz=557625000;
    options.standardize_noise=true;MatchedDetector normalized(context,2,options);DetectionTimings normalized_time;
    fill(0);unsigned normalized_count=0;
    auto normalized_sink=[&](std::size_t,DopplerMap&& map){
        check(map.amplitude_units=="noise_standard_deviation","Async normalization omitted output units");
        const auto zero=map.frequencies/2,index=4*delay*map.frequencies+zero;
        const auto recovered=map.values[index]*std::sqrt(map.noise_range[4*delay]*map.noise_doppler[zero]);
        check(std::abs(recovered-a)<2e-6,"Async normalization changed the physical complex map");++normalized_count;
    };
    normalized.append(SarBatch(capture,batch,{{0,0,1}}),normalized_sink,&normalized_time);poison();normalized.finish(normalized_sink,&normalized_time);
    check(normalized_count==1 && normalized_time.standardization_cpu>0,"Async normalization was lost or mistimed");
    std::cout<<"Owned async profiles, interchannel phase, caller-thread delivery, drain, normalization and failures: PASS\n";
}
}
int main(){try{
    auto context=std::make_shared<gpu::Context>();
    std::cout<<"Storage buffer limit: "<<context->gpu.physical.getProperties().limits.maxStorageBufferRange<<'\n';
    stationary();calibration();noise(context);noise(context,.01F);noise(context,100);noise(context,1,4);noise(context,1,6);
    noise(context,1,2,true);
    noise(context,1,2,false,10000);
    finite(context);finite(context,32);finite(context,4);
    focus(context);focus(context,191625000);focus(context,191625000,true);focus(context,191625000,true,true);
    focus(context,191625000,true,true,true);noise_steps(context);
    owned_focus(context);
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
