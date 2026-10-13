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
#include <future>

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
void finite(std::shared_ptr<gpu::Context> context,unsigned guard=16,bool pilots=false,bool drifting=false,unsigned rows=259) {
    constexpr unsigned delay=700;const unsigned cp=8192/guard;constexpr double origin=128.375,frequency=200;
    const C channel{.8F,.3F},amplitude=std::polar(.02F,.73F);
    std::mt19937 rng(63214);const auto points=qam_constellation(2);
    std::vector<C> s(rows*bins),aligned(s.size()),y(s.size()),h(bins,channel),stationary(bins),useful(rows*nfft),previous(rows*nfft);
    std::vector<float> offsets(rows);std::vector<double> local(rows,origin);
    for(unsigned row=0;row<rows;++row){offsets[row]=drifting ? std::remainder(.37+.002*row,1.) : 0;local[row]-=offsets[row];}
    for(unsigned k=0;k<bins;++k)stationary[k]=static_cast<C>(std::polar(.4,-tau*(int(k)-3408)*180.25/nfft+.31));
    std::vector<std::uint32_t> v(s.size(),1);std::vector<float> d(bins,1);
    Fft fft(nfft);std::vector<C> work(nfft);
    for(unsigned r=0;r<rows;++r) {
        std::fill(work.begin(),work.end(),C{});
        for(unsigned k=0;k<bins;++k) {
            s[r*bins+k]=points[rng()%4];
            if(pilots && k%12==r%4*3)s[r*bins+k]={rng()%2 ? 4.F/3 : -4.F/3,0};
            aligned[r*bins+k]=s[r*bins+k]*static_cast<C>(std::polar(1.0,-tau*(int(k)-3408)*local[r]/nfft));
            work[(k+nfft-3408)%nfft]=s[r*bins+k]*static_cast<C>(std::polar(1.0,-tau*(int(k)-3408)*(local[r]-std::floor(local[r]))/nfft));
        }
        fft.transform(work,true);std::copy(work.begin(),work.end(),useful.begin()+r*nfft);
    }
    for(unsigned r=1;r<rows;++r){
        std::fill(work.begin(),work.end(),C{});
        for(unsigned k=0;k<bins;++k)work[(k+nfft-3408)%nfft]=s[(r-1)*bins+k]*static_cast<C>(std::polar(1.,-tau*(int(k)-3408)*(local[r]-std::floor(local[r]))/nfft));
        fft.transform(work,true);std::copy(work.begin(),work.end(),previous.begin()+r*nfft);
    }
    auto source=[&](unsigned r,int coordinate) {
        if(coordinate < -int(cp))return previous[r*nfft+(coordinate+int(nfft+cp))%nfft];
        return useful[r*nfft+(coordinate+int(nfft))%nfft];
    };
    for(unsigned r=1;r<rows;++r) {
        for(unsigned j=0;j<nfft;++j)work[j]=amplitude*source(r,int(j)-int(std::floor(local[r]))-delay)*
            static_cast<C>(std::polar(1.0,tau*frequency*(double(j)-4095.5)/8e6));
        fft.transform(work);
        for(unsigned k=0;k<bins;++k)y[r*bins+k]=channel*((C{1,0}+stationary[k])*aligned[r*bins+k]+work[(k+nfft-3408)%nfft]);
    }
    std::copy_n(aligned.begin(),bins,y.begin());for(unsigned k=0;k<bins;++k)y[k]*=channel;
    Plane sp(context,s.size()*sizeof(C)),xp(context,s.size()*sizeof(C)),yp(context,s.size()*sizeof(C)),
        hp(context,h.size()*sizeof(C)),vp(context,v.size()*sizeof(v[0])),dp(context,d.size()*sizeof(float)),op(context,rows*sizeof(float));
    sp.set<C>(s);xp.set<C>(aligned);yp.set<C>(y);hp.set<C>(h);vp.set<std::uint32_t>(v);dp.set<float>(d);
    op.set<float>(offsets);
    FrameBatch frame;frame.first=1;frame.last=rows;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {sp.view,vp.view,rows},{hp.view,{},rows,origin,0,op.view},
        {xp.view,{},{},hp.view,vp.view,rows,origin}};
    GpuMatchedProfiles processor(context);ComputeTimings timing;
    const auto result=processor.calculate(batch,dp.view,guard,&timing,true,stationary);
    const double gain=std::norm(channel)*bins/nfft;
    double worst=0,phase=0;
    for(unsigned r=1;r<rows;++r) {
        std::fill(work.begin(),work.end(),C{});
        for(unsigned k=0;k<bins;++k)work[(k+nfft-3408)%nfft]=(y[r*bins+k]-channel*(C{1,0}+stationary[k])*aligned[r*bins+k])*std::conj(channel);
        fft.transform(work,true);
        for(int lag:{-64,-32,0,16,128,384,600,700,800,928})for(unsigned moment=0;moment<2;++moment) {
            std::complex<double> expected{};
            for(unsigned j=0;j<nfft;++j)expected+=std::complex<double>(work[j])*std::conj(std::complex<double>(source(r,int(j)-int(std::floor(local[r]))-lag)))*
                (moment ? (double(j)-4095.5)/nfft : 1);
            expected/=gain;
            const auto index=lag<0 ? matched_profile_size+lag : lag;
            const auto actual=result.values[(moment*result.count+r-1)*matched_profile_size+index];
            worst=std::max(worst,std::abs(std::complex<double>(actual)-expected));
            if(lag==delay && moment==0)phase=std::max(phase,std::abs(std::arg(std::complex<double>(actual)/expected))*180/std::numbers::pi);
            check(worst<(drifting ? 2e-5 : 2e-6),"Finite correlation, fractional reference phase or Doppler moment wrong");
        }
    }
    check(phase<.37,"Source-clock bookkeeping exceeded the phase budget");
    std::cout<<"Finite CP context, source drift="<<drifting<<", maximum complex error="<<worst<<", phase error="<<phase<<" degrees; GPU "<<timing.gpu*1000<<" ms\n";
}
void stationary() {
    for(float scale:{.01F,1.F,100.F}){
        std::vector<C> h(bins,scale*C{.8F,.3F}),expected(bins);
        std::vector<CarrierTraining> training(bins);
        for(unsigned k=0;k<bins;++k){
            expected[k]=static_cast<C>(std::polar(.25,-tau*(int(k)-3408)*200.5/nfft));
            training[k]={h[k]*(C{1,0}+expected[k]),900,scale*scale*.01F};
        }
        FrameBatch frame;frame.first=100;frame.last=1000;
        SpectralBatch batch{frame,{}, {}, {{},{},1020,0,0},{}};
        batch.estimate.response={{},{},h.size()*sizeof(C),h.data()};
        StationaryBackground predictor;
        for(unsigned i=0;i<3;++i)predictor.observe(batch,.001088,training);
        check(!predictor.ready(),"Stationary estimate was released before independent training completed");
        predictor.observe(batch,.001088,training);check(predictor.ready(),"Stationary estimate did not complete");
        for(unsigned k=0;k<bins;++k)check(std::abs(predictor.response()[k]-expected[k])<3e-7F,"Stationary transfer lost phase or magnitude scaling");
        check(std::abs(predictor.variance()[0]/(.01/(4*900*.73))-1)<1e-6,"Frozen predictor lost its conditional covariance");
        const std::vector<C> frozen(predictor.response().begin(),predictor.response().end());
        for(auto& statistic:training)statistic.transfer*=C{0,2};
        predictor.observe(batch,.001088,training);
        check(std::ranges::equal(frozen,predictor.response()),"Future targets changed the frozen stationary estimate");
        predictor.reset();check(!predictor.ready(),"Stationary estimate survived an acquisition reset");
        training.back().energy=0;
        for(unsigned i=0;i<4;++i)predictor.observe(batch,.001088,training);
        check(!predictor.ready(),"Incomplete decoded source completed stationary training");
        training.back().energy=900;
        for(unsigned i=0;i<2;++i)predictor.observe(batch,.001088,training);
        check(predictor.ready(),"Stationary training did not recover when decoded source became available");
    }
    std::cout<<"Independent stationary training, freeze, reset and 80 dB magnitude span: PASS\n";
}
void cancellation(std::shared_ptr<gpu::Context> context){
    constexpr unsigned rows=1020,first=128,last=896;
    std::vector<C> x(rows*bins),y(x.size()),h(bins,C{.8F,.3F}),truth(bins);
    std::vector<std::uint32_t> valid(x.size(),1);
    std::mt19937 rng(620415);std::normal_distribution<float> normal;
    const auto points=qam_constellation(6);
    for(auto& value:x)value=points[rng()%points.size()];
    for(unsigned k=0;k<bins;++k)truth[k]=static_cast<C>(std::polar(.25,.31-tau*(int(k)-3408)*17.5/8192));
    Plane xp(context,x.size()*sizeof(C)),yp(context,y.size()*sizeof(C)),hp(context,h.size()*sizeof(C)),vp(context,valid.size()*sizeof(valid[0]));
    xp.set<C>(x);hp.set<C>(h);vp.set<std::uint32_t>(valid);
    FrameBatch frame;frame.first=first;frame.last=last;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {xp.view,vp.view,rows},{hp.view,{},rows,0,0},{xp.view,{},{},hp.view,vp.view,rows,0}};
    GpuNoise noise(context);StationaryBackground background;
    for(double variance:{.01,.1,.001,.05}){
        for(unsigned row=0;row<rows;++row)for(unsigned k=0;k<bins;++k){
            const auto at=row*bins+k;
            y[at]=row<first || row>=last ? C{1000,-2000} : h[k]*(C{1,0}+truth[k])*x[at]+float(std::sqrt(variance/2))*C{normal(rng),normal(rng)};
        }
        yp.set<C>(y);noise.estimate(batch);background.observe(batch,.001088,noise.training());
    }
    check(background.ready(),"Unique-row calibration never completed");
    double normalized=0;
    for(unsigned k=0;k<bins;++k)normalized+=std::norm(std::complex<double>(background.response()[k])-std::complex<double>(truth[k]))/background.variance()[k];
    normalized/=bins;
    check(std::abs(normalized-1)<.06,"Cancellation covariance disagrees with independent carrier trials");
    std::cout<<"Unique training with contaminated context and 20 dB noise steps: error power / predicted variance="<<normalized<<'\n';
}
void focus(std::shared_ptr<gpu::Context> context,double rf=557625000,bool full_cpi=false,bool remove_dc=false,bool weighted=false,bool erasures=false) {
    const unsigned rows=full_cpi ? 2758 : 512,size=full_cpi ? 11520 : 2048,tone=full_cpi ? 2305 : 401;
    constexpr double dt=.001088,delay=167.375;
    const double fd=tone/(size*dt),mid=(rows-1)*dt*.5,x=fd*.001024;
    const double sinc=std::sin(std::numbers::pi*x)/(std::numbers::pi*x);
    const double derivative=(std::numbers::pi*x*std::cos(std::numbers::pi*x)-std::sin(std::numbers::pi*x))/(std::numbers::pi*x*x);
    const C amplitude=std::polar(.17F,1.23F);
    std::vector<double> information(rows,1);double total=0;
    if(weighted)for(unsigned n=0;n<rows;++n)information[n]=n<rows/3 ? 1 : n<2*rows/3 ? .1 : 10;
    if(erasures)for(unsigned n=0;n<rows;++n)if(n%43==0 || (n>=rows/2 && n<rows/2+16))information[n]=0;
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
    check(result.amplitude_units=="complex_reflectivity" && result.noise_range.empty() && result.noise_doppler.empty(),
          "Focus applied an empirical map normalization");
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
    float previous=0;
    for(float variance:{.01F,.1F,.001F}){
        std::normal_distribution<float> normal(0,std::sqrt(variance/2));
        for(unsigned i=0;i<source.size();++i)measured[i]=response[i%bins]*source[i]+C{normal(random),normal(random)};
        for(unsigned r=0;r<rows;++r)measured[r*bins+5840]+=C{4,0};yp.set<C>(measured);
        auto result=estimator.estimate(batch);const auto* values=static_cast<const float*>(result.mapped);
        std::vector<float> ordered(values,values+bins);std::ranges::sort(ordered);
        if(previous)check(std::abs(ordered[bins/2]/previous-1)<.12,"Primary noise changed its own weights");
        result=estimator.estimate(batch);values=static_cast<const float*>(result.mapped);
        ordered.assign(values,values+bins);std::ranges::sort(ordered);previous=variance;
        check(std::abs(ordered[bins/2]/variance-1)<.12,"Carrier weights failed to track a common noise-level step");
        check(values[5840]>8,"Noise-level adaptation underestimated a stable narrow interferer");
        std::cout<<"Noise step true/median="<<variance<<" / "<<ordered[bins/2]<<'\n';
    }
}
void covariance_efficiency(std::shared_ptr<gpu::Context> context){
    constexpr unsigned rows=512;std::mt19937 random(13657);std::normal_distribution<float> normal(0,1);
    const auto points=qam_constellation(6);
    std::vector<C> source(rows*bins),measured(source.size()),response(bins);
    std::vector<float> truth(bins);std::vector<std::uint32_t> valid(source.size(),1);
    for(auto& z:source)z=points[random()%points.size()];
    for(unsigned k=0;k<bins;++k)response[k]=static_cast<C>(std::exp(.3*std::polar(1.0,-tau*(int(k)-3408)/8192)));
    Plane sp(context,source.size()*sizeof(C)),yp(context,source.size()*sizeof(C)),hp(context,response.size()*sizeof(C)),vp(context,valid.size()*sizeof(valid[0]));
    sp.set<C>(source);hp.set<C>(response);vp.set<std::uint32_t>(valid);
    FrameBatch frame;frame.last=rows;frame.carriers.measured=yp.view;
    SpectralBatch batch{frame,{}, {sp.view,vp.view,rows},{hp.view,{},rows,0,0},{sp.view,{},{},hp.view,vp.view,rows,0}};
    for(unsigned model=0;model<3;++model){
        for(unsigned k=0;k<bins;++k)truth[k]=model==1 ? float(.0001*std::pow(10.,2*std::pow((int(k)-3408)/3408.,2))) : .0001F;
        if(model==2)truth[5840]=10000;
        for(unsigned i=0;i<source.size();++i){
            measured[i]=source[i]*response[i%bins]+std::sqrt(truth[i%bins]/2)*C{normal(random),normal(random)};
        }
        yp.set<C>(measured);GpuNoise estimator(context);const auto trained=estimator.estimate(batch);
        std::vector<float> prediction(static_cast<const float*>(trained.mapped),static_cast<const float*>(trained.mapped)+bins);
        // A new, strong moving echo must not determine its own carrier weights.
        for(unsigned r=0;r<rows;++r)for(unsigned k=0;k<bins;++k)
            measured[r*bins+k]+=source[r*bins+k]*static_cast<C>(std::polar(.7,tau*(37*r*.001088-(int(k)-3408)*180./8192)));
        yp.set<C>(measured);const auto primary=estimator.estimate(batch);
        check(std::equal(prediction.begin(),prediction.end(),static_cast<const float*>(primary.mapped)),"Primary target altered its own covariance weights");
        double cross=0,variance=0,optimum=0;
        for(unsigned k=0;k<bins;++k){const double energy=std::norm(response[k]),d=prediction[k];cross+=energy/d;variance+=energy*truth[k]/(d*d);optimum+=energy/truth[k];}
        const double loss=-10*std::log10(cross*cross/(variance*optimum));
        std::cout<<"Independent diagonal covariance model "<<model<<": matched-filter SNR loss="<<loss<<" dB\n";
        check(loss<.03,"Covariance estimation left significant SNR on the table in the stated Gaussian model");
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
    auto append=[&](MatchedDetector& target,unsigned channel,const MatchedDetector::Consumer& sink,DetectionTimings* t=nullptr){
        fill(channel);target.append(SarBatch(capture,batch,{{0,0,1}}),sink,t);
        fill(channel);target.append(SarBatch(capture,batch,{{0,0,1}}),sink,t);poison();
    };
    for(unsigned channel=0;channel<2;++channel)append(detector,channel,consume,&timing);
    detector.finish(consume,&timing);detector.finish(consume,&timing);
    check(count==2 && timing.focus.batches==2,"Async finish lost, duplicated or mistimed a map");
    const auto zero=maps[0].frequencies/2;
    const auto a=maps[0].values[4*delay*maps[0].frequencies+zero],b=maps[1].values[4*delay*maps[1].frequencies+zero];
    check(std::abs(a)> .019 && std::abs(b)>.019,"Async focus retained borrowed buffers instead of owned profiles");
    check(std::abs(maps[0].phase_epoch-maps[1].phase_epoch)<1e-12,"Async channels acquired different phase epochs");
    check(std::abs(std::arg(b*std::conj(a))-1.12)<.003,"Async scheduling changed interchannel phase");
    ++frame.segment;append(detector,0,consume);
    bool failed=false;
    try{detector.finish([](std::size_t,DopplerMap&&){throw std::runtime_error("sink failed");});}
    catch(const std::runtime_error& error){failed=std::string_view(error.what())=="sink failed";}
    check(failed,"Async map consumer failure was swallowed");
    capture.channels[0].rf_hz=50e6;++frame.segment;append(detector,0,consume);
    failed=false;try{detector.finish(consume);}catch(const std::invalid_argument&){failed=true;}
    check(failed,"Async numerical failure was swallowed");capture.channels[0].rf_hz=557625000;
    for(const auto& map:maps)check(map.amplitude_units=="complex_reflectivity" && map.noise_range.empty() && map.noise_doppler.empty(),
        "Async detector applied an empirical map normalization");
    // Exercise admission through the production detector, not just GpuFocus.
    // One failed oracle row used to flush the CPI despite continuous ADC data.
    std::fill_n(valid.begin()+256*bins,bins,0U);++frame.segment;
    MatchedDetector erased(context,2,options);unsigned erased_count=0;
    auto erased_sink=[&](std::size_t,DopplerMap&& map){
        check(map.samples==maps[0].samples && map.phase_epoch==maps[0].phase_epoch,
              "Oracle erasure changed the coherent interval or phase epoch");
        const auto actual=map.values[4*delay*map.frequencies+map.frequencies/2];
        check(std::abs(actual-a)<4e-5 && map.enbw>maps[0].enbw,
              "Erased oracle row was invented or changed the known target phase");
        ++erased_count;
    };
    append(erased,0,erased_sink);erased.finish(erased_sink);
    check(erased_count==1,"A decoder erasure discarded an otherwise complete coherent interval");
    // A stalled optional projector must not hold up scientific profiles.
    // The watchdog releases it even on failure so this regression cannot hang.
    std::fill(valid.begin(),valid.end(),1U);++frame.segment;
    options.best_effort_maps=true;options.update_seconds=.001;
    std::promise<void> started,returned,release;
    auto entered=started.get_future(),completed=returned.get_future();auto resume=release.get_future().share();
    bool blocked=false;unsigned retained=0;
    MatchedDetector live(context,2,options,[&](auto&,const auto&){started.set_value();resume.wait();},
        [&](const auto&,const auto&,auto,auto){++retained;});
    std::jthread watchdog([&]{entered.wait();blocked=completed.wait_for(std::chrono::seconds(5))!=std::future_status::ready;release.set_value();});
    append(live,0,[](std::size_t,DopplerMap&&){});
    returned.set_value();watchdog.join();
    unsigned live_maps=0;
    live.finish([&](std::size_t,DopplerMap&& map){++live_maps;
        check(map.samples==maps[0].samples,"Busy display shortened its coherent aperture");});
    check(!blocked && retained==1 && live_maps==1,"Busy live display blocked capture, lost profiles or queued stale maps");
    std::cout<<"Owned async profiles, interchannel phase, physical units, caller-thread delivery, drain and failures: PASS\n";
}
}
int main(){try{
    auto context=std::make_shared<gpu::Context>();
    std::cout<<"Storage buffer limit: "<<context->gpu.physical.getProperties().limits.maxStorageBufferRange<<'\n';
    stationary();cancellation(context);noise(context);noise(context,.01F);noise(context,100);noise(context,1,4);noise(context,1,6);
    noise(context,1,2,true);
    noise(context,1,2,false,10000);
    finite(context);finite(context,32);finite(context,4);
    // The normal 259-row case covers 128- and 16-row FFT tiles. Exercise
    // both intermediate cached shapes against the same finite-waveform oracle.
    finite(context,16,false,false,19);finite(context,16,false,false,34);
    finite(context,16,true);
    finite(context,16,true,true);
    focus(context);focus(context,191625000);focus(context,191625000,true);focus(context,191625000,true,true);
    focus(context,191625000,true,true,true);noise_steps(context);
    focus(context,191625000,true,true,true,true);
    covariance_efficiency(context);owned_focus(context);
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
