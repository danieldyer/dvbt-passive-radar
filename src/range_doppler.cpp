#include "range_doppler.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace radar {
namespace {
constexpr double tau=2*std::numbers::pi;
double i0(double x) {double sum=1,term=1;for(int k=1;k<80;++k){term*=x*x/(4*k*k);sum+=term;if(term<sum*1e-16)break;}return sum;}
std::size_t smooth_ceil(std::size_t n) {
    std::size_t best=std::bit_ceil(n);
    for(std::size_t a=1;a<=best;a*=2)for(std::size_t b=a;b<=best;b*=3)for(std::size_t c=b;c<=best;c*=5)
        if(c>=n)best=std::min(best,c);
    return best;
}
}
std::string DopplerMap::json() const {
    std::ostringstream s;s<<std::setprecision(17);
    s<<"{\"version\":1,\"sequence\":"<<sequence<<",\"segment\":"<<segment
     <<",\"ranges\":"<<ranges<<",\"frequencies\":"<<frequencies<<",\"samples\":"<<samples
     <<",\"fft_size\":"<<fft_size<<",\"dc_removed\":"<<(dc_removed ? "true" : "false")
     <<",\"dc_removal\":\""<<(dc_removed ? dc_method : "none")<<'"'
     <<",\"range_start_m\":0,\"range_step_m\":"<<range_step
     <<",\"frequency_start_hz\":"<<frequency_start<<",\"frequency_step_hz\":"<<frequency_step
     <<",\"time_start_s\":"<<time_start<<",\"time_end_s\":"<<time_end
     <<",\"integration_s\":"<<integration<<",\"coherence\":"<<coherence
     <<",\"history_limit_s\":"<<history_limit<<",\"requested_integration_s\":"<<requested_integration
     <<",\"window\":\""<<window<<"\",\"enbw_hz\":"<<(enbw>0 ? enbw : 1.5/integration)
     <<",\"motion_model\":\""<<motion_model<<"\",\"phase_epoch_s\":"<<phase_epoch
     <<",\"physical_rf_hz\":"<<rf_hz<<",\"reference_origin_samples\":"<<reference_origin_samples<<",\"channel\":"<<channel
     <<",\"stationary_prediction\":\""<<stationary_prediction<<"\",\"background_start_s\":"<<background_start<<",\"background_end_s\":"<<background_end
     <<",\"format\":\"cf32le\",\"order\":\"range,doppler\",\"time_reference\":\"recording_relative\""
     <<",\"amplitude_units\":\""<<amplitude_units<<'"';
    auto array=[&](std::string_view name,const std::vector<float>& values){
        if(values.empty())return;
        s<<",\""<<name<<"\":[";for(std::size_t i=0;i<values.size();++i){if(i)s<<',';s<<values[i];}s<<']';
    };
    array("noise_range_factor",noise_range);array("noise_doppler_power",noise_doppler);s<<'}';
    return s.str();
}
void DopplerMap::write(std::ostream& out) const {
    static_assert(std::endian::native==std::endian::little);
    const auto header=json();const std::uint32_t length=static_cast<std::uint32_t>(header.size());
    const std::uint64_t bytes=values.size()*sizeof(C);
    out.write("RDM1",4);out.write(reinterpret_cast<const char*>(&length),4);out.write(reinterpret_cast<const char*>(&bytes),8);
    out.write(header.data(),length);out.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(bytes));
    if(!out)throw std::runtime_error("Range-Doppler stream write failed");
}
void DopplerMap::write(const std::filesystem::path& path) const {
    const auto temporary=path.string()+".tmp";
    std::ofstream out(temporary,std::ios::binary);
    write(out);
    out.close();if(!out)throw std::runtime_error("Range-Doppler output write failed");
    std::filesystem::rename(temporary,path);
}
std::size_t coherent_suffix(std::span<const RangeProfile> rows,double threshold,double& coherence) {
    std::complex<double> low{},high{};std::size_t best=0;coherence=0;
    for(std::size_t n=1;n<=rows.size();++n) {
        const auto& r=rows[rows.size()-n];
        const double edge=tau*3408*r.timing/8192;
        low+=std::polar(1.0,r.phase-edge);high+=std::polar(1.0,r.phase+edge);
        const double score=std::min(std::abs(low),std::abs(high))/n;
        if(score>=threshold){best=n;coherence=score;}
    }
    return best;
}
DopplerMap make_doppler_map(std::span<const RangeProfile> rows,double interval,double limit,bool remove_dc) {
    if(rows.size()<8 || !std::isfinite(interval) || interval<=0 || limit<=0 || limit>=.5/interval)
        throw std::runtime_error("Invalid range-Doppler transform dimensions");
    const auto n=smooth_ceil(rows.size());
    DopplerMap map;map.ranges=rows.front().values->size();map.samples=rows.size();map.integration=rows.size()*interval;
    map.fft_size=n;map.dc_removed=remove_dc;
    map.frequency_step=1/(n*interval);
    const auto half=static_cast<int>(std::floor(limit/map.frequency_step));
    map.frequency_start=-half*map.frequency_step;map.frequencies=2*half+1;
    map.time_start=rows.front().time;map.time_end=rows.back().time;
    map.phase_epoch=map.time_start;
    map.values.resize(map.ranges*map.frequencies);
    constexpr std::size_t tile=32;
    Fft fft(n);std::vector<C> columns(tile*n);std::vector<float> window(rows.size());double sum=0;
    for(std::size_t t=0;t<rows.size();++t){window[t]=static_cast<float>(.5-.5*std::cos(tau*t/rows.size()));sum+=window[t];}
    // Transpose a tile at a time, reusing each fetched cache line instead of
    // gathering a single complex sample from every profile on every FFT pass.
    for(std::size_t base=0;base<map.ranges;base+=tile) {
        const auto count=std::min(tile,map.ranges-base);
        std::fill(columns.begin(),columns.end(),C{});
        std::array<std::complex<double>,tile> means{};
        // Project out the complex constant using the same Hann weights as the
        // transform. An unweighted mean leaves DC for off-bin moving echoes;
        // blanking just the FFT's zero bin leaves stationary window leakage.
        if(remove_dc) {
            for(std::size_t t=0;t<rows.size();++t)for(std::size_t r=0;r<count;++r)
                means[r]+=std::complex<double>((*rows[t].values)[base+r])*double(window[t]);
            for(auto& mean:means)mean/=sum;
        }
        for(std::size_t t=0;t<rows.size();++t)for(std::size_t r=0;r<count;++r) {
            const auto value=(*rows[t].values)[base+r];
            columns[r*n+t]=remove_dc ? C((std::complex<double>(value)-means[r])*double(window[t])) : value*window[t];
        }
        for(std::size_t r=0;r<count;++r) {
            auto column=std::span<C>{columns}.subspan(r*n,n);fft.transform(column);
            for(int k=-half;k<=half;++k)map.values[(base+r)*map.frequencies+k+half]=column[(k+static_cast<int>(n))%n]/static_cast<float>(sum);
        }
    }
    return map;
}
RangeDoppler::RangeDoppler(MapOptions options,double symbol_interval,std::filesystem::path output)
    : options_(options),interval_(symbol_interval),output_(std::move(output)),fft_buffer_(8192) {
    if(!std::isfinite(options.range_metres) || options.range_metres<=0 || options.range_metres>299792458.0*interval_ ||
       !std::isfinite(options.history_seconds) || options.history_seconds<.5 || options.history_seconds>3600 ||
       options.integration_seconds<0 || options.integration_seconds>options.history_seconds || options.update_seconds<=0 ||
       options.slow_decimation<1 || options.slow_decimation>16 || options.doppler_hz<=0 ||
       options.doppler_hz>.44/(interval_*options.slow_decimation) || options.coherence_degrees<=0 || options.coherence_degrees>30)
        throw std::runtime_error("Invalid map limits (Doppler must fit the slow-time filter passband)");
    ranges_=std::min<std::size_t>(8192,static_cast<std::size_t>(options.range_metres/(299792458.0/8e6))+1);
    // 129 taps per decimation factor; symmetric Kaiser low-pass retains the
    // requested Doppler band and rejects aliases before slow-time downsampling.
    const auto half=64*options.slow_decimation;
    taps_.resize(options.slow_decimation==1 ? 1 : 2*half+1);
    double sum=0;
    for(std::size_t i=0;i<taps_.size();++i) {
        const auto x=static_cast<double>(i)-static_cast<double>(taps_.size()/2);
        const double cutoff=.5/options.slow_decimation;
        const double sinc=x==0 ? 2*cutoff : std::sin(tau*cutoff*x)/(std::numbers::pi*x);
        taps_[i]=static_cast<float>(sinc*i0(12*std::sqrt(std::max(0.0,1-x*x/(half*half))))/i0(12));sum+=taps_[i];
    }
    for(auto& t:taps_)t/=static_cast<float>(sum);
    filter_ring_.resize(taps_.size()*ranges_);
    if(!output_.empty())std::filesystem::create_directories(output_);
}
RangeDoppler::~RangeDoppler()=default;
void RangeDoppler::append(std::span<const C> cross,double time,double phase,double timing,bool valid,std::uint64_t segment,RangeTimings* timings) {
    if(cross.size()!=6817)throw std::runtime_error("Map input must contain the 6817 active carriers");
    if(segment_!=segment || !valid) {
        if(!history_.empty())publish(true);
        history_.clear();diagnostics_.clear();input_rows_=0;segment_=segment;last_publish_=-1e30;
        if(!valid)return;
    }
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    std::fill(fft_buffer_.begin(),fft_buffer_.end(),C{});
    for(std::size_t k=0;k<cross.size();++k)fft_buffer_[(k+8192-3408)%8192]=cross[k];
    range_fft_.transform(fft_buffer_,true);
    if(timings) {const auto now=Clock::now();timings->inverse+=std::chrono::duration<double>(now-mark).count();mark=now;}
    std::copy_n(fft_buffer_.begin(),ranges_,filter_ring_.begin()+static_cast<std::ptrdiff_t>((input_rows_%taps_.size())*ranges_));
    diagnostics_.push_back({{},time,phase,timing});
    if(diagnostics_.size()>taps_.size())diagnostics_.pop_front();
    ++input_rows_;
    if(input_rows_<taps_.size() || (input_rows_-taps_.size())%options_.slow_decimation) {
        if(timings)timings->filter+=std::chrono::duration<double>(Clock::now()-mark).count();
        return;
    }
    auto profile=diagnostics_[taps_.size()/2];auto values=std::make_shared<std::vector<C>>(ranges_);
    for(std::size_t t=0;t<taps_.size();++t) {
        const auto* src=filter_ring_.data()+((input_rows_-taps_.size()+t)%taps_.size())*ranges_;
        for(std::size_t r=0;r<ranges_;++r)(*values)[r]+=src[r]*taps_[t];
    }
    profile.values=std::move(values);history_.push_back(std::move(profile));
    while(history_.size()*interval_*options_.slow_decimation>options_.history_seconds)history_.pop_front();
    if(timings)timings->filter+=std::chrono::duration<double>(Clock::now()-mark).count();
    if(time-last_publish_>=options_.update_seconds && history_.size()*interval_*options_.slow_decimation>=.5) {
        publish(false);last_publish_=time;
    }
}
void RangeDoppler::publish(bool final) {
    if(pending_.valid()) {
        if(!final && !options_.stream && pending_.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
        pending_.get();
    }
    if(history_.size()<8)return;
    // The viewer writes a scalar request atomically. It changes the next map,
    // never the sampling, reconstruction, or stored coherent history.
    if(!options_.stream && !output_.empty()) {
        if(std::ifstream request(output_/"integration.txt");request) {
            double seconds;if(request>>seconds && std::isfinite(seconds) && seconds>=0 && seconds<=options_.history_seconds)
                options_.integration_seconds=seconds;
        }
    }
    const double interval=interval_*options_.slow_decimation;
    const auto requested=options_.integration_seconds>0 ? static_cast<std::size_t>(options_.integration_seconds/interval) : history_.size();
    const auto count=std::min(history_.size(),requested);
    std::vector<RangeProfile> rows(history_.end()-static_cast<std::ptrdiff_t>(count),history_.end());
    double coherence=0;
    const auto n=coherent_suffix(rows,std::exp(-.5*std::pow(options_.coherence_degrees*std::numbers::pi/180,2)),coherence);
    if(n<8)return;
    rows.erase(rows.begin(),rows.end()-static_cast<std::ptrdiff_t>(n));
    const auto sequence=++sequence_;const auto segment=segment_;const auto path=output_.empty() ? output_ : output_/"latest.rdm";const auto limit=options_.doppler_hz;
    const auto history_limit=options_.history_seconds,requested_integration=options_.integration_seconds;
    auto* stream=options_.stream;const auto remove_dc=options_.remove_dc;
    pending_=std::async(std::launch::async,[rows=std::move(rows),interval,limit,sequence,segment,coherence,path,history_limit,requested_integration,stream,remove_dc]() {
        auto map=make_doppler_map(rows,interval,limit,remove_dc);map.sequence=sequence;map.segment=segment;
        map.history_limit=history_limit;map.requested_integration=requested_integration;
        map.coherence=coherence;if(!path.empty())map.write(path);
        if(stream) {map.write(*stream);stream->flush();if(!*stream)throw std::runtime_error("Range-Doppler stream flush failed");}
    });
    if(final)pending_.get();
}
void RangeDoppler::finish(){publish(true);}
}
