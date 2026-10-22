#include "gpu_frontend.hpp"
#include "channel_clock_interpolator.hpp"
#include "clock_interpolator.hpp"
#include "small_clock_interpolator.hpp"
#include "frontend_reference.hpp"
#include "gpu_compute.hpp"
#include <algorithm>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double unwarp(double u,std::span<const ClockSegment> segments,std::uint64_t origin=0) {
    auto s=segments.front();
    for(auto next:segments)if(u>=double(next.start)+next.offset)s=next;
    return double(std::int64_t(s.start)-std::int64_t(origin))+(u-double(s.start)-s.offset)/s.rate;
}
// Exact modular arithmetic for quarter-Hz test frequencies and a large epoch.
// It does not share the production compensated floating-point NCO formula.
std::complex<double> epoch_nco(std::uint64_t epoch,double f,std::uint32_t fs) {
    const auto numerator=std::int64_t(std::llround(4*f)),denominator=std::int64_t(4)*fs;
    const auto remainder=(std::int64_t(epoch%denominator)*numerator)%denominator;
    return std::polar(1.0,-tau*double(remainder)/denominator);
}
void interpolation() {
    ChannelClockInterpolator filter;double maximum=0,gain=0,small_error=0;
    for(unsigned p=0;p<=256;++p) {
        const double mu=(p+.317)/257,phase=mu*filter.phases;
        const auto index=unsigned(phase);const double alpha=phase-index;
        for(unsigned k=0;k<=1024;++k) {
            const double f=.5*k/1024;std::complex<double> response{};
            for(unsigned j=0;j<filter.taps;++j) {
                const double h=(1-alpha)*filter.coefficients[index*filter.taps+j]+alpha*filter.coefficients[(index+1)*filter.taps+j];
                response+=h*std::polar(1.0,tau*f*(double(j)-filter.left-mu));
            }
            gain=std::max(gain,std::abs(response));
            if(f<=3671875/16e6)maximum=std::max(maximum,std::abs(response-1.0));
        }
    }
    std::cout<<"Channel-grid interpolation: complex error="<<maximum<<", maximum full-band gain="<<gain<<'\n';
    for(unsigned p=0;p<=64;++p)for(unsigned k=0;k<=1024;++k) {
        // Include nearest-bin centre error and clock scaling above the occupied
        // edge, rather than assuming the requested centre lies on an FFT bin.
        const double mu=SmallClockInterpolator::limit*(2.0*p/64-1),f=3340000/16e6*k/1024;
        std::complex<double> derivative{};
        for(unsigned j=1;j<=3;++j)derivative+=SmallClockInterpolator::weights[j-1]*
            (std::polar(1.0,tau*f*j)-std::polar(1.0,-tau*f*j));
        const auto response=1.0+mu*derivative+mu*mu*(std::cos(tau*f)-1);
        small_error=std::max(small_error,std::abs(response-std::polar(1.0,tau*f*mu)));
    }
    std::cout<<"Small residual interpolation: complex error="<<small_error<<'\n';
    require(small_error<5e-6,"Small residual warp exceeds its declared error bound");
    require(maximum<1e-6 && gain<1.001,"Channel-grid interpolator violates phase/amplitude or blocker-gain bound");
}
void waveform(double rate) {
    constexpr std::uint32_t fs=56000000;
    constexpr std::uint64_t epoch=(1ULL<<40)+137,first=epoch+7000,start=epoch+5000;
    constexpr unsigned count=4096;
    const std::array centres{-21e6,-9.5e6,132812.25,9e6,21e6}; // quarter-Hz below a half-bin centre
    const std::array frequencies{-3328125.0,-2900000.0,-1250123.5,437500.0,3328125.0};
    auto clock=std::make_shared<SampleClock>(fs,centres);
    clock->schedule(epoch-50000,rate);
    clock->schedule(first+8031,std::abs(rate-1)<.0001 ? 2-rate : 1+.99*(rate-1));clock->schedule(first+17343,rate);
    const auto segments=clock->segments();
    GpuChannelBank bank(fs,centres);bank.set_clock(clock);
    bank.origin=first;
    std::vector<C> input(40000);
    for(std::size_t j=0;j<input.size();++j) {
        const double n=unwarp(double(start+j),segments,epoch);
        std::complex<double> value{};
        for(unsigned c=0;c<centres.size();++c)for(unsigned k=0;k<frequencies.size();++k)
            value+=std::polar(.015,tau*(centres[c]+frequencies[k])*n/fs+.173*c-.117*k);
        input[j]=static_cast<C>(value);
    }
    const auto output=bank.process_clocked(input,start,first,count);
    // Alternate distinct resident buffers through the SAME caller variable.
    // This catches stale VkFFT descriptors (its launch replacement test uses
    // pointer identity), and checks the path with no accessible host mapping.
    std::array<gpu::Buffer,2> resident;
    std::array<gpu::BufferView,2> views;
    auto rotated=input;for(auto& value:rotated)value*=C{.6f,-.8f};
    const std::array expected{output,bank.process_clocked(rotated,start,first,count)};
    for(unsigned v=0;v<2;++v) {
        const auto bytes=bank.input_capacity()*sizeof(C);
        resident[v]=gpu::make_buffer(bank.context()->gpu,bytes);
        auto* mapped=static_cast<C*>(bank.context()->gpu.device->mapMemory(*resident[v].memory,0,bytes));
        const auto& values=v ? rotated : input;
        std::copy(values.begin(),values.end(),mapped);std::fill(mapped+values.size(),mapped+bank.input_capacity(),C{});
        views[v]={bank.context(),*resident[v].handle,bytes,nullptr};
    }
    for(unsigned v:{0U,1U,0U}) {
        const auto view=views[v];
        const auto batch=bank.process_clocked_device(view,input.size(),start,first,count);
        const auto* data=static_cast<const C*>(batch.samples.mapped);
        require(batch.first==0 && batch.count==count && batch.channels==centres.size(),"Resident input lost common-grid coordinates");
        for(unsigned c=0;c<centres.size();++c)for(unsigned m=0;m<count;++m)
            require(data[c*batch.stride+m]==expected[v][c][m],"Resident and host input paths differ or retained a stale FFT descriptor");
    }
    double maximum=0,relative_phase=0,energy=0,error=0;
    for(unsigned m=0;m<count;++m) {
        std::array<std::complex<double>,centres.size()> ratio{};
        std::array<double,centres.size()> amplitude{};
        for(unsigned c=0;c<centres.size();++c) {
            std::complex<double> expected{};
            const double n=double(first-epoch)+7*m;
            for(unsigned k=0;k<frequencies.size();++k)
                expected+=std::polar(.015,tau*frequencies[k]*n/fs+.173*c-.117*k);
            expected*=epoch_nco(epoch,centres[c],fs);
            const double e=std::abs(std::complex<double>(output[c][m])-expected);
            maximum=std::max(maximum,e);energy+=std::norm(expected);error+=e*e;
            amplitude[c]=std::abs(expected);
            ratio[c]=std::complex<double>(output[c][m])/expected;
        }
        for(unsigned c=1;c<centres.size();++c)if(amplitude[0]>.01 && amplitude[c]>.01)
            relative_phase=std::max(relative_phase,std::abs(std::arg(ratio[c]*std::conj(ratio[0]))));
    }
    const auto overlap=bank.process_clocked(input,start,first+7*777,count-777);
    double boundary=0;
    for(unsigned c=0;c<centres.size();++c)for(unsigned m=0;m<count-777;++m)
        boundary=std::max(boundary,double(std::abs(output[c][m+777]-overlap[c][m])));
    std::cout<<"Five distinct channels, rate="<<rate<<": RMS error="<<std::sqrt(error/energy)
        <<", max="<<maximum<<", relative phase="<<relative_phase*180/std::numbers::pi
        <<" degrees, chunk error="<<boundary<<'\n';
    require(maximum<5e-6 && relative_phase<.01*std::numbers::pi/180 && boundary<2e-6,
        "Spectral clock correction loses multichannel coherence, large-epoch phase or chunk independence");
}
void blocker() {
    constexpr std::uint32_t fs=56000000;
    const std::array centres{4625000.0};
    auto clock=std::make_shared<SampleClock>(fs,centres);clock->schedule(64,1.00003);clock->schedule(13237,.999985);
    GpuChannelBank bank(fs,centres);bank.set_clock(clock);
    constexpr std::uint64_t first=7000,start=5000;
    const auto segments=clock->segments();std::vector<C> input(24000);
    double maximum=0;
    for(double offset:{-8e6,-4e6,-3671875.0,3671875.0,4e6,8e6,17e6}) {
        for(std::size_t j=0;j<input.size();++j)
            input[j]=static_cast<C>(std::polar(1.0,tau*(centres[0]+offset)*unwarp(double(start+j),segments)/fs));
        const auto output=bank.process_clocked(input,start,first,2048);
        for(auto value:output[0])maximum=std::max(maximum,double(std::abs(value)));
    }
    std::cout<<"Clocked full-scale blocker maximum="<<20*std::log10(maximum)<<" dBFS\n";
    require(maximum<1e-5,"Spectral retiming compromises the 100 dB blocker mask");
}
void rates() {
    double maximum=0;
    for(unsigned d:{1U,2U,3U,4U,7U,8U,9U,16U}) {
        const std::uint32_t fs=d*8000000;
        const std::array centres{d==1 ? 0.0 : fs/8.0};
        auto clock=std::make_shared<SampleClock>(fs,centres);clock->schedule(64,1.000005);
        GpuChannelBank bank(fs,centres);bank.set_clock(clock);
        const auto first=std::uint64_t(bank.taps()+4096);
        const auto start=first-bank.taps()-2048;
        const unsigned count=unsigned(std::min<std::size_t>(512,bank.capacity()));
        std::vector<C> input(bank.taps()+8192+(count-1)*d);
        const auto segments=clock->segments();
        for(std::size_t j=0;j<input.size();++j) {
            const double n=unwarp(double(start+j),segments);
            input[j]=static_cast<C>(std::polar(.2,tau*(centres[0]+437500)*n/fs));
        }
        const auto output=bank.process_clocked(input,start,first,count);
        for(unsigned m=0;m<count;++m)
            maximum=std::max(maximum,std::abs(std::complex<double>(output[0][m])-std::polar(.2,tau*437500*double(first+d*m)/fs)));
    }
    std::cout<<"Supported-rate spectral synthesis: maximum complex error="<<maximum<<'\n';
    require(maximum<5e-6,"Spectral resampling fails at a supported ADC decimation ratio");
}
void rational_rate(unsigned source_rate) {
    constexpr unsigned fs=64'000'000,d=8,count=4096;
    const std::array centres{-24.5e6,-17.5e6,-10.375e6,17.5e6,24.5e6};
    const std::array frequencies{-3328125.0,0.0,3328125.0};
    const std::uint64_t first=64'000'000ULL*1200+8000;
    const std::uint64_t start=first/fs*source_rate;
    auto clock=std::make_shared<SampleClock>(fs,centres);
    clock->schedule(first-1000,1.00003);clock->schedule(first+15001,.999985);
    GpuChannelBank bank(source_rate,centres);bank.set_clock(clock);bank.origin=first;
    const auto segments=clock->segments();std::vector<C> input(45000);
    for(std::size_t j=0;j<input.size();++j) {
        const auto raw=start+j;
        const auto numerator=(raw%source_rate)*std::uint64_t(fs);
        // Generate the independent analytic waveform in local coordinates.
        // Adding a fractional sample to a 20-minute absolute epoch would itself
        // inject phase error into this reference before testing the resampler.
        const double nominal=double(std::int64_t(raw/source_rate*fs+numerator/source_rate)-std::int64_t(first))+
            double(numerator%source_rate)/source_rate;
        auto segment=segments.front();
        for(auto next:segments)
            if(nominal>=double(std::int64_t(next.start)-std::int64_t(first))+next.offset)segment=next;
        const double local_start=double(std::int64_t(segment.start)-std::int64_t(first));
        const double n=(nominal-segment.offset+local_start*(segment.rate-1))/segment.rate;
        std::complex<double> value{};
        for(unsigned c=0;c<centres.size();++c)for(double f:frequencies)
            value+=std::polar(.02,tau*(centres[c]+f)*n/fs+.31*c);
        input[j]=static_cast<C>(value);
    }
    const auto output=bank.process_clocked(input,start,first,count);
    const auto overlap=bank.process_clocked(input,start,first+777*d,count-777);
    double error=0,chunk=0,phase=0;
    for(unsigned m=0;m<count;++m) {
        std::array<std::complex<double>,5> ratios{};
        double amplitude=0;
        for(unsigned c=0;c<centres.size();++c) {
            std::complex<double> expected{};
            for(double f:frequencies)expected+=std::polar(.02,tau*f*(m*d)/fs+.31*c);
            expected*=epoch_nco(first,centres[c],fs);
            amplitude=std::abs(expected);
            error=std::max(error,std::abs(std::complex<double>(output[c][m])-expected));
            ratios[c]=std::complex<double>(output[c][m])/expected;
            if(m>=777)chunk=std::max(chunk,double(std::abs(output[c][m]-overlap[c][m-777])));
        }
        // Phase is undefined at destructive interference nulls of the three
        // tones; complex absolute error above still tests every such sample.
        if(amplitude>.005)for(unsigned c=1;c<centres.size();++c)
            phase=std::max(phase,std::abs(std::arg(ratios[c]*std::conj(ratios[0]))));
    }
    std::cout<<double(source_rate)/1e6<<" MS/s, five VHF channels, late seek and clock boundaries: max="<<error<<", chunk="<<chunk<<", differential phase="<<phase*180/std::numbers::pi<<" degrees\n";
    require(error<6e-6 && chunk<3e-6 && phase<.01*std::numbers::pi/180,"Rational ADC rate conversion loses multichannel coherence");
}
void phase_histories() {
    constexpr unsigned fs=56'000'000,count=2048,decimation=7;
    constexpr std::uint64_t epoch=(1ULL<<40)+137;
    constexpr double tone=437500;
    const std::array centres{-21e6,-9.5e6,132812.25,9e6,21e6};
    auto clock=std::make_shared<SampleClock>(fs,centres);
    clock->schedule(epoch-50000,1.000005);
    clock->schedule(epoch+fs/3,1.00003);
    clock->schedule(epoch+2*fs/3,.999985);
    const auto segments=clock->segments();
    GpuChannelBank bank(fs,centres);bank.set_clock(clock);
    auto phase=[](unsigned c,double t){return .31*c+.7*(double(c)-2)*t+
        1.8*std::sin(tau*(.37+.13*c)*t)+.015*c*t*t;};
    double maximum=0,differential=0;
    // Disjoint probes cover ten seconds, long epochs, and windows crossing both
    // clock boundaries. Each band has its own linear, quadratic and sinusoidal
    // phase history. No output phase origins or phase slopes are fitted away.
    const std::array<std::uint64_t,9> times{fs/10,fs/3-2000,fs/3,fs/2,
        2*fs/3-2000,2*fs/3,fs,3ULL*fs,10ULL*fs};
    for(const auto time:times){
        const auto first=epoch+time;
        const auto start=std::uint64_t(std::int64_t(first)+std::int64_t(std::floor(clock->at(double(first)).offset_at(double(first))))-6000);
        std::vector<C> input(45000);
        for(std::size_t j=0;j<input.size();++j){
            const double n=unwarp(double(start+j),segments,epoch),t=n/fs;
            std::complex<double> sum{};
            for(unsigned c=0;c<centres.size();++c)
                sum+=std::polar(.1,tau*(centres[c]+tone)*n/fs+phase(c,t));
            input[j]=static_cast<C>(sum);
        }
        const auto output=bank.process_clocked(input,start,first,count);
        for(unsigned m=0;m<count;++m){
            const double n=double(time+decimation*m),t=n/fs;
            std::array<std::complex<double>,5> ratio;
            for(unsigned c=0;c<centres.size();++c){
                const auto expected=std::polar(.1,tau*tone*n/fs+phase(c,t))*epoch_nco(epoch,centres[c],fs);
                ratio[c]=std::complex<double>(output[c][m])/expected;
                maximum=std::max(maximum,std::abs(ratio[c]-1.));
            }
            for(unsigned c=1;c<centres.size();++c)
                differential=std::max(differential,std::abs(std::arg(ratio[c]*std::conj(ratio[0]))));
        }
    }
    std::cout<<"Independent band phase histories through shared GPU retiming: relative complex error="<<maximum
        <<", differential phase error="<<differential*180/std::numbers::pi<<" degrees\n";
    require(maximum<5e-5 && differential<.01*std::numbers::pi/180,
        "Shared retiming absorbed time-varying interband phase");
}
}
int main(int argc,char**) {
    try {
        interpolation();
        if(argc>1) {waveform(1.000005);waveform(1.00003);waveform(1.001);waveform(.999);blocker();rates();rational_rate(61'000'000);rational_rate(61'440'000);phase_histories();}
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
