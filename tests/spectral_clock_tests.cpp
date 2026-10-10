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
}
int main(int argc,char**) {
    try {
        interpolation();
        if(argc>1) {waveform(1.000005);waveform(1.00003);waveform(1.001);waveform(.999);blocker();rates();}
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
