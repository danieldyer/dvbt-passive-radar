#include "gpu_frontend.hpp"
#include "fractional_resampler.hpp"
#include "frontend_reference.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
void wave_file(const std::filesystem::path& path,std::span<const C> data) {
    std::ofstream wave(path,std::ios::binary);
    auto put=[&](auto v){wave.write(reinterpret_cast<const char*>(&v),sizeof(v));};
    wave.write("RIFF",4);put(std::uint32_t(36+data.size_bytes()));wave.write("WAVEfmt ",8);put(std::uint32_t{16});
    put(std::uint16_t{3});put(std::uint16_t{2});put(std::uint32_t{56000000});put(std::uint32_t{448000000});
    put(std::uint16_t{8});put(std::uint16_t{32});wave.write("data",4);put(std::uint32_t(data.size_bytes()));
    wave.write(reinterpret_cast<const char*>(data.data()),data.size_bytes());
}
}
int main() {
    try {
        // All supported ADC rates meet the same adjacent-channel mask. FFTW
        // evaluates a dense grid independently of the coefficient generator.
        Fft response_fft(262144);std::vector<C> response(262144);
        for(unsigned d=1;d<=16;++d) {
            const auto prototype=GpuChannelBank::prototype(d*8e6);
            std::fill(response.begin(),response.end(),C{});
            std::copy(prototype.begin(),prototype.end(),response.begin());response_fft.transform(response);
            double ripple=0,stop=0;
            for(std::size_t k=0;k<=response.size()/2;++k) {
                const double f=k*(d*8e6)/response.size();
                if(f<=3328125)ripple=std::max(ripple,std::abs(double(std::abs(response[k]))-1));
                if(f>=3671875)stop=std::max(stop,double(std::abs(response[k])));
            }
            require(ripple<1e-5 && stop<1e-5,"Prototype mask fails at a supported input rate");
        }
        const auto path=std::filesystem::temp_directory_path()/("gpu-bank-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".wav");
        struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove(path,e);}} cleanup{path};
        const std::array offsets{-18e6,-9.512345e6,137123.25,9e6,18e6};
        std::vector<C> data(7*(2*GpuChannelBank::batch_size+109)+1100);
        for(std::size_t n=0;n<data.size();++n) {
            std::complex<double> z{};
            for(std::size_t c=0;c<offsets.size();++c)z+=std::polar(.1,tau*(offsets[c]+.4375e6)*n/56e6+.23*c)+
                std::polar(.04,tau*(offsets[c]-.925e6)*n/56e6-.14*c);
            data[n]=static_cast<C>(z);
        }
        wave_file(path,data);
        WaveReader wave(path);GpuChannelBank bank(wave,offsets,0,1),resident(wave,offsets,0,1);
        std::size_t sample=0,blocks=0;double worst=0;
        while(true) {
            const auto values=bank.next();const auto device=resident.next_device();
            if(values.empty()) {require(device.count==0,"Device channel bank tail differs");break;}
            ++blocks;
            require(device.first==sample && device.count==values[0].size() && device.channels==offsets.size(),"Device bank sample coordinates differ");
            const auto* mapped=static_cast<const C*>(device.samples.mapped);
            for(std::size_t c=0;c<offsets.size();++c)for(std::size_t m=0;m<values[c].size();++m) {
                require(values[c][m]==mapped[c*device.stride+m],"Device bank output differs from host materialization");
                const auto n=bank.origin+(sample+m)*7;
                const auto expected=std::polar(.1,tau*.4375e6*n/56e6+.23*c)+std::polar(.04,-tau*.925e6*n/56e6-.14*c);
                worst=std::max(worst,std::abs(std::complex<double>(values[c][m])-expected));
            }
            sample+=values[0].size();
        }
        require(blocks>=3 && sample==bank.output_count && worst<5e-6,"GPU bank translation, phase, gain, tail or boundary failure");
        GpuChannelBank sought(wave,offsets,.002,.001);
        const auto values=sought.next();double seek_error=0;
        for(std::size_t c=0;c<offsets.size();++c)for(std::size_t m=0;m<values[c].size();++m) {
            const auto n=sought.origin+m*7;
            const auto expected=std::polar(.1,tau*.4375e6*n/56e6+.23*c)+std::polar(.04,-tau*.925e6*n/56e6-.14*c);
            seek_error=std::max(seek_error,std::abs(std::complex<double>(values[c][m])-expected));
        }
        require(seek_error<5e-6,"GPU bank seek changes absolute phase");
        std::cout<<"Five arbitrary centres: max error="<<worst<<", seek="<<seek_error<<", taps="<<bank.taps()<<'\n';
        // Independent scalar oracle with broadband random input covers all
        // polyphase branches, not just convenient sinusoidal cancellations.
        std::mt19937 random(719);std::uniform_real_distribution<float> uniform(-1,1);
        data.resize(17000);for(auto& z:data)z={uniform(random),uniform(random)};
        wave_file(path,data);WaveReader random_input(path);GpuChannelBank random_bank(random_input,offsets,0,1);
        const auto random_output=random_bank.next();const auto taps=GpuChannelBank::prototype(56e6);
        double reference_error=0;
        for(std::size_t c=0;c<offsets.size();++c) {
            const auto expected=channel_reference(std::span{data}.subspan(random_bank.origin-taps.size()/2),taps,offsets[c],56e6,
                random_bank.origin,7,random_output[c].size());
            for(std::size_t m=0;m<expected.size();++m)
                reference_error=std::max(reference_error,double(std::abs(expected[m]-random_output[c][m])));
        }
        require(reference_error<2e-6,"GPU bank disagrees with independent double-precision scalar convolution");
        std::cout<<"Scalar FIR CPU oracle max error="<<reference_error<<'\n';
        // Large absolute coordinates must never be rounded to FP32. Generate
        // this oracle with integer modular arithmetic, independent of the NCO.
        const std::array off_grid{137123.25};GpuChannelBank distant(56000000,off_grid);
        const std::uint64_t centre=(std::uint64_t{1}<<40)+3345;
        data.resize(8192);
        for(std::size_t j=0;j<data.size();++j) {
            const auto n=centre-distant.taps()/2+j;
            const double cycles=double(((n%224000000)*2548493)%224000000)/224000000;
            data[j]=static_cast<C>(std::polar(1.0,tau*cycles));
        }
        const auto distant_output=distant.process(data,centre,1000);
        double distant_phase=0;
        for(std::size_t m=0;m<1000;++m) {
            const auto expected=std::polar(1.0,tau*double((centre+7*m)%112)/112);
            const auto ratio=std::complex<double>(distant_output[0][m])*std::conj(expected);
            distant_phase=std::max(distant_phase,std::abs(std::arg(ratio)));
            require(std::abs(std::abs(ratio)-1)<1e-5,"Large-coordinate NCO gain error");
        }
        require(distant_phase<1e-6,"Large-coordinate NCO lost phase precision");
        std::cout<<"NCO at ADC sample 2^40: max phase error="<<distant_phase<<" rad\n";
        // Full-scale single-tone blockers, including transition edge and alias
        // images. A wanted tone 80 dB below full scale must survive unchanged.
        const std::array single{4.625e6};double leakage=0;
        data.resize(7*8192+1100);
        for(double blocker : {-25e6,-12e6,-4e6,-3.671875e6,3.671875e6,4e6,4.673e6,7e6,13.37e6,21e6}) {
            for(std::size_t n=0;n<data.size();++n)data[n]=static_cast<C>(
                std::polar(1.0,tau*(single[0]+blocker)*n/56e6)+
                std::polar(1e-4,tau*(single[0]+.5e6)*n/56e6));
            wave_file(path,data);WaveReader input(path);GpuChannelBank filter(input,single,0,1);
            const auto filtered=filter.next();
            for(std::size_t m=0;m<filtered[0].size();++m) {
                const auto wanted=std::polar(1e-4,tau*.5e6*(filter.origin+m*7)/56e6);
                leakage=std::max(leakage,std::abs(std::complex<double>(filtered[0][m])-wanted));
            }
        }
        std::cout<<"Full-scale blocker worst leakage="<<20*std::log10(leakage)<<" dBFS\n";
        require(leakage<1e-5,"GPU FIR leakage exceeds 100 dB design requirement");
        GpuResampler gpu;FractionalResampler cpu;
        data.resize(70000);
        for(std::size_t n=0;n<data.size();++n)data[n]=static_cast<C>(std::polar(.3,tau*.416015625*n)+
            std::polar(.2,-tau*.397*n+.3)+std::polar(.5,tau*.0273*n));
        double comparison=0,analytic=0;
        for(double rate:{.9998,1.000005,1.0002})for(double mu:{.0,.4999,.999999}) {
            const auto out=gpu.process(data,48+mu,rate,65537);
            for(std::size_t m=0;m<out.size();++m) {
                const double t=48+mu+m*rate,integer=std::floor(t);
                comparison=std::max(comparison,double(std::abs(out[m]-cpu.interpolate(data.data()+std::size_t(integer)-47,t-integer))));
                const auto expected=std::polar(.3,tau*.416015625*t)+std::polar(.2,-tau*.397*t+.3)+std::polar(.5,tau*.0273*t);
                analytic=std::max(analytic,std::abs(std::complex<double>(out[m])-expected));
            }
        }
        std::cout<<"Resampler CPU/GPU="<<comparison<<", analytic="<<analytic<<'\n';
        require(comparison<3e-6 && analytic<5e-5,"GPU resampler phase/rate/interpolation error");
        bool rejected=false;
        try{gpu.process(data,46.9,1,100);}catch(const std::runtime_error&){rejected=true;}
        require(rejected,"Resampler accepts missing left support");
        std::cout<<"GPU front-end PASS\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
