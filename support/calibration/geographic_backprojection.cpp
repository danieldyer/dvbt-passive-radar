// Offline exact point backprojection. Input geometry and coefficients remain
// explicit; no calibration is inferred or normalized by this renderer.
#include "gpu_compute.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

namespace {
using Complex=std::complex<float>;
using Quad=std::array<float,4>;
constexpr unsigned phases=4096,taps=64;
constexpr std::uint32_t code[] =
#include "geographic_backprojection.inc"
;
struct Parameters {unsigned pixels,channels,phases,unused;};

double bessel0(double x){
    // Convergent power series on the bounded Kaiser argument 0 <= x <= 8.
    double sum=1,term=1;
    for(unsigned k=1;k<40;++k){term*=x*x/(4*k*k);sum+=term;if(term<sum*1e-16)break;}
    return sum;
}

std::vector<float> interpolation(){
    std::vector<float> table((phases+1)*taps);
    for(unsigned phase=0;phase<=phases;++phase){
        double sum=0;std::array<double,taps> weights;
        for(unsigned tap=0;tap<taps;++tap){
            const double x=double(phase)/phases-double(tap)+31;
            const double sinc=x==0 ? 1. : std::sin(std::numbers::pi*x)/(std::numbers::pi*x);
            weights[tap]=sinc*bessel0(8*std::sqrt(std::max(0.,1-x*x/1024)))/bessel0(8.);
            sum+=weights[tap];
        }
        for(unsigned tap=0;tap<taps;++tap)table[phase*taps+tap]=float(weights[tap]/sum);
    }
    return table;
}

class Backprojection {
    radar::gpu::Kernel kernel_{code,sizeof(Parameters)};
    unsigned pixels_,channels_;
public:
    Backprojection(std::span<const Quad> geometry,unsigned channels):pixels_(channels ? unsigned(geometry.size()/channels) : 0),channels_(channels){
        if(!channels || channels>64 || geometry.empty() || geometry.size()%channels)
            throw std::invalid_argument("Invalid geometry dimensions");
        kernel_.upload<Quad>(0,geometry);const auto coefficients=interpolation();kernel_.upload<float>(3,coefficients);
        kernel_.storage(4,pixels_*sizeof(Quad));
    }
    std::span<const Quad> calculate(std::span<const Complex> profiles,std::span<const Quad> weights){
        if(profiles.size()!=channels_*1024 || weights.size()!=channels_)throw std::invalid_argument("Invalid profile dimensions");
        kernel_.upload<Complex>(1,profiles);kernel_.upload<Quad>(2,weights);
        kernel_.run(Parameters{pixels_,channels_,phases,0},pixels_);
        return {static_cast<const Quad*>(kernel_.view(4).mapped),pixels_};
    }
};

int self_test(){
    constexpr unsigned channels=5,pixels=1024;
    std::mt19937 random(271828);std::uniform_real_distribution<float> uniform(-1,1);
    std::vector<Complex> profiles(channels*1024);for(auto& z:profiles)z={uniform(random),uniform(random)};
    std::array<Quad,channels> weights;
    for(auto& w:weights)w={2+uniform(random),uniform(random),uniform(random),1.1F};
    std::vector<Quad> geometry(pixels*channels);
    for(auto& g:geometry){const double phase=uniform(random)*std::numbers::pi;g={450+450*uniform(random),float(std::cos(phase)),float(std::sin(phase)),1};}
    for(unsigned c=0;c<channels;++c)geometry[c][3]=0;
    Backprojection gpu(geometry,channels);const auto output=gpu.calculate(profiles,weights);const auto table=interpolation();
    double error=0,energy=0;
    for(unsigned pixel=0;pixel<pixels;++pixel){
        std::complex<double> sum{};double independent=0,information=0;
        for(unsigned channel=0;channel<channels;++channel){
            const auto g=geometry[pixel*channels+channel],w=weights[channel];if(!g[3])continue;
            const int base=int(std::floor(g[0]));const double fraction=(double(g[0])-base)*phases;
            const unsigned phase=std::min(unsigned(fraction),phases-1);const double mix=fraction-phase;
            std::complex<double> z{};
            for(unsigned tap=0;tap<taps;++tap){
                const int lag=base+int(tap)-31;if(lag < -64 || lag>928)continue;
                const double a=table[phase*taps+tap],b=table[(phase+1)*taps+tap];
                z+=std::complex<double>(profiles[channel*1024+(lag+1024)%1024])*(a+(b-a)*mix);
            }
            sum+=z*std::complex<double>(g[1],g[2])*std::complex<double>(w[1],w[2])*double(w[0]);
            information+=(w[1]*w[1]+w[2]*w[2])*w[3]*w[0];
            independent+=std::norm(z)*w[0]/w[3];
        }
        const std::array expected{sum.real(),sum.imag(),independent,information};
        for(unsigned k=0;k<4;++k){error+=std::pow(output[pixel][k]-expected[k],2);energy+=expected[k]*expected[k];}
    }
    const double relative=std::sqrt(error/energy);std::cerr<<"GPU/double-reference relative RMS error "<<relative<<'\n';
    // Independent analytic test: fractional-delay, bandlimited point echoes.
    // Their RF propagation phases must add constructively after backprojection.
    constexpr std::array rf{177.5e6,184.5e6,191.625e6,219.5e6,226.5e6};
    const Complex amplitude{.4F,.7F};double maximum=0;
    for(unsigned trial=0;trial<12;++trial){
        std::array<Quad,channels> point;
        for(unsigned channel=0;channel<channels;++channel){
            const float coordinate=trial==0 ? .125F : 30+800*(1+uniform(random))/2;
            const double phase=std::remainder(2*std::numbers::pi*rf[channel]*coordinate/8e6,2*std::numbers::pi);
            const Complex rotation{float(std::cos(phase)),float(std::sin(phase))};
            point[channel]={coordinate,rotation.real(),rotation.imag(),1};weights[channel]={1,1,0,1};
            for(unsigned sample=0;sample<1024;++sample){
                const int lag=sample>=960 ? int(sample)-1024 : int(sample);
                const double argument=.85*(lag-double(coordinate));
                const double sinc=argument==0 ? 1. : std::sin(std::numbers::pi*argument)/(std::numbers::pi*argument);
                profiles[channel*1024+sample]=amplitude*std::conj(rotation)*float(sinc);
            }
        }
        Backprojection focused(point,channels);const auto value=focused.calculate(profiles,weights).front();
        maximum=std::max(maximum,std::abs(std::complex<double>(value[0],value[1])/double(channels)-std::complex<double>(amplitude)));
    }
    std::cerr<<"Analytic fractional point/phase maximum complex error "<<maximum<<'\n';
    return relative<2e-6 && maximum<5e-5 ? 0 : 1;
}
}

int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(Complex)==8 && sizeof(Quad)==16);
    if(argc==2 && std::string_view(argv[1])=="--self-test")return self_test();
    if(argc!=3)throw std::invalid_argument("geographic_backprojection GEOMETRY.bin CHANNELS < frames.bin > images.bin; or --self-test");
    const auto channels=unsigned(std::stoul(argv[2]));if(!channels || channels>64)throw std::invalid_argument("Invalid channel count");
    std::ifstream source(argv[1],std::ios::binary|std::ios::ate);source.exceptions(std::ios::badbit|std::ios::failbit);
    const auto bytes=source.tellg();if(bytes<=0 || bytes%std::streamoff(sizeof(Quad)*channels))throw std::invalid_argument("Incomplete geometry");
    std::vector<Quad> geometry(std::size_t(bytes)/sizeof(Quad));source.seekg(0);source.read(reinterpret_cast<char*>(geometry.data()),bytes);
    for(const auto& g:geometry)if(!std::ranges::all_of(g,[](float v){return std::isfinite(v);}))throw std::invalid_argument("Nonfinite geometry");
    Backprojection gpu(geometry,channels);std::vector<Complex> profiles(channels*1024);std::vector<Quad> weights(channels);
    std::cin.exceptions(std::ios::badbit);std::cout.exceptions(std::ios::badbit|std::ios::failbit);
    while(std::cin.peek()!=std::char_traits<char>::eof()){
        std::cin.read(reinterpret_cast<char*>(profiles.data()),std::streamsize(profiles.size()*sizeof(Complex)));
        std::cin.read(reinterpret_cast<char*>(weights.data()),std::streamsize(weights.size()*sizeof(Quad)));
        if(!std::cin)throw std::runtime_error("Incomplete input frame");
        for(const auto& w:weights)if(!std::ranges::all_of(w,[](float v){return std::isfinite(v);}) || w[0]<0 || w[3]<=0)
            throw std::invalid_argument("Invalid information or template energy");
        const auto output=gpu.calculate(profiles,weights);
        std::cout.write(reinterpret_cast<const char*>(output.data()),std::streamsize(output.size_bytes()));std::cout.flush();
    }
}catch(const std::exception& e){std::cerr<<"Backprojection: "<<e.what()<<'\n';return 1;}}
