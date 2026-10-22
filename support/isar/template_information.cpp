// Offline matched statistics and template information with fixed covariance.
// The variance calculation uses only the known model, not observed data.
#include "gpu_compute.hpp"
#include <array>
#include <bit>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
using Quad=std::array<float,4>;
using Complex=std::complex<float>;
constexpr std::uint32_t code[]=
#include "template_information.inc"
;
struct Parameters {
    std::uint32_t stride,bands,ranges,cells,degree,pixels,radius;
    std::int32_t first_cell;
    float symbol_duration;
};
template<class T>T read(std::istream& stream){T value;stream.read(reinterpret_cast<char*>(&value),sizeof(value));return value;}
template<class T>std::vector<T> array(std::istream& stream,std::size_t size){std::vector<T> value(size);stream.read(reinterpret_cast<char*>(value.data()),std::streamsize(size*sizeof(T)));return value;}
void run(const char* input_path,const char* output_path,const char* pixel_path){
    std::ifstream input(input_path,std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    if(read<std::array<char,8>>(input)!=std::array{'F','I','E','L','D','I','N','2'})throw std::invalid_argument("Invalid template-information model");
    auto p=read<Parameters>(input);
    if(!p.stride || p.stride>65536 || !p.bands || p.bands>8 || !p.ranges || p.ranges>33 ||
       !p.cells || p.cells>64 || p.degree<6 || p.degree>12 || !p.pixels || p.pixels>1000000 || p.radius!=32)
        throw std::invalid_argument("Invalid template-information dimensions");
    const auto rows=std::size_t(p.stride)*p.bands;
    const auto geometry=array<Quad>(input,rows*10);
    const auto pulses=array<Complex>(input,rows*2*p.radius*(p.degree+1));
    const auto norms=array<float>(input,rows*p.cells*(2*p.degree+1));
    const auto whitening=array<Complex>(input,std::size_t(p.bands)*p.ranges*p.ranges);
    auto pixels=array<Quad>(input,p.pixels);
    const auto offsets=array<std::int32_t>(input,p.ranges);
    const auto moments=array<float>(input,rows*7);
    const auto observed=array<Complex>(input,rows*p.cells*(p.degree+1));
    const auto point=array<Complex>(input,rows*p.cells*(p.degree+1));
    if(input.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing template-information input");
    if(pixel_path){
        std::ifstream points(pixel_path,std::ios::binary);points.exceptions(std::ios::badbit|std::ios::failbit);
        const auto count=read<std::uint32_t>(points);
        if(!count || count>1000000)throw std::invalid_argument("Invalid replacement pixel count");
        pixels=array<Quad>(points,count);p.pixels=count;
        if(points.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing replacement pixels");
    }
    radar::gpu::Kernel kernel{code,sizeof(Parameters)};
    kernel.upload<Quad>(0,geometry);kernel.upload<Complex>(1,pulses);kernel.upload<float>(2,norms);
    kernel.upload<Complex>(3,whitening);kernel.upload<Quad>(4,pixels);kernel.upload<std::int32_t>(5,offsets);
    kernel.storage(6,std::size_t(p.bands)*p.pixels*sizeof(Quad));
    constexpr std::array<std::uint32_t,1> zero{0};kernel.upload<std::uint32_t>(7,zero);
    kernel.upload<float>(8,moments);kernel.upload<Complex>(9,observed);kernel.upload<Complex>(10,point);
    kernel.storage(11,std::size_t(p.bands)*p.pixels*sizeof(Quad));
    const auto cmd=kernel.begin();kernel.groups(cmd,kernel.primary(),p,(p.pixels+63)/64,p.bands);kernel.finish(cmd);
    if(*static_cast<const unsigned*>(kernel.view(7).mapped))throw std::invalid_argument("A template exceeded the retained pulse support");
    std::ofstream output(output_path,std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    output.write(reinterpret_cast<const char*>(kernel.view(6).mapped),std::streamsize(std::size_t(p.bands)*p.pixels*sizeof(Quad)));
    output.write(reinterpret_cast<const char*>(kernel.view(11).mapped),std::streamsize(std::size_t(p.bands)*p.pixels*sizeof(Quad)));
}
}
int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(Parameters)==36 && sizeof(Quad)==16 && sizeof(Complex)==8);
    if(argc!=3 && argc!=4)throw std::invalid_argument("isar_template_information MODEL.bin OUTPUT.bin [PIXELS.bin]");
    run(argv[1],argv[2],argc==4 ? argv[3] : nullptr);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
