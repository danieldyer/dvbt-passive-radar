// Offline dense-image autofocus. Fixed known-waveform polynomials are uploaded
// once; trial geometry changes range, carrier and symbol Doppler together.
#include "gpu_compute.hpp"
#include "binary_pipe.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using Complex=std::complex<float>;
using Quad=std::array<float,4>;
constexpr std::uint32_t code[]=
#include "focus_patch.inc"
;
constexpr std::uint32_t reduction[]=
#include "focus_reduce.inc"
;
constexpr std::uint32_t adjoint_code[]=
#include "focus_adjoint.inc"
;
constexpr std::uint32_t adjoint_reduction[]=
#include "focus_adjoint_reduce.inc"
;
constexpr std::uint32_t history_code[]=
#include "focus_histories.inc"
;
struct Parameters {
    std::uint32_t stride,bands,cells,degree,pixels,blocks;
    std::int32_t first_cell;
    std::uint32_t block_rows;
    float symbol_duration;
    unsigned derivatives,derivative_mode;
};
template<class T>T read(std::istream& input){T result;if(&input==&std::cin)radar::support::read_stdin(&result,sizeof(result));else input.read(reinterpret_cast<char*>(&result),sizeof(result));return result;}
template<class T>std::vector<T> read_array(std::istream& input,std::size_t size){
    std::vector<T> result(size);if(&input==&std::cin)radar::support::read_stdin(result.data(),result.size()*sizeof(T));else input.read(reinterpret_cast<char*>(result.data()),std::streamsize(result.size()*sizeof(T)));return result;
}
int run(const char* path,unsigned derivatives){
    if(derivatives<1 || derivatives>8)throw std::invalid_argument("Unsupported derivative count");
    std::ifstream input(path,std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    const auto magic=read<std::array<char,8>>(input);
    if(std::string_view(magic.data(),8)!=std::string_view("SHIPAF2\0",8))throw std::invalid_argument("Invalid focus model header");
    const auto stride=read<std::uint32_t>(input),bands=read<std::uint32_t>(input),cells=read<std::uint32_t>(input);
    const auto degree=read<std::uint32_t>(input);auto pixels=read<std::uint32_t>(input);const auto max_pixels=pixels;const auto first_cell=read<std::int32_t>(input);
    constexpr unsigned block_rows=128;
    if(!stride || stride>65536 || stride%block_rows || !bands || bands>32 || !cells || cells>64 || degree>24 || !pixels || pixels>1000000)
        throw std::invalid_argument("Invalid focus model dimensions");
    Parameters p{stride,bands,cells,degree,pixels,stride/block_rows,first_cell,block_rows,.001024F,derivatives,derivatives};
    const auto count=std::size_t(stride)*bands;
    const auto coefficients=read_array<Complex>(input,count*cells*(degree+1));
    const auto coordinates=read_array<Quad>(input,pixels);
    if(input.rdbuf()->sgetc()!=std::char_traits<char>::eof())throw std::invalid_argument("Trailing focus model data");
    radar::gpu::Kernel kernel{code,sizeof(Parameters)};const auto reduce=kernel.pipeline(reduction);
    const auto adjoint=kernel.pipeline(adjoint_code),adjoint_reduce=kernel.pipeline(adjoint_reduction);
    const auto histories=kernel.pipeline(history_code);
    kernel.upload<Complex>(1,coefficients);kernel.upload<Quad>(2,coordinates);
    kernel.storage(3,std::size_t(bands)*pixels*sizeof(Complex));kernel.storage(4,std::max(std::size_t(bands)*p.blocks*(derivatives+1)*pixels*sizeof(Complex),count*((pixels+2047)/2048)*derivatives*sizeof(float)));
    kernel.storage(5,std::max(std::size_t(bands)*(derivatives+1)*pixels*sizeof(Complex),count*derivatives*sizeof(float)));kernel.storage(6,sizeof(std::uint32_t));
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    bool geometry_ready=false;
    for(;;){
        std::cin.exceptions(std::ios::badbit|std::ios::failbit);
        std::uint32_t mode;if(!radar::support::read_stdin(&mode,sizeof(mode),true))break;
        if(mode>8)throw std::invalid_argument("Unknown focus command");
        if(mode==5){
            const auto next_pixels=read<unsigned>(std::cin);
            if(!next_pixels || next_pixels>max_pixels)throw std::invalid_argument("Invalid focus pixel update");
            const auto points=read_array<Quad>(std::cin,next_pixels);kernel.upload<Quad>(2,points);pixels=next_pixels;p.pixels=pixels;
            constexpr unsigned success=0;radar::support::write_stdout(&success,4);std::cin.exceptions(std::ios::badbit);continue;
        }
        if(mode==2){
            const auto next=read_array<Complex>(std::cin,count*cells*(degree+1));
            const auto points=read_array<Quad>(std::cin,max_pixels);pixels=max_pixels;p.pixels=pixels;
            kernel.upload<Complex>(1,next);kernel.upload<Quad>(2,points);geometry_ready=false;
            constexpr unsigned success=0;radar::support::write_stdout(&success,4);
            std::cin.exceptions(std::ios::badbit);continue;
        }
        auto trial=p;
        if(mode==4){trial.derivative_mode=read<unsigned>(std::cin);if(!trial.derivative_mode || trial.derivative_mode>derivatives)throw std::invalid_argument("Invalid active derivative count");}
        if(mode==3 || mode==7 || mode==8)trial.derivative_mode=0;
        if(mode==0 || mode==3){const auto geometry=read_array<Quad>(std::cin,(10+8*derivatives)*count);kernel.upload<Quad>(0,geometry);geometry_ready=true;}
        else if(mode==1 || mode==4){const auto weight=read_array<Complex>(std::cin,std::size_t(bands)*pixels);kernel.upload<Complex>(3,weight);}
        if(!geometry_ready)throw std::invalid_argument("Load geometry before evaluating or differentiating pixels");
        if(mode==8)kernel.storage(5,count*pixels*sizeof(Complex));
        constexpr std::array<std::uint32_t,1> zero{0};kernel.upload<std::uint32_t>(6,zero);
        const auto cmd=kernel.begin();
        if(mode==8){
            kernel.groups(cmd,*histories,trial,(pixels+63)/64,bands*stride);
        }else if(mode==0 || mode==3 || mode==6 || mode==7){
            kernel.groups(cmd,kernel.primary(),trial,(pixels+63)/64,bands*p.blocks);kernel.barrier(cmd);
            kernel.groups(cmd,*reduce,trial,(pixels+63)/64,bands);
        }else{
            kernel.groups(cmd,*adjoint,trial,(pixels+2047)/2048,bands*stride);kernel.barrier(cmd);
            kernel.groups(cmd,*adjoint_reduce,trial,(bands*stride+63)/64,1);
        }
        kernel.finish(cmd);
        const auto status=*static_cast<const std::uint32_t*>(kernel.view(6).mapped);
        radar::support::write_stdout(&status,sizeof(status));
        const auto output=kernel.view(5);
        radar::support::write_stdout(output.mapped,mode==8 ? count*pixels*sizeof(Complex) : std::size_t((mode==0 || mode==3 || mode==6 || mode==7) ? std::size_t(bands)*((mode==3 || mode==7) ? 1 : derivatives+1)*pixels*sizeof(Complex) : count*derivatives*sizeof(float)));
        std::cin.exceptions(std::ios::badbit);
    }
    return 0;
}
}
int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(Quad)==16 && sizeof(Complex)==8 && sizeof(Parameters)==44);
    if(argc<2 || argc>3)throw std::invalid_argument("isar_focus_patch MODEL.bin < trial-geometry-packets > complex-images-and-gradients");
    return run(argv[1],argc==3 ? std::stoul(argv[2]) : 5);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
