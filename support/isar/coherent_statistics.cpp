// Offline five-band coherent GLRT. Inputs are already whitened and phase
// calibrated complex amplitudes. This never changes the scientific images.
#include "gpu_compute.hpp"
#include "binary_pipe.hpp"
#include <array>
#include <bit>
#include <complex>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr std::uint32_t code[]=
#include "coherent_glrt.inc"
;
struct Parameters { std::uint32_t pixels; };
int run(){
    radar::gpu::Kernel kernel{code,sizeof(Parameters)};
    std::vector<std::complex<float>> input;
    for(;;){
        Parameters p{};
        if(!radar::support::read_stdin(&p,sizeof(p),true))return 0;
        if(!p.pixels || p.pixels>4'000'000)throw std::invalid_argument("Invalid coherent-statistic image size");
        input.resize(std::size_t(p.pixels)*5);
        radar::support::read_stdin(input.data(),input.size()*sizeof(input[0]));
        kernel.upload<std::complex<float>>(0,input);kernel.storage(1,std::size_t(p.pixels)*2*sizeof(float));
        kernel.run(p,p.pixels);
        radar::support::write_stdout(&p.pixels,sizeof(p.pixels));
        radar::support::write_stdout(kernel.view(1).mapped,std::size_t(p.pixels)*2*sizeof(float));
    }
}
}
int main(){try{
    static_assert(std::endian::native==std::endian::little && sizeof(std::complex<float>)==8 && sizeof(Parameters)==4);
    return run();
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
