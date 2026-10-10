#include "gpu_frontend.hpp"
#include "spectral_clock.hpp"
#include "gpu_compute.hpp"
#include "clock_interpolator.hpp"
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

using namespace radar;
namespace {
constexpr std::uint32_t reference_code[] =
#include "raw_clock_reference.inc"
;
double reference(std::shared_ptr<gpu::Context> context,std::span<const double> centres,std::size_t count) {
    double highest=0;for(double f:centres)highest=std::max(highest,std::abs(f)+3328125);
    ClockInterpolator filter(highest/56e6);
    gpu::Kernel kernel(reference_code,8,std::array<std::uint32_t,1>{filter.taps},context);
    kernel.storage(0,count*sizeof(C));
    std::vector<C> input(count+2000,C{.1f,.3f});kernel.upload<C>(5,input);kernel.upload<float>(6,filter.coefficients);
    std::vector<std::array<float,4>> coordinates(2*((count+63)/64));
    for(std::size_t g=0;g<coordinates.size()/2;++g) {
        const double offset=.371+double(g*64)*.000005;
        coordinates[2*g]={std::bit_cast<float>(std::int32_t(filter.left+g*64+std::size_t(std::floor(offset)))),
            float(offset-std::floor(offset)),.000005f,.000005f};
        coordinates[2*g+1]={64,0,0,0};
    }
    kernel.upload<std::array<float,4>>(7,coordinates);
    struct Push {std::uint32_t count,padded;};const Push push{std::uint32_t(count),std::uint32_t(count)};
    for(unsigned i=0;i<32;++i)kernel.run(push,count);
    ComputeTimings timing;
    for(unsigned i=0;i<128;++i)kernel.run(push,count,&timing);
    if(!timing.gpu_timestamps_available)throw std::runtime_error("GPU timestamps unavailable");
    return timing.gpu/(128*double(count)/56e6);
}
}
int main(int argc,char** argv) {
    try {
        const unsigned iterations=argc>1 ? unsigned(std::stoul(argv[1])) : 1024;
        if(iterations<16 || iterations>100000)throw std::runtime_error("Use 16..100000 measurement pairs");
        std::cerr<<"Measured pairs per channel count: "<<iterations<<'\n';
        std::mt19937 random(39);std::uniform_real_distribution<float> sample(-1,1);
        std::cout<<std::setprecision(12)<<"channels,raw_resampler_ms_s,identity_frontend_ms_s,clocked_frontend_ms_s,incremental_clock_ms_s,paired_standard_error_ms_s\n";
        for(unsigned channels:{1U,5U,7U}) {
            std::vector<double> centres;
            for(unsigned c=0;c<channels;++c)centres.push_back(channels==1 ? 4625000 : (double(c)-3)*7e6+123.375);
            auto context=std::make_shared<gpu::Context>();
            const auto prototype=GpuChannelBank::prototype(56e6);
            SpectralClock bank(56000000,centres,prototype,context);
            const auto count=bank.capacity();
            std::vector<C> input((count-1)*7+2*bank.support()+2000);
            for(auto& value:input)value={sample(random),sample(random)};
            const std::array<ClockSegment,1> identity{{{0,0,1}}},clocked{{{0,0,1.000005}}};
            auto run=[&](bool clock,ComputeTimings* timing=nullptr) {
                bank.process(input,6000,7000,count,clock ? std::span{clocked} : std::span{identity},timing);
            };
            for(unsigned i=0;i<32;++i){run(false);run(true);}
            ComputeTimings timing[2];double mean=0,squares=0;
            for(unsigned i=0;i<iterations;++i) {
                double elapsed[2]{};
                for(unsigned j=0;j<2;++j) {
                    const unsigned which=(i+j)%2;const double previous=timing[which].gpu;
                    run(which!=0,&timing[which]);elapsed[which]=timing[which].gpu-previous;
                }
                const double difference=elapsed[1]-elapsed[0],delta=difference-mean;
                mean+=delta/(i+1);squares+=delta*(difference-mean);
            }
            if(!timing[0].gpu_timestamps_available || !timing[1].gpu_timestamps_available)throw std::runtime_error("GPU timestamps unavailable");
            const double duration=double(count)/8e6;
            const double raw=reference(context,centres,count*7);
            std::cout<<channels<<','<<raw*1000<<','<<timing[0].gpu/(iterations*duration)*1000<<','
                <<timing[1].gpu/(iterations*duration)*1000<<','<<mean/duration*1000<<','
                <<std::sqrt(squares/(iterations-1)/iterations)/duration*1000<<'\n'<<std::flush;
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
