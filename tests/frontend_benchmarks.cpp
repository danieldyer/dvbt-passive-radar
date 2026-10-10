#include "gpu_frontend.hpp"
#include <iomanip>
#include <iostream>
#include <random>

using namespace radar;
int main(int argc,char**) {
    try {
        double checksum=0;
        std::mt19937 random(33);std::uniform_real_distribution<float> uniform(-1,1);
        std::cout<<std::setprecision(9)<<"component,channels,gpu_seconds_per_signal_second,stage0,stage1,stage2,stage3,stage4\n";
        std::cerr<<"Stage layouts: channelizer = input marker / forward FFT / fold / inverse FFT / NCO; "
            "shared_clock_channelizer = forward FFT / boundary correction / fused synthesis / marker / marker\n";
        for(bool clocked:{false,true}) {
        for(std::size_t channels:{1,5,7}) {
            std::vector<double> offsets;
            for(std::size_t c=0;c<channels;++c)offsets.push_back((double(c)-3)*7e6+123.375);
            GpuChannelBank bank(56000000,offsets);
            auto clock=std::make_shared<SampleClock>(56000000,offsets,3);
            if(clocked) {clock->schedule(64,1.000005);bank.set_clock(clock);}
            const auto count=bank.capacity();
            std::vector<C> samples((count-1)*7+bank.taps()+2000);
            for(auto& z:samples)z={uniform(random),uniform(random)};
            auto run=[&](FrontendTimings* timing=nullptr) {
                return clocked ? bank.process_clocked(samples,6000,7000,count,timing) : bank.process(samples,7000,count,timing);
            };
            for(unsigned i=0;i<8;++i)(void)run();
            FrontendTimings timing;
            timing.detailed=argc>1;
            constexpr unsigned iterations=64;
            for(unsigned i=0;i<iterations;++i) {
                const auto result=run(&timing);
                checksum+=result[0][0].real();
            }
            if(!timing.gpu_timestamps_available)throw std::runtime_error("Device timestamps unavailable; cannot report kernel execution time");
            const double duration=double(iterations*count)/8e6;
            std::cout<<(clocked ? "shared_clock_channelizer," : "channelizer,")<<channels<<','<<timing.gpu/duration;
            for(double stage:timing.stages)std::cout<<','<<stage/duration;
            std::cout<<'\n'<<std::flush;
        }
        }
        GpuResampler gpu;
        constexpr std::size_t count=8*8704;
        std::vector<C> samples(count+100);
        for(auto& z:samples)z={uniform(random),uniform(random)};
        for(unsigned i=0;i<4;++i)(void)gpu.process(samples,47.37,1.000005,count);
        FrontendTimings timing;
        for(unsigned i=0;i<256;++i) {
            const auto result=gpu.process(samples,47.37,1.000005,count,&timing);
            checksum+=result.front().real();
        }
        std::cout<<"resampler,1,"<<timing.gpu/(256*count/8e6)<<",,,,,\n";
        std::cerr<<"checksum="<<checksum<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
