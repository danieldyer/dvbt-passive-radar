// Component timings use the production implementations, warmed data and repeated trials.
// They complement an end-to-end recording run; they are not correctness tests.
#include "pipeline.hpp"
#include "fractional_resampler.hpp"
#include "gpu_residual.hpp"
#include "vulkan_context.hpp"
#include <vkFFT.h>
#include <gnuradio/prefs.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

using namespace radar;
namespace {
using Clock=std::chrono::steady_clock;
std::ostream results(std::cout.rdbuf());
double checksum=0;
template<class Work> double benchmark(std::string_view name,std::size_t units,Work&& work,int trials=7) {
    checksum+=work(); // Warm plans, pages and caches before measuring.
    std::vector<double> seconds;
    for(int trial=0;trial<trials;++trial) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto start=Clock::now();
        checksum+=work();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        seconds.push_back(std::chrono::duration<double>(Clock::now()-start).count());
    }
    std::sort(seconds.begin(),seconds.end());
    const auto median=seconds[seconds.size()/2];
    results<<name<<','<<units<<','<<median<<','<<seconds.front()<<','<<seconds.back()<<'\n'<<std::flush;
    return median;
}
void stages(std::string_view prefix,const ReconstructionTimings& t,double calls) {
    for(const auto& [name,value] : {std::pair{"demap",t.demap},{"viterbi",t.viterbi},
        {"reed_solomon",t.reed_solomon},{"reencode",t.reencode},{"remap",t.remap}})
        std::cerr<<prefix<<'.'<<name<<'='<<value/calls<<" seconds/call\n";
}
std::vector<Symbol> fixture(Modem& modem,Tps mode,std::size_t count) {
    std::mt19937 random(321);
    std::vector<std::uint8_t> bytes((count*5000/188+32)*188);
    for(auto& b:bytes)b=static_cast<std::uint8_t>(random());
    for(std::size_t i=0;i<bytes.size()/188;++i)bytes[i*188]=i%8 ? 0x47 : 0xb8;
    const auto carriers=modem.synthesize(bytes,mode,count);
    std::vector<Symbol> symbols(count);
    for(std::size_t i=0;i<count;++i) {
        auto& s=symbols[i];s.modulo4=static_cast<int>(i%4);
        s.equalized.assign(carriers.begin()+i*6817,carriers.begin()+(i+1)*6817);
        s.measured=s.equalized;
    }
    return symbols;
}
// Benchmark the proposed FFT offload at the CPU/GPU boundary, including both
// copies and completion. Plan creation is excluded, just as for the CPU FFTs.
void gpu_fft_benchmarks() {
    auto instance=gpu::make_instance();auto gpu=gpu::make_device(*instance);
    const auto device=*gpu.device;
    auto pool=device.createCommandPoolUnique({{},gpu.family});auto fence=device.createFenceUnique({});
    VkDevice d=device;VkPhysicalDevice physical=gpu.physical;
    VkQueue queue=device.getQueue(gpu.family,0);VkCommandPool command_pool=*pool;VkFence f=*fence;
    auto check=[](VkFFTResult result){if(result!=VKFFT_SUCCESS)throw std::runtime_error("Benchmark VkFFT error "+std::to_string(result));};
    for(std::size_t n:{8192U,14336U,57344U})for(std::size_t batch:{1U,32U}) {
        std::vector<C> input(n*batch),output(input.size());
        for(std::size_t i=0;i<input.size();++i)input[i]={float(i%31)/31,float(i%23)/23};
        Fft cpu(n);auto expected=input;
        for(std::size_t b=0;b<batch;++b)cpu.transform(std::span{expected}.subspan(b*n,n));
        const auto bytes=input.size()*sizeof(C);auto buffer=gpu::make_buffer(gpu,bytes);
        VkBuffer handle=*buffer.handle;std::uint64_t buffer_bytes=bytes;
        VkFFTConfiguration config{};config.FFTdim=1;config.size[0]=n;config.numberBatches=batch;
        config.device=&d;config.physicalDevice=&physical;config.queue=&queue;config.commandPool=&command_pool;
        config.fence=&f;config.buffer=&handle;config.bufferSize=&buffer_bytes;
        struct Plan {
            VkFFTApplication app{};
            explicit Plan(VkFFTConfiguration config) {
                const auto result=initializeVkFFT(&app,config);
                if(result!=VKFFT_SUCCESS)throw std::runtime_error("Benchmark VkFFT initialization error "+std::to_string(result));
            }
            ~Plan(){deleteVkFFT(&app);}
        } plan(config);
        auto commands=device.allocateCommandBuffersUnique({*pool,vk::CommandBufferLevel::ePrimary,1});
        auto cmd=*commands.front();cmd.begin(vk::CommandBufferBeginInfo{});
        VkCommandBuffer command=cmd;VkFFTLaunchParams launch{};launch.commandBuffer=&command;
        check(VkFFTAppend(&plan.app,-1,&launch));
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eHostRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eHost,{},barrier,{},{});cmd.end();
        const auto name="fft_"+std::to_string(n)+"_batch"+std::to_string(batch);
        benchmark(name+"_vulkan_roundtrip",input.size(),[&]{
            auto* mapped=device.mapMemory(*buffer.memory,0,bytes);std::memcpy(mapped,input.data(),bytes);device.unmapMemory(*buffer.memory);
            device.getQueue(gpu.family,0).submit(vk::SubmitInfo{{},{},cmd});device.waitIdle();
            mapped=device.mapMemory(*buffer.memory,0,bytes);std::memcpy(output.data(),mapped,bytes);device.unmapMemory(*buffer.memory);
            return double(output.front().real());
        });
        double error=0,energy=0;
        for(std::size_t i=0;i<output.size();++i){error+=std::norm(output[i]-expected[i]);energy+=std::norm(expected[i]);}
        const auto relative=std::sqrt(error/energy);std::cerr<<name<<".relative_rms="<<relative<<'\n';
        if(!(relative<2e-6))throw std::runtime_error("CPU/Vulkan FFT benchmark mismatch");
        benchmark(name+"_cpu_copy_transform",input.size(),[&]{
            output=input;for(std::size_t b=0;b<batch;++b)cpu.transform(std::span{output}.subspan(b*n,n));return double(output.front().real());
        });
    }
}
}
int main(int argc,char** argv) {
    // GNU Radio and device discovery also log to cout; keep the CSV stream clean.
    struct Redirect {
        std::streambuf* previous=std::cout.rdbuf(std::cerr.rdbuf());
        ~Redirect(){std::cout.rdbuf(previous);}
    } redirect;
    try {
        gr::prefs::singleton()->set_string("LOG","log_file","stderr");
        std::filesystem::path input;bool gpu=false;
        for(int i=1;i<argc;++i) {
            const std::string_view arg=argv[i];
            if(arg=="--gpu")gpu=true;
            else if(arg=="--input" && i+1<argc)input=argv[++i];
            else throw std::runtime_error("Usage: radar_benchmarks [--gpu] [--input IQ.wav]");
        }
        results<<std::setprecision(9)<<"component,units,median_seconds,min_seconds,max_seconds\n";
        std::mt19937 random(123);std::uniform_real_distribution<float> uniform(-1,1);
        auto signal=[&](std::size_t n){std::vector<C> v(n);for(auto& x:v)x={uniform(random),uniform(random)};return v;};
        for(const std::size_t n:{57344U,14336U,8192U,2304U}) {
            Fft fft(n);auto seed=signal(n),work=seed;
            for(bool inverse:{false,true})benchmark("fft_"+std::to_string(n)+(inverse ? "_inverse" : "_forward"),n*128,[&]{
                for(int i=0;i<128;++i){work=seed;fft.transform(work,inverse);}return double(work[0].real());
            });
        }
        FractionalResampler resampler;const auto samples=signal(65536+96);
        benchmark("resampler_96_taps",65536,[&]{
            double sum=0;for(std::size_t i=0;i<65536;++i)sum+=resampler.interpolate(samples.data()+i,(i%2048+.37)/2048).real();return sum;
        });
        Modem modem;Tps mode;mode.mode=1;mode.guard=1;mode.rate=2;
        for(int constellation=0;constellation<3;++constellation)for(int rate=0;rate<5;++rate) {
            mode.constellation=constellation;mode.rate=rate;
            const auto symbols=fixture(modem,mode,136);ReconstructionTimings timing;
            const auto name="reconstruct_m"+std::to_string(2+2*constellation)+"_rate"+std::to_string(rate);
            benchmark(name,symbols.size()*6048,[&]{auto r=modem.reconstruct(symbols,mode,&timing);return double(r.packets-r.bad_packets);},3);
            stages(name,timing,4);
        }
        const auto x=signal(1020*6817);auto y=x;std::vector<C> h(6817,C{.95F,.02F});
        for(std::size_t i=0;i<x.size();++i)y[i]=x[i]*h[i%h.size()]+C{.01F,-.01F};
        benchmark("response_order8",x.size(),[&]{auto r=fit_minimum_phase(y,x,6817,8,16);return r.relative_error;},5);
        if(gpu) {
            GpuResidual residual;std::vector<ResidualTimings> trials;trials.reserve(8);
            benchmark("residual_vulkan_transfer_dispatch_readback",x.size(),[&]{
                ResidualTimings timing;auto r=residual.calculate(y,x,h,&timing);trials.push_back(timing);return double(r.front().real());
            });
            for(const auto& [name,member]:{std::pair{"upload",&ResidualTimings::upload},
                {"dispatch_wait",&ResidualTimings::dispatch_wait},{"readback",&ResidualTimings::readback}}) {
                std::vector<double> values;
                for(std::size_t i=1;i<trials.size();++i)values.push_back(trials[i].*member);
                std::sort(values.begin(),values.end());std::cerr<<"residual."<<name<<'='<<values[values.size()/2]<<" median seconds/call\n";
            }
        }
        std::vector<C> residual(x.size());
        benchmark("residual_cpu",x.size(),[&]{
            for(std::size_t i=0;i<x.size();++i)residual[i]=(y[i]-h[i%h.size()]*x[i])*std::conj(x[i]);return double(residual.front().real());
        });
        std::vector<RangeProfile> rows(2297);
        for(std::size_t t=0;t<rows.size();++t) {rows[t].values=std::make_shared<std::vector<C>>(signal(801));rows[t].time=t*.004352;}
        for(bool dc:{false,true})benchmark(dc ? "doppler_10s_dc" : "doppler_10s",rows.size()*801,[&]{
            auto map=make_doppler_map(rows,.004352,100,dc);return double(map.values.front().real());
        });
        benchmark("coherence_10s",rows.size(),[&]{double value=0;return double(coherent_suffix(rows,.996,value));});
        const auto scratch=std::filesystem::temp_directory_path()/
            ("radar-benchmark-"+std::to_string(Clock::now().time_since_epoch().count()));
        struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove_all(path,e);}} cleanup{scratch};
        RangeTimings range;MapOptions options;options.history_seconds=10;options.update_seconds=1e9;
        benchmark("range_ifft_filter_1020_symbols",1020*6817,[&]{
            RangeDoppler maps(options,.001088,scratch);
            for(std::size_t row=0;row<1020;++row)maps.append(std::span{x}.subspan(row*6817,6817),row*.001088,0,0,true,0,&range);
            maps.finish();return double(maps.published());
        },3);
        std::cerr<<"range.inverse="<<range.inverse/4<<" range.filter="<<range.filter/4<<" seconds/1020-symbol-call\n";
        if(!input.empty()) {
            WaveReader wave(input);std::vector<C> data(56318);
            benchmark("wav_read_validate",data.size()*256,[&]{
                double sum=0;for(int i=0;i<256;++i){wave.read(std::uint64_t{56000000}+i*56315,data);sum+=data.front().real();}return sum;
            },3);
            ChannelizerTimings timing;
            benchmark("channelize_1s",8000000,[&]{
                Channelizer channel(wave,4625000,1,1);double sum=0;
                for(;;){auto b=channel.next(&timing);if(b.empty())break;sum+=b.front().real();}return sum;
            },3);
            for(const auto& [name,value] : {std::pair{"read",timing.read},{"mix",timing.mix},
                {"forward",timing.forward},{"fold",timing.fold},{"inverse",timing.inverse}})
                std::cerr<<"channelize."<<name<<'='<<value/4<<" seconds/recording-second\n";
            Channelizer channel(wave,4625000,1,1.3);std::vector<C> iq;
            for(;;){auto b=channel.next();if(b.empty())break;iq.insert(iq.end(),b.begin(),b.end());}
            std::vector<Symbol> symbols;
            benchmark("synchronization_1.3s",iq.size(),[&]{symbols.clear();synchronize_samples(iq,16,[&](Symbol s){symbols.push_back(std::move(s));});return double(symbols.size());},3);
            std::size_t first=0;
            while(first+68<=symbols.size() && !modem.tps(std::span{symbols}.subspan(first,68),mode))++first;
            if(first+68>symbols.size())throw std::runtime_error("Benchmark recording did not acquire validated TPS");
            const auto count=std::min<std::size_t>(1020,(symbols.size()-first)/68*68);
            const auto batch=std::span{symbols}.subspan(first,count);ReconstructionTimings real;
            benchmark("reconstruct_recording",count*6048,[&]{auto r=modem.reconstruct(batch,mode,&real);return double(r.packets-r.bad_packets);},3);
            stages("reconstruct_recording",real,4);
        }
        if(gpu)gpu_fft_benchmarks();
        std::cerr<<"checksum="<<checksum<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
