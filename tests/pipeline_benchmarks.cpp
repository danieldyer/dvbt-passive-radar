#include "gpu_frontend.hpp"
#include "gpu_pipeline.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <time.h>

using namespace radar;
namespace {
using Timer=std::chrono::steady_clock;
double cpu(clockid_t id){timespec t{};clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
}
// Recorded samples are loaded BEFORE any processing starts. This avoids both
// I/O timing and the opportunity to hide asynchronous FEC inside excluded I/O.
int main(int argc,char** argv){try {
    if(argc<2)throw std::invalid_argument("pipeline_benchmarks INPUT.wav [seconds=10] [channels=1] [workers=0] [slots=0] [timings=1]");
    const double duration=argc>2 ? std::stod(argv[2]) : 10;
    const int profile_mode=argc>6 ? std::stoi(argv[6]) : 1;
    const bool profiled=profile_mode!=0;
    const unsigned channels=argc>3 ? std::stoul(argv[3]) : 1;
    if(duration<=5 || !channels || channels>7)throw std::invalid_argument("Need >5 seconds, 1..7 channels");
    WaveReader wave(argv[1]);
    const auto available=std::min(wave.info.samples,std::uint64_t(duration*wave.info.sample_rate));
    std::vector<C> recording(available);wave.read(0,recording);
    std::vector<double> offsets(channels,4625000);
    auto clock=std::make_shared<SampleClock>(wave.info.sample_rate,offsets,3);
    GpuChannelBank bank(wave.info.sample_rate,offsets);bank.set_clock(clock);
    // Conservative input halo. It contains FIR/interpolation support at the
    // entire supported clock-rate bound; no edge is fabricated by padding.
    const std::uint64_t halo=bank.taps()/2+512;
    bank.origin=(halo+512+bank.decimation-1)/bank.decimation*bank.decimation;
    GpuPipelineConfig config{std::vector<unsigned>(channels,16),bank.capacity(),bank.origin,bank.decimation,8,0,true};
    if(argc>4)config.fec_workers=std::stoul(argv[4]);
    if(argc>5)config.frame_slots=std::stoul(argv[5]);
    GpuPipeline pipeline(config,bank.context(),clock);
    FrontendTimings frontend;GpuPipelineTimings timing;
    std::uint64_t produced=0,first=0,rows=0,packets=0;
    std::vector<std::uint64_t> checksums(channels,0);
    bool measuring=false;double consumer=0,consumer_cpu=0,process_start=0,main_start=0;Timer::time_point start;
    auto consume=[&](const SpectralBatch& s){const auto begin=Timer::now();const auto cpu_begin=cpu(CLOCK_THREAD_CPUTIME_ID);
        for(const auto& packet:s.packets.packets)if(!packet.valid)throw std::runtime_error("Recorded frame failed FEC");
        if(measuring) {
            rows+=s.frame.last-s.frame.first;packets+=s.packets.count;
            auto& checksum=checksums[s.frame.channel];
            for(const auto& packet:s.packets.packets)for(auto byte:packet.bytes)checksum=(checksum*1099511628211ULL)^byte;
        }
        consumer+=std::chrono::duration<double>(Timer::now()-begin).count();consumer_cpu+=cpu(CLOCK_THREAD_CPUTIME_ID)-cpu_begin;
    };
    auto original=[&](std::uint64_t n){const auto segment=clock->at(double(n));return std::int64_t(n)+std::int64_t(std::floor(segment.offset_at(double(n))));};
    for(;;) {
        if(!measuring && double(produced)/8e6>=5) {
            pipeline.drain(consume,profiled ? &timing : nullptr);measuring=true;first=produced;frontend={};timing={};consumer=consumer_cpu=0;
            frontend.detailed=timing.receiver.detailed=timing.frames.detailed=timing.reconstruction.detailed=timing.response.detailed=profile_mode>1;
            start=Timer::now();process_start=cpu(CLOCK_PROCESS_CPUTIME_ID);main_start=cpu(CLOCK_THREAD_CPUTIME_ID);
        }
        const auto centre=bank.origin+produced*bank.decimation;
        if(centre+halo+128>=available)break;
        auto count=std::min<std::uint64_t>(bank.capacity(),(available-centre-halo-128)/bank.decimation);
        if(!count)break;
        clock->update(centre+((halo+63)/64)*64);
        const auto begin=original(centre-halo)-128;
        auto end=original(centre+(count-1)*bank.decimation+halo)+128;
        if(begin<0)throw std::runtime_error("Missing raw left support");
        while(count && end>std::int64_t(available)){count-=std::min(count,std::uint64_t((end-available+bank.decimation-1)/bank.decimation+1));if(count)end=original(centre+(count-1)*bank.decimation+halo)+128;}
        if(!count)break;
        const auto input=bank.process_clocked_device(std::span<const C>{recording}.subspan(begin,end-begin),begin,centre,count,profiled ? &frontend : nullptr);
        pipeline.process(input,consume,profiled ? &timing : nullptr);produced+=count;
    }
    pipeline.finish(consume,profiled ? &timing : nullptr);
    const double seconds=double(produced-first)/8e6,scale=1000/seconds;
    const double process=cpu(CLOCK_PROCESS_CPUTIME_ID)-process_start-consumer_cpu,main=cpu(CLOCK_THREAD_CPUTIME_ID)-main_start-consumer_cpu;
    const double compute=std::chrono::duration<double>(Timer::now()-start).count()-consumer;
    if(!measuring || seconds<=0 || !rows)throw std::runtime_error("No valid measurement");
    auto checksum=checksums.front();for(unsigned c=1;c<channels;++c)checksum^=checksums[c]+0x9e3779b97f4a7c15ULL+(checksum<<6)+(checksum>>2);
    std::cout<<std::setprecision(10)<<"{\"signal_seconds\":"<<seconds<<",\"profiled\":"<<profiled<<",\"channels\":"<<channels<<",\"compute_ms_per_s\":"<<compute*scale
        <<",\"main_cpu_ms_per_s\":"<<main*scale<<",\"process_cpu_ms_per_s\":"<<process*scale
        <<",\"fec_demap_ms_per_s\":"<<timing.fec.demap*scale<<",\"fec_viterbi_ms_per_s\":"<<timing.fec.viterbi*scale
        <<",\"fec_rs_ms_per_s\":"<<timing.fec.reed_solomon*scale<<",\"gpu_frontend_ms_per_s\":"<<frontend.gpu*scale
        <<",\"gpu_receiver_ms_per_s\":"<<timing.receiver.gpu*scale<<",\"gpu_frames_ms_per_s\":"<<timing.frames.gpu*scale
        <<",\"gpu_reconstruction_ms_per_s\":"<<timing.reconstruction.gpu*scale<<",\"gpu_response_ms_per_s\":"<<timing.response.gpu*scale
        <<",\"rows\":"<<rows<<",\"packets\":"<<packets<<",\"checksum\":"<<checksum<<"}\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
