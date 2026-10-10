#include "gpu_frontend.hpp"
#include "sar_handoff.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace radar;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv){try {
    if(argc<2)throw std::invalid_argument("fec_pipeline_benchmark INPUT.wav [seconds=60] [channels=1]");
    const double duration=argc>2 ? std::stod(argv[2]) : 60;
    const unsigned channels=argc>3 ? std::stoul(argv[3]) : 1;
    if(duration<=5 || !channels || channels>7)throw std::invalid_argument("Need >5 seconds, 1..7 channels");
    WaveReader wave(argv[1]);std::vector<double> offsets(channels,4625000);
    auto clock=std::make_shared<SampleClock>(wave.info.sample_rate,offsets,3);
    GpuChannelBank bank(wave,offsets,0,duration);bank.set_clock(clock);
    GpuPipeline pipeline({std::vector<unsigned>(channels,16),bank.capacity(),bank.origin,bank.decimation,8,0,true},bank.context(),clock);
    FrontendTimings frontend;GpuPipelineTimings timing;
    double compute=0,consumer=0;std::uint64_t first=0,packets=0,bad=0,corrected=0,rows=0,checksum=0;bool measuring=false;
    auto consume=[&](const SpectralBatch& s){const auto begin=Clock::now();
        if(measuring) {
            rows+=s.frame.last-s.frame.first;packets+=s.packets.count;
            for(const auto& p:s.packets.packets){bad+=!p.valid;corrected+=p.corrected;for(auto b:p.bytes)checksum=(checksum*1099511628211ULL)^b;}
        }
        consumer+=std::chrono::duration<double>(Clock::now()-begin).count();
    };
    for(;;) {
        if(!measuring && bank.produced/8e6>=5){pipeline.drain(consume,&timing);measuring=true;first=bank.produced;frontend={};timing={};}
        const auto begin=Clock::now();const auto read_before=frontend.read;consumer=0;
        const auto input=bank.next_device(&frontend);
        if(input.count)pipeline.process(input,consume,&timing);else pipeline.finish(consume,&timing);
        if(measuring)compute+=std::chrono::duration<double>(Clock::now()-begin).count()-(frontend.read-read_before)-consumer;
        if(!input.count)break;
    }
    const double seconds=(bank.produced-first)/8e6,scale=1000/seconds;
    std::cout<<std::setprecision(10)<<"{\"signal_seconds\":"<<seconds<<",\"channels\":"<<channels<<",\"compute_ms_per_s\":"<<compute*scale
        <<",\"cpu_fec_demap_ms_per_s\":"<<timing.fec.demap*scale<<",\"cpu_fec_viterbi_ms_per_s\":"<<timing.fec.viterbi*scale
        <<",\"cpu_fec_rs_ms_per_s\":"<<timing.fec.reed_solomon*scale<<",\"gpu_frontend_ms_per_s\":"<<frontend.gpu*scale
        <<",\"gpu_ofdm_ms_per_s\":"<<timing.receiver.gpu*scale<<",\"gpu_frames_ms_per_s\":"<<timing.frames.gpu*scale
        <<",\"gpu_reconstruction_ms_per_s\":"<<timing.reconstruction.gpu*scale<<",\"gpu_response_residual_ms_per_s\":"<<timing.response.gpu*scale
        <<",\"packets\":"<<packets<<",\"rejected\":"<<bad<<",\"corrected_bytes\":"<<corrected<<",\"rows\":"<<rows<<",\"packet_checksum\":"<<checksum<<"}\n";
    if(!rows || bad)return 2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
