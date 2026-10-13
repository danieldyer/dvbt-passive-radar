// Replays archived channelised IQ through the production receiver and detector.
// Deliberately excludes channelisation: the saved samples are already retimed.
#include "gpu_pipeline.hpp"
#include "gpu_commands.hpp"
#include "matched_detector.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <time.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

using namespace radar;
namespace {
using Timer=std::chrono::steady_clock;
double cpu(clockid_t id){timespec t{};clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
template<class T>T read(std::istream& input){T value{};input.read(reinterpret_cast<char*>(&value),sizeof(value));return value;}
boost::json::object json(std::istream& input,std::size_t size){
    if(!size || size>1024*1024)throw std::runtime_error("Invalid archive metadata size");
    std::string text(size,'\0');input.read(text.data(),std::streamsize(size));return boost::json::parse(text).as_object();
}
}
int main(int argc,char** argv){try{
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
#endif
    if(argc<2)throw std::invalid_argument("live_pipeline_benchmark CAPTURE.dvblive [end_seconds=35] [map_channels=0] [profile=1]");
    const double end=argc>2 ? std::stod(argv[2]) : 35;
    const unsigned map_channels=argc>3 ? std::stoul(argv[3]) : 0;
    const bool profiled=argc>4 ? std::stoi(argv[4])!=0 : true;
    if(end<=15)throw std::invalid_argument("Benchmark needs more than 15 seconds including warmup");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    char magic[8];input.read(magic,8);if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Invalid live archive");
    const auto header=json(input,read<std::uint32_t>(input));
    if(header.at("archive_mode")!="both" && header.at("archive_mode")!="channels")throw std::runtime_error("Channelised IQ is required");
    SarCapture capture;capture.input_rate=boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
    capture.source_rate=boost::json::value_to<std::uint32_t>(header.at("source_nominal_rate_hz"));
    capture.decimation=boost::json::value_to<unsigned>(header.at("decimation"));
    capture.origin=boost::json::value_to<std::uint64_t>(header.at("origin"));
    capture.recording_rf_hz=boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
    GpuPipelineConfig config;config.max_samples=262144;config.frames_per_batch=6;config.residual_spectra=false;
    for(const auto& value:header.at("applied_channels").as_array()){
        const auto& c=value.as_object();const auto guard=boost::json::value_to<unsigned>(c.at("guard"));
        capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),boost::json::value_to<double>(c.at("nco_hz")),guard});
        config.guards.push_back(guard);
    }
    const auto channels=capture.channels.size();if(map_channels>channels)throw std::invalid_argument("Too many displayed channels");
    auto context=std::make_shared<gpu::Context>();GpuPipeline pipeline(config,context,{});
    auto storage=gpu::make_buffer(context->gpu,channels*config.max_samples*sizeof(C));
    auto* mapped=context->gpu.device->mapMemory(*storage.memory,0,channels*config.max_samples*sizeof(C));
    gpu::BufferView samples{context,*storage.handle,channels*config.max_samples*sizeof(C),mapped};
    DetectionOptions options;options.publish_maps=map_channels>0;options.update_seconds=.25;options.focus.doppler_hz=100;
    // A checksum replaces the optional display projector; no full-map readback.
    double profile_checksum=0;std::vector<double> map_checksums(channels);
    MatchedDetector detector(context,channels,options,[&](const auto& map,const auto& values){map_checksums[map.channel]+=std::norm(static_cast<const C*>(values.mapped)[17]);},
        [&](const auto&,const auto& profiles,auto,auto){profile_checksum+=std::norm(profiles.values[profiles.values.size()/2]);});
    for(unsigned c=0;c<channels;++c)detector.set_publish_maps(c,c<map_channels);
    std::vector<ClockSegment> clock;GpuPipelineTimings processing;DetectionTimings detection;
    bool measuring=false;std::uint64_t rows=0,good=0,bad=0,maps=0,samples_processed=0;
    std::vector<std::uint64_t> checksums(channels);
    double io=0,validation=0,validation_cpu=0,main_start=0,process_start=0;
    Timer::time_point start;
    auto publish=[&](std::size_t,DopplerMap&&){if(measuring)++maps;};
    auto consume=[&](const SpectralBatch& batch){
        const auto mark=Timer::now();const auto active=cpu(CLOCK_THREAD_CPUTIME_ID);
        if(measuring){rows+=batch.frame.last-batch.frame.first;
            for(const auto& packet:batch.packets.packets){
                if(!packet.valid){++bad;continue;}++good;
                for(auto byte:packet.bytes)checksums[batch.frame.channel]=(checksums[batch.frame.channel]*1099511628211ULL)^byte;
            }
        }
        validation+=std::chrono::duration<double>(Timer::now()-mark).count();validation_cpu+=cpu(CLOCK_THREAD_CPUTIME_ID)-active;
        detector.append(SarBatch(capture,batch,clock),publish,measuring ? &detection : nullptr);
    };
    for(;;){
        const auto read_start=Timer::now();char tag[4];input.read(tag,4);
        const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);const auto meta=json(input,length);
        const auto payload=input.tellg();const std::string_view type(tag,4);
        if(bytes>64*1024*1024)throw std::runtime_error("Oversized archive payload");
        if(type=="CLCK")for(const auto& value:meta.at("segments").as_array()){
            const auto& s=value.as_array();clock.push_back({boost::json::value_to<std::uint64_t>(s[0]),boost::json::value_to<double>(s[1]),boost::json::value_to<double>(s[2])});
        }
        if(type=="CHIQ"){
            const auto first=boost::json::value_to<std::uint64_t>(meta.at("first"));
            const auto count=boost::json::value_to<std::size_t>(meta.at("count"));
            if(first/8e6>=end)break;
            if(!count || count>config.max_samples || bytes!=channels*count*sizeof(C) ||
               boost::json::value_to<std::size_t>(meta.at("channels"))!=channels || meta.at("order")!="channel,sample")
                throw std::runtime_error("Invalid channel payload");
            input.read(static_cast<char*>(mapped),std::streamsize(bytes));
            if(measuring)io+=std::chrono::duration<double>(Timer::now()-read_start).count();
            if(!measuring && first/8e6>=15){
                pipeline.drain(consume);detector.finish(publish);processing={};detection={};
                for(auto* t:{&processing.receiver,&processing.frames,&processing.reconstruction,&processing.response,&detection.noise,&detection.profiles,&detection.focus}){t->timestamp_queries=profiled;t->detailed=profiled;}
                measuring=true;validation=validation_cpu=0;start=Timer::now();main_start=cpu(CLOCK_THREAD_CPUTIME_ID);process_start=cpu(CLOCK_PROCESS_CPUTIME_ID);
            }
            pipeline.process({samples,first,count,channels,count},consume,measuring ? &processing : nullptr);
            if(measuring)samples_processed+=count;
        }else input.seekg(payload+std::streamoff(bytes));
        if(type=="DONE")break;
    }
    pipeline.finish(consume,measuring ? &processing : nullptr);detector.finish(publish,measuring ? &detection : nullptr);
    const double seconds=samples_processed/8e6,scale=1000/seconds;
    if(!measuring || seconds<=0 || !rows)throw std::runtime_error("No measured output");
    boost::json::object result{{"signal_seconds",seconds},{"channels",channels},{"map_channels",map_channels},{"profiled",profiled},
        {"wall_ms_per_s",(std::chrono::duration<double>(Timer::now()-start).count()-validation)*scale},{"read_ms_per_s",io*scale},
        {"main_cpu_ms_per_s",(cpu(CLOCK_THREAD_CPUTIME_ID)-main_start-validation_cpu)*scale},
        {"process_cpu_ms_per_s",(cpu(CLOCK_PROCESS_CPUTIME_ID)-process_start-validation_cpu)*scale},
        {"viterbi_worker_ms_per_s",processing.fec.viterbi*scale},{"rs_worker_ms_per_s",processing.fec.reed_solomon*scale},
        {"rows",rows},{"rs_good",good},{"rs_bad",bad},{"maps",maps},{"packet_checksums",boost::json::value_from(checksums)},
        {"map_checksum",std::accumulate(map_checksums.begin(),map_checksums.end(),0.)},{"profile_checksum",profile_checksum}};
    auto gpu_time=[&](const char* name,const ComputeTimings& t){result[name]=t.gpu*scale;};
    gpu_time("receiver_gpu_ms_per_s",processing.receiver);gpu_time("frames_gpu_ms_per_s",processing.frames);
    gpu_time("reconstruction_gpu_ms_per_s",processing.reconstruction);gpu_time("response_gpu_ms_per_s",processing.response);
    gpu_time("noise_gpu_ms_per_s",detection.noise);gpu_time("profiles_gpu_ms_per_s",detection.profiles);gpu_time("focus_gpu_ms_per_s",detection.focus);
    std::cout<<boost::json::serialize(result)<<'\n';
    context->gpu.device->unmapMemory(*storage.memory);
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
