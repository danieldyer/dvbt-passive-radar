#include "gpu_frontend.hpp"
#include "gpu_pipeline.hpp"
#include "matched_detector.hpp"
#include "recording_metadata.hpp"
#include "live_input.hpp"
#include <bit>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <time.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

using namespace radar;
namespace {
using Timer=std::chrono::steady_clock;
double cpu(clockid_t id){timespec t{};clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
}
// Resident mode excludes all reads by preloading before processing starts.
// Stream/live modes include reads, so asynchronous FEC cannot hide in excluded I/O.
int main(int argc,char** argv){try {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
#endif
    if(argc<2)throw std::invalid_argument("detection_pipeline_benchmark INPUT.wav [seconds=20] [channels=1] [workers=0] [slots=0] [timings=1] [detector=1] [metadata.json] [frames_per_batch=15] [input=resident|stream|live] [query_mask=511]");
    const double duration=argc>2 ? std::stod(argv[2]) : 20;
    const int detector_mode=argc>7 ? std::stoi(argv[7]) : 1;
    const int profile_mode=argc>6 ? std::stoi(argv[6]) : 1;
    const bool profiled=profile_mode!=0;
    const unsigned channels=argc>3 ? std::stoul(argv[3]) : 1;
    const std::string input_mode=argc>10 ? argv[10] : "resident";
    if(input_mode!="resident" && input_mode!="stream" && input_mode!="live")throw std::invalid_argument("Invalid benchmark input mode");
    const bool streaming=input_mode!="resident";
    const unsigned query_mask=argc>11 ? std::stoul(argv[11]) : 511;
    if(duration<=10 || !channels || channels>7 || detector_mode<0 || detector_mode>3)throw std::invalid_argument("Need >10 seconds, 1..7 channels, detector mode 0..3");
    WaveReader wave(argv[1]);
    const auto available=std::min(wave.info.samples,std::uint64_t(duration*wave.info.sample_rate));
    std::vector<C> recording;
    // Do not retain WaveReader's equally large int16 conversion scratch for
    // the entire benchmark: that is not part of the production working set.
    if(!streaming){recording.resize(available);WaveReader preload(argv[1]);preload.read(0,recording);}
    std::vector<double> offsets(channels,4625000);
    SarCapture capture;capture.recording_rf_hz=553e6;capture.source_rate=wave.info.sample_rate;
    for(unsigned c=0;c<channels;++c)capture.channels.push_back({557625000,4625000,16});
    if(argc>8){
        const auto metadata=RecordingMetadata::read(argv[8]);metadata.validate_wave(wave.info);capture={};metadata.configure(capture);capture.source_rate=wave.info.sample_rate;
        if(channels>capture.channels.size())throw std::invalid_argument("More channels than declared signals");
        capture.channels.resize(channels);
        for(unsigned c=0;c<channels;++c)offsets[c]=capture.channels[c].nco_hz;
    }
    GpuChannelBank bank(wave.info.sample_rate,offsets);
    auto clock=std::make_shared<SampleClock>(bank.coordinate_rate(),offsets,3);bank.set_clock(clock);
    std::unique_ptr<GpuLiveInput> live;
    std::ifstream native_input;
    if(input_mode=="live"){
        if(wave.info.sample_bytes!=sizeof(Iq16))throw std::invalid_argument("Live benchmark input requires int16 IQ");
        live=std::make_unique<GpuLiveInput>(bank.context(),std::bit_ceil(bank.input_capacity()),bank.input_capacity());
        native_input.exceptions(std::ios::failbit|std::ios::badbit);native_input.open(argv[1],std::ios::binary);
    }
    // Conservative input halo. It contains FIR/interpolation support at the
    // entire supported clock-rate bound; no edge is fabricated by padding.
    const std::uint64_t halo=bank.taps()/2+512;
    bank.origin=(halo+512+bank.decimation-1)/bank.decimation*bank.decimation;
    std::vector<unsigned> guards;for(const auto& c:capture.channels)guards.push_back(c.guard);
    GpuPipelineConfig config{guards,bank.capacity(),bank.origin,bank.decimation,8,0,true};
    if(argc>4)config.fec_workers=std::stoul(argv[4]);
    if(argc>5)config.frame_slots=std::stoul(argv[5]);
    if(argc>9)config.frames_per_batch=std::stoul(argv[9]);
    config.residual_spectra=detector_mode!=1 && detector_mode!=3;
    GpuPipeline pipeline(config,bank.context(),clock);
    capture.input_rate=bank.coordinate_rate();capture.decimation=bank.decimation;capture.origin=bank.origin;
    std::unique_ptr<MatchedDetector> detector;
    std::vector<std::unique_ptr<RangeDoppler>> legacy;
    if(detector_mode==1 || detector_mode==3){DetectionOptions options;options.focus.doppler_hz=100;options.publish_maps=detector_mode==1;detector=std::make_unique<MatchedDetector>(bank.context(),channels,options);}
    if(detector_mode==2)for(unsigned c=0;c<channels;++c){
        MapOptions options;options.integration_seconds=options.history_seconds=3;options.update_seconds=1;
        options.doppler_hz=200;options.slow_decimation=2;options.remove_dc=true;
        legacy.push_back(std::make_unique<RangeDoppler>(options,.001088,std::filesystem::path{}));
    }
    DetectionTimings detection_timing;std::uint64_t maps=0;double detection_checksum=0;
    std::vector<std::uint64_t> legacy_start_counts(channels);

    FrontendTimings frontend,unpacking;GpuPipelineTimings timing;
    auto configure_timings=[&]{
        // Selective queries diagnose instrumentation effects without changing
        // the work, command grouping or CPU timing counters.
        std::array<ComputeTimings*,9> stages{&frontend,&unpacking,&timing.receiver,&timing.frames,
            &timing.reconstruction,&timing.response,&detection_timing.noise,&detection_timing.profiles,&detection_timing.focus};
        for(unsigned i=0;i<stages.size();++i){stages[i]->timestamp_queries=(query_mask>>i)&1U;stages[i]->detailed=profile_mode>1;}
    };
    configure_timings();
    std::uint64_t produced=0,first=0,rows=0,packets=0;
    std::vector<std::uint64_t> checksums(channels,0);
    bool measuring=false;double consumer=0,consumer_cpu=0,process_start=0,main_start=0,input_read=0;Timer::time_point start;
    auto map_consumer=[&](std::size_t,DopplerMap&& map){
        if(measuring)++maps;
        detection_checksum+=std::norm(map.values[map.values.size()/2]);};
    auto consume=[&](const SpectralBatch& s){const auto begin=Timer::now();const auto cpu_begin=cpu(CLOCK_THREAD_CPUTIME_ID);
        for(const auto& packet:s.packets.packets)if(!packet.valid)throw std::runtime_error("Recorded frame failed FEC");
        if(measuring) {
            rows+=s.frame.last-s.frame.first;packets+=s.packets.count;
            auto& checksum=checksums[s.frame.channel];
            for(const auto& packet:s.packets.packets)for(auto byte:packet.bytes)checksum=(checksum*1099511628211ULL)^byte;
        }
        consumer+=std::chrono::duration<double>(Timer::now()-begin).count();consumer_cpu+=cpu(CLOCK_THREAD_CPUTIME_ID)-cpu_begin;
        if(detector)detector->append(SarBatch(capture,s,clock->segments()),map_consumer,measuring && profiled ? &detection_timing : nullptr);
        if(!legacy.empty()){
            const auto* cross=static_cast<const C*>(s.residual.cross.mapped);
            const auto* valid=static_cast<const std::uint32_t*>(s.transmitted.valid.mapped);
            for(std::size_t row=s.frame.first;row<s.frame.last;++row){bool complete=true;
                for(unsigned k=0;k<6817;++k)complete&=valid[row*6817+k]!=0;
                legacy[s.frame.channel]->append({cross+row*6817,6817},double(capture.origin+s.frame.symbols[row].window.sample*capture.decimation)/capture.input_rate,
                                                 0,0,complete,s.frame.segment);
            }
        }
    };
    for(;;) {
        if(!measuring && double(produced)/8e6>=10) {
            pipeline.drain(consume,profiled ? &timing : nullptr);
            if(detector)detector->finish(map_consumer);
            measuring=true;first=produced;frontend={};unpacking={};timing={};detection_timing={};maps=0;consumer=consumer_cpu=0;
            configure_timings();
            for(unsigned c=0;c<legacy.size();++c)legacy_start_counts[c]=legacy[c]->published();
            start=Timer::now();process_start=cpu(CLOCK_PROCESS_CPUTIME_ID);main_start=cpu(CLOCK_THREAD_CPUTIME_ID);
        }
        const auto centre=bank.origin+produced*bank.decimation;
        const auto count=bank.capacity();bank.update_clock(centre);
        const auto support=bank.input_support(centre,count);
        if(support.first<0)throw std::runtime_error("Missing raw left support");
        if(support.last>std::int64_t(available))break;
        std::span<const C> samples;
        ChannelBatch input;
        if(live){
            const auto read_start=Timer::now();
            native_input.seekg(wave.info.data_offset+support.first*sizeof(Iq16));
            native_input.read(reinterpret_cast<char*>(live->storage().data()),(support.last-support.first)*sizeof(Iq16));
            if(measuring)input_read+=std::chrono::duration<double>(Timer::now()-read_start).count();
            const auto raw=live->gather(0,support.last-support.first,profiled ? &unpacking : nullptr);
            input=bank.process_clocked_device(raw,support.last-support.first,support.first,centre,count,profiled ? &frontend : nullptr);
        }else if(streaming){
            const auto read_start=Timer::now();recording.resize(support.last-support.first);wave.read(support.first,recording);samples=recording;
            if(measuring)input_read+=std::chrono::duration<double>(Timer::now()-read_start).count();
        }else samples=std::span<const C>{recording}.subspan(support.first,support.last-support.first);
        if(!live)input=bank.process_clocked_device(samples,support.first,centre,count,profiled ? &frontend : nullptr);
        pipeline.process(input,consume,profiled ? &timing : nullptr);produced+=count;
    }
    pipeline.finish(consume,profiled ? &timing : nullptr);
    if(detector)detector->finish(map_consumer,profiled ? &detection_timing : nullptr);
    for(unsigned c=0;c<legacy.size();++c){legacy[c]->finish();maps+=legacy[c]->published()-legacy_start_counts[c];}
    const double seconds=double(produced-first)/8e6,scale=1000/seconds;
    const double process=cpu(CLOCK_PROCESS_CPUTIME_ID)-process_start-consumer_cpu,main=cpu(CLOCK_THREAD_CPUTIME_ID)-main_start-consumer_cpu;
    const double compute=std::chrono::duration<double>(Timer::now()-start).count()-consumer;
    if(!measuring || seconds<=0 || !rows)throw std::runtime_error("No valid measurement");
    auto checksum=checksums.front();for(unsigned c=1;c<channels;++c)checksum^=checksums[c]+0x9e3779b97f4a7c15ULL+(checksum<<6)+(checksum>>2);
    std::cout<<std::setprecision(10)<<"{\"signal_seconds\":"<<seconds<<",\"profiled\":"<<profiled<<",\"query_mask\":"<<query_mask<<",\"channels\":"<<channels<<",\"compute_ms_per_s\":"<<compute*scale
        <<",\"input_mode\":\""<<input_mode<<"\",\"input_read_ms_per_s\":"<<input_read*scale
        <<",\"wall_including_validation_ms_per_s\":"<<(compute+consumer)*scale
        <<",\"frames_per_batch\":"<<config.frames_per_batch
        <<",\"main_cpu_ms_per_s\":"<<main*scale<<",\"process_cpu_ms_per_s\":"<<process*scale
        <<",\"fec_demap_ms_per_s\":"<<timing.fec.demap*scale<<",\"fec_viterbi_ms_per_s\":"<<timing.fec.viterbi*scale
        <<",\"fec_rs_ms_per_s\":"<<timing.fec.reed_solomon*scale<<",\"gpu_frontend_ms_per_s\":"<<frontend.gpu*scale
        <<",\"gpu_unpack_ms_per_s\":"<<unpacking.gpu*scale
        <<",\"gpu_receiver_ms_per_s\":"<<timing.receiver.gpu*scale<<",\"gpu_frames_ms_per_s\":"<<timing.frames.gpu*scale
        <<",\"gpu_reconstruction_ms_per_s\":"<<timing.reconstruction.gpu*scale<<",\"gpu_response_ms_per_s\":"<<timing.response.gpu*scale
        <<",\"gpu_reencode_ms_per_s\":"<<timing.reconstruction.stages[0]*scale
        <<",\"gpu_remap_ms_per_s\":"<<timing.reconstruction.stages[1]*scale
        <<",\"gpu_response_reduce_ms_per_s\":"<<timing.response.stages[0]*scale
        <<",\"gpu_response_solver_ms_per_s\":"<<timing.response.stages[1]*scale
        <<",\"detector_mode\":"<<detector_mode<<",\"maps\":"<<maps
        <<",\"gpu_noise_ms_per_s\":"<<detection_timing.noise.gpu*scale
        <<",\"gpu_profiles_ms_per_s\":"<<detection_timing.profiles.gpu*scale
        <<",\"gpu_focus_ms_per_s\":"<<detection_timing.focus.gpu*scale
        <<",\"calibration_cpu_ms_per_s\":"<<0
        <<",\"standardization_cpu_ms_per_s\":"<<0
        <<",\"detection_checksum\":"<<detection_checksum
        <<",\"rows\":"<<rows<<",\"packets\":"<<packets<<",\"checksum\":"<<checksum<<"}\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
