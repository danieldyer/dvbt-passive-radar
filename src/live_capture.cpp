#include "live_capture.hpp"
#include "capture_source.hpp"
#include "live_archive.hpp"
#include "capture_diagnostics.hpp"
#include "gpu_commands.hpp"
#include <boost/json.hpp>
#include <bit>
#include <charconv>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <random>
#include <map>
#include <unistd.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

namespace radar {
namespace {
volatile std::sig_atomic_t interrupted=0;
void interrupt(int){interrupted=1;}
double number(std::string_view value){double n{};auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),n);
    if(error!=std::errc{} || end!=value.data()+value.size() || !std::isfinite(n))throw std::invalid_argument("Invalid number");return n;}
struct Options {
    std::filesystem::path metadata,output,simulation;
    double seconds=0,gap_seconds=0,cpi=3,update=.25,doppler=100,range=100000;
    std::size_t gap_samples=0;
    bool unpaced=false,tui=true,remove_dc=true,profile=false,simulate_once=false,simulate_cyclic=false;
    std::string archive="profiles";
};
struct ChannelStatus {std::uint64_t rows=0,good=0,bad=0,corrected=0,maps=0,valid_rows=0;double fit_error=0,map_time=0;Tps mode;};
class SignalScope {
    using Handler=void(*)(int);
    Handler previous_int_,previous_term_;
public:
    SignalScope():previous_int_(std::signal(SIGINT,interrupt)),previous_term_(std::signal(SIGTERM,interrupt)){interrupted=0;}
    ~SignalScope(){std::signal(SIGINT,previous_int_);std::signal(SIGTERM,previous_term_);}
};
class TextMonitor final:public CaptureMonitor {
    std::chrono::steady_clock::time_point last_{};
    CapturePhase phase_=CapturePhase::failed;
public:
    bool stop_requested() const override{return interrupted!=0;}
    void publish(const CaptureStatus& s) override {
        const auto now=std::chrono::steady_clock::now();
        if(s.phase==phase_ && now-last_<std::chrono::seconds(5))return;
        last_=now;phase_=s.phase;
        std::cerr<<capture_phase_name(s.phase)<<" | "<<s.message<<'\n';
        if(!s.applied)return;
        std::cerr<<std::fixed<<std::setprecision(2)<<"RX "<<s.settings.rate_hz/1e6<<" MS/s  received "<<s.received
            <<"  missing "<<s.missing<<"  discarded "<<s.host_discarded<<"  unknown "<<s.unknown_gaps
            <<"  replay loops "<<s.replay_loops<<"  DSP "<<s.headroom<<"x\n";
        for(const auto& c:s.channels)std::cerr<<c.id<<" "<<(c.receiver.ready ? "TRACK" : c.receiver.locked ? "ANCHOR" : "SEARCH")
            <<" CFO "<<c.receiver.frequency_hz<<" Hz  drift "<<c.receiver.residual_rate_ppm<<" ppm  timing RMS "
            <<c.receiver.timing_rms_samples*125<<" ns  phase RMS "<<c.receiver.phase_rms_degrees
            <<" deg  RS "<<c.rs_good<<"/"<<c.rs_bad<<"  rows "<<c.rows<<'\n';
    }
};
boost::json::object capture_stats(const CaptureCounters& s){
    return {{"received_samples",s.samples.load()},{"overflows",s.overflows.load()},{"sequence_errors",s.sequence_errors.load()},
        {"timeouts",s.timeouts.load()},{"timestamp_gaps",s.timestamp_gaps.load()},{"missing_samples",s.missing_samples.load()},
        {"host_discarded_samples",s.host_discarded.load()},{"unquantified_gaps",s.unknown_gaps.load()},{"segments",s.segments.load()},{"replay_loops",s.replay_loops.load()}};
}
void warm_receiver(GpuPipeline& pipeline,const GpuPipelineConfig& config,std::shared_ptr<gpu::Context> context,const GpuPipeline::Consume& consume,const CaptureMonitor& monitor){
    // Exercise acquisition, decoding and all correlation/focus shapes before
    // receiving. Synthetic observations are reset and never enter the archive.
    constexpr unsigned symbols=4080,fft_size=8192;
    Tps mode;mode.mode=1;mode.constellation=2;mode.rate=2;
    std::mt19937 random(158932);std::vector<std::uint8_t> bytes(72'000*188);
    for(auto& byte:bytes)byte=std::uint8_t(random());for(unsigned p=0;p<72'000;++p)bytes[p*188]=p%8 ? 0x47 : 0xb8;
    std::map<unsigned,std::vector<C>> waveforms;std::size_t maximum=0;Fft fft(fft_size);std::vector<C> work(fft_size);
    for(auto guard:config.guards){if(waveforms.contains(guard))continue;
        Modem modem{int(guard)};mode.guard=guard==32 ? 0 : guard==16 ? 1 : guard==8 ? 2 : 3;
        const auto carriers=modem.synthesize(bytes,mode,symbols);const unsigned prefix=fft_size/guard,length=fft_size+prefix;
        auto& waveform=waveforms[guard];waveform.resize(symbols*length);maximum=std::max(maximum,waveform.size());
        for(unsigned s=0;s<symbols;++s){std::fill(work.begin(),work.end(),C{});
            for(unsigned k=0;k<6817;++k)work[(k+fft_size-3408)%fft_size]=carriers[s*6817+k];
            fft.transform(work,true);std::copy(work.end()-prefix,work.end(),waveform.begin()+s*length);std::copy(work.begin(),work.end(),waveform.begin()+s*length+prefix);}}
    auto storage=gpu::make_buffer(context->gpu,config.guards.size()*config.max_samples*sizeof(C));
    auto* mapped=static_cast<C*>(context->gpu.device->mapMemory(*storage.memory,0,config.guards.size()*config.max_samples*sizeof(C)));
    const gpu::BufferView input{context,*storage.handle,config.guards.size()*config.max_samples*sizeof(C),mapped};
    for(std::size_t start=0;start<maximum;start+=config.max_samples){
        if(monitor.stop_requested())break;
        const auto count=std::min(config.max_samples,maximum-start);
        for(std::size_t c=0;c<config.guards.size();++c){const auto& waveform=waveforms.at(config.guards[c]);auto* output=mapped+c*config.max_samples;
            const auto copied=start<waveform.size() ? std::min(count,waveform.size()-start) : 0;
            if(copied)std::copy_n(waveform.data()+start,copied,output);std::fill(output+copied,output+count,C{});}
        pipeline.process({input,start,count,config.guards.size(),config.max_samples},consume);
    }
    pipeline.finish(consume);pipeline.reset();context->gpu.device->unmapMemory(*storage.memory);
}
}
namespace {
int run_capture(const Options& options,const RecordingMetadata& description,const LiveDisplayFactory& display_factory,CaptureMonitor& monitor){
#ifdef __APPLE__
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED,0);
#endif
        const bool save_profiles=options.archive=="profiles" || options.archive=="both";
        const bool save_channels=options.archive=="channels" || options.archive=="both";
        CaptureStatus view;view.simulation=!options.simulation.empty();view.looping=view.simulation && !options.simulate_once;
        view.source=view.simulation ? options.simulation.string() : description.capture->device_args;
        view.output=options.output.string();view.metadata=options.metadata.string();view.archive=options.archive;
        view.profile=options.profile;view.remove_dc=options.remove_dc;view.integration_seconds=options.cpi;view.doppler_hz=options.doppler;view.update_seconds=options.update;view.range_metres=options.range;
        if(display_factory)view.display="Native ImPlot; press v for selected channel";
        for(const auto& signal:description.signals){CaptureChannelStatus channel;channel.id=signal.id;channel.name=signal.name;
            channel.rf_hz=signal.channel.rf_hz;channel.nco_hz=signal.channel.nco_hz;view.channels.push_back(std::move(channel));}
        view.message="Configuring capture source";monitor.publish(view);
        auto source=options.simulation.empty() ? make_uhd_source(*description.capture) :
            make_replay_source(options.simulation,*description.capture,!options.unpaced,options.gap_seconds,options.gap_samples,!options.simulate_once,options.simulate_cyclic);
        const auto applied=source->applied();view.settings=applied;view.applied=true;view.phase=CapturePhase::warming;
        view.message="Preparing Vulkan channeliser and receiver";monitor.publish(view);SarCapture capture;description.configure(capture);
        capture.source_rate=std::uint32_t(std::llround(applied.rate_hz));
        capture.input_rate=(capture.source_rate+7'999'999)/8'000'000*8'000'000;capture.decimation=capture.input_rate/8'000'000;
        capture.recording_rf_hz=applied.centre_hz;
        std::vector<double> centres;GpuPipelineConfig config;
        for(auto& channel:capture.channels){channel.nco_hz+=description.recording_rf_hz-applied.centre_hz;centres.push_back(channel.nco_hz);config.guards.push_back(channel.guard);}
        for(std::size_t c=0;c<centres.size();++c)view.channels[c].nco_hz=centres[c];
        view.phase_budget=capture.clock_phase_degrees;
        capture.validate();auto clock=std::make_shared<SampleClock>(capture.input_rate,centres,capture.clock_phase_degrees,description.capture->clock);
        // Integer source coordinates label samples; this tiny initial slope
        // retains UHD's fractional applied rate without independently retiming channels.
        clock->schedule(64,applied.rate_hz/capture.source_rate);
        GpuChannelBank bank(capture.source_rate,centres);bank.set_clock(clock);
        config.max_samples=bank.capacity();config.decimation=bank.decimation;config.frames_per_batch=6;config.residual_spectra=false;
        GpuPipeline pipeline(config,bank.context(),clock);
        const auto ring_size=std::bit_ceil(std::max<std::size_t>(std::size_t(applied.rate_hz*description.capture->buffer_seconds),bank.input_capacity()*2));
        GpuLiveInput gpu_input(bank.context(),ring_size,bank.input_capacity());CaptureRing ring(gpu_input.storage());
        if(!options.output.parent_path().empty())std::filesystem::create_directories(options.output.parent_path());
        if(!std::filesystem::create_directory(options.output))throw std::runtime_error("Output directory already exists (nothing was overwritten)");
        description.write(options.output/"metadata.json");
        const auto& model=description.capture->clock;
        CaptureDiagnostics diagnostics(options.output/"diagnostics.jsonl",{
            {"metadata",boost::json::parse(description.json())},{"archive",options.archive},{"range_metres",options.range},
            {"integration_seconds",options.cpi},{"update_seconds",options.update},{"doppler_hz",options.doppler},
            {"display_remove_dc",options.remove_dc},{"profile_stationary_subtraction",false},{"simulation",view.simulation},
            {"simulation_cyclic",options.simulate_cyclic},{"simulation_once",options.simulate_once},{"unpaced",options.unpaced},
            {"requested_seconds",options.seconds},{"gpu_timestamps_enabled",options.profile},
            {"applied_capture",boost::json::object{{"device",applied.device},{"serial",applied.serial},{"antenna",applied.antenna},
                {"rate_hz",applied.rate_hz},{"centre_hz",applied.centre_hz},{"gain_db",applied.gain_db},
                {"bandwidth_hz",applied.bandwidth_hz},{"master_clock_hz",applied.master_clock_hz},
                {"clock_source",applied.clock_source},{"time_source",applied.time_source}}},
            {"actual_rate_hz",applied.rate_hz},{"actual_centre_hz",applied.centre_hz},{"buffer_seconds",ring.capacity()/applied.rate_hz},
            {"corrected_adc_rate_hz",capture.input_rate},{"source_coordinate_rate_hz",capture.source_rate},{"phase_budget_degrees",capture.clock_phase_degrees},
            {"clock_model",boost::json::object{{"receiver_white_fm",model.receiver.white_fm},{"receiver_random_walk_fm",model.receiver.random_walk_fm},
                {"receiver_drift_diffusion",model.receiver.drift_diffusion},{"transmitter_white_fm",model.transmitter.white_fm},
                {"transmitter_random_walk_fm",model.transmitter.random_walk_fm},{"settling_seconds",model.settling_seconds},
                {"innovation_huber_sigma",model.innovation_huber_sigma},{"unknown_sigma_samples",model.unknown_sigma_samples},{"physical_noise_parameters_calibrated",false}}}});
        std::unique_ptr<LiveArchive> archive;
        if(options.archive!="none")archive=std::make_unique<LiveArchive>(options.output/"capture.dvblive",description,applied,capture,options.archive);
        LiveDisplay display;if(display_factory)display=display_factory(bank.context(),description);
        DetectionOptions detection;detection.integration_seconds=options.cpi;detection.update_seconds=options.update;
        detection.focus.doppler_hz=options.doppler;detection.focus.remove_dc=options.remove_dc;
        detection.focus.range_metres=options.range;
        detection.publish_maps=bool(display.project);
        detection.best_effort_maps=true;
        detection.stationary_prediction=false;
        bool running=false;std::unique_ptr<MatchedDetector> detector;
        auto project=[&](auto& map,const auto& values){if(running && display.project)display.project(map,values);};
        if(save_profiles || display.publish)detector=std::make_unique<MatchedDetector>(bank.context(),centres.size(),detection,
            display.project ? MatchedDetector::Projector{project} : MatchedDetector::Projector{},
            save_profiles ? MatchedDetector::ProfileConsumer{[&](const auto& batch,const auto& profiles,auto stationary,auto variance){if(running)archive->profiles(batch,profiles,stationary,variance);}} : MatchedDetector::ProfileConsumer{});
        CaptureCounters counters;std::vector<ChannelStatus> channels(centres.size());
        std::atomic<bool> stopping=false,producer_done=false;
        std::exception_ptr producer_failure;std::mutex failure_mutex;
        std::int64_t first_hardware_tick=0;bool have_hardware_tick=false;
        auto publish=[&](std::size_t channel,DopplerMap&& map){++channels.at(channel).maps;channels[channel].map_time=map.time_end;
            if(display.publish)display.publish(channel,std::move(map));};
        GpuPipelineTimings processing;ComputeTimings unpacking,channelizing;DetectionTimings detecting;
        for(auto* timing:{&processing.receiver,&processing.frames,&processing.reconstruction,&processing.response,
            &unpacking,&channelizing,&detecting.noise,&detecting.profiles,&detecting.focus})timing->timestamp_queries=options.profile;
        auto consume=[&](const SpectralBatch& spectra){
            auto& status=channels.at(spectra.frame.channel);status.rows+=spectra.frame.last-spectra.frame.first;
            status.fit_error=spectra.estimate.relative_error;
            status.mode=spectra.frame.frames.back();const auto* flags=static_cast<const std::uint32_t*>(spectra.transmitted.valid.mapped);
            for(auto row=spectra.frame.first;row<spectra.frame.last;++row)
                status.valid_rows+=!std::ranges::any_of(std::span{flags+row*6817,std::size_t(6817)},[](auto valid){return !valid;});
            for(const auto& packet:spectra.packets.packets){if(packet.valid){++status.good;status.corrected+=packet.corrected;}else ++status.bad;}
            if(!running && !detector)return;
            const SarBatch scientific(capture,spectra,clock->segments());
            if(running)diagnostics.spectra(scientific);
            if(detector){
                if(running && display.active){
                    const bool requested=display.active(spectra.frame.channel);
                    // Optional map jobs yield while more than a quarter of
                    // the receive buffer is outstanding. Never alter profiles
                    // or the coherent aperture to make a display refresh fit.
                    const bool admitted=ring.head()-ring.tail()<ring.capacity()/4;
                    if(requested && !admitted)++view.display_pressure_skips;
                    detector->set_publish_maps(spectra.frame.channel,requested && admitted);
                }
                detector->append(scientific,publish,&detecting);
            }
        };
        // Complete the first channelizer dispatch before starting the source.
        // UHD must never wait for shader/FFT initialization in its receive loop.
        // Materialise the whole receive ring before UHD starts, including the
        // pages beyond the first FFT tile. Avoid first-touch faults on recv().
        std::ranges::fill(gpu_input.storage(),Iq16{});
        const auto warm_first=std::uint64_t(16384);
        const auto warm_support=bank.input_support(warm_first,bank.capacity());
        const auto warm=gpu_input.gather(0,std::size_t(warm_support.last-warm_support.first));
        bank.process_clocked_device(warm,std::size_t(warm_support.last-warm_support.first),std::uint64_t(warm_support.first),warm_first,bank.capacity());
        view.message="Warming OFDM, FEC, reconstruction and detector";monitor.publish(view);
        warm_receiver(pipeline,config,bank.context(),[&](const auto& spectra){
            if(detector)detector->append(SarBatch(capture,spectra,clock->segments()),[](std::size_t,DopplerMap&&){});
        },monitor);
        if(detector)detector->reset();running=true;clock->enable_diagnostics();
        if(const auto anchor=source->time_anchor()){diagnostics.time_anchor(*anchor);if(archive)archive->time_anchor(*anchor);}
        view.phase=monitor.stop_requested() ? CapturePhase::draining : CapturePhase::running;view.message="Receiving and processing";monitor.publish(view);
        const auto started=std::chrono::steady_clock::now();
        std::jthread producer([&]{
#ifdef __APPLE__
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
            try {
                CaptureTimeline timeline;std::vector<Iq16> discard(262144);if(!monitor.stop_requested())source->start();
                while(!stopping && !monitor.stop_requested()){
                    if(options.seconds>0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>=options.seconds)break;
                    auto target=ring.writable(262144);
                    if(target.empty() && options.unpaced){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
                    const bool discard_block=target.empty();if(discard_block)target=discard;
                    auto result=source->receive(target);
                    if(result.end)break;
                    if(result.error==CaptureRead::Error::fatal)throw std::runtime_error("UHD receive: "+result.message);
                    if(result.error==CaptureRead::Error::timeout)++counters.timeouts;
                    if(result.count){
                        if(result.restart)timeline.restart();if(result.looped)++counters.replay_loops;
                        if(result.tick && !have_hardware_tick){first_hardware_tick=*result.tick;have_hardware_tick=true;}
                        const auto position=timeline.receive(result.tick,result.count,counters,!discard_block);
                        if(discard_block || !ring.commit(result.count,position.first,position.segment))timeline.discard(result.count,counters);
                    }
                    if(result.error==CaptureRead::Error::overflow || result.sequence)timeline.error(result.error==CaptureRead::Error::overflow,result.sequence,counters);
                }
                source->stop();
            }catch(...){std::lock_guard lock(failure_mutex);producer_failure=std::current_exception();try{source->stop();}catch(...){}}
            producer_done=true;ring.close();
        });
        // Always stop/join capture before its ring, archive or GPU owners unwind.
        struct Stop {std::atomic<bool>& stop;CaptureRing& ring;std::jthread& producer;~Stop(){stop=true;ring.close();if(producer.joinable())producer.join();}} stop{stopping,ring,producer};
        auto last_status=started;
        double dsp_wall=0,processed_seconds=0,status_dsp=0,status_signal=0;
        std::uint64_t first_raw=0,ring_begin=0,raw_end=0,next_center=0,segment=UINT64_MAX,rf_end=0;
        auto first_center=[&](std::uint64_t raw){
            const auto nominal=nominal_corrected_sample(raw,capture.source_rate,capture.input_rate);
            std::uint64_t lo=std::max<std::uint64_t>(16384/capture.decimation,(nominal-nominal/500)/capture.decimation),
                hi=std::max<std::uint64_t>(lo+1,(nominal+nominal/500+bank.input_capacity()*2)/capture.decimation);
            while(lo<hi){const auto mid=lo+(hi-lo)/2;if(bank.input_support(mid*capture.decimation,1).first<std::int64_t(raw))lo=mid+1;else hi=mid;}
            return lo*capture.decimation;
        };
        auto run_available=[&](bool final){
            for(;;){
                bank.update_clock(next_center);
                std::size_t lo=0,hi=bank.capacity();
                while(lo<hi){const auto mid=lo+(hi-lo+1)/2;const auto support=bank.input_support(next_center,mid);
                    if(support.first>=std::int64_t(first_raw) && support.last<=std::int64_t(raw_end))lo=mid;else hi=mid-1;}
                if(!lo || (!final && lo<bank.capacity()))break;
                const auto support=bank.input_support(next_center,lo);
                const auto mark=std::chrono::steady_clock::now();const auto count=std::size_t(support.last-support.first);
                const auto skip=std::min<std::uint64_t>(count,rf_end>std::uint64_t(support.first) ? rf_end-std::uint64_t(support.first) : 0);
                const auto input=gpu_input.gather(ring_begin+std::uint64_t(support.first)-first_raw,count,&unpacking,skip);
                view.rf.add(gpu_input.health());rf_end=std::uint64_t(support.last);
                const auto batch=bank.process_clocked_device(input,count,std::uint64_t(support.first),next_center,lo,&channelizing);
                // Channelisation has finished reading the ring. Only the next
                // centred-filter halo remains live; decoder/display/archive
                // backpressure must not retain already consumed ADC samples.
                const auto next=next_center+lo*capture.decimation;
                const auto keep=std::max<std::int64_t>(first_raw,bank.input_support(next,1).first);
                ring.release(ring_begin+std::min<std::uint64_t>(std::uint64_t(keep)-first_raw,raw_end-first_raw));
                if(save_channels)archive->channels(batch,clock->segments());
                pipeline.process(batch,consume,&processing);
                if(display.observe_time)display.observe_time(double(next_center+lo*capture.decimation)/capture.input_rate);
                const double block_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
                view.maximum_block_ms=std::max(view.maximum_block_ms,1000*block_seconds);
                dsp_wall+=block_seconds;processed_seconds+=lo/8e6;
                next_center=next;
            }
        };
        auto show=[&](bool force){
            const auto now=std::chrono::steady_clock::now();if(!force && now-last_status<std::chrono::milliseconds(250))return;last_status=now;
            const auto status=pipeline.status();const auto observers=clock->status(double(next_center));
            view.elapsed=std::chrono::duration<double>(now-started).count();view.processed_seconds=processed_seconds;
            view.headroom=processed_seconds/std::max(dsp_wall,1e-9);
            if(dsp_wall>status_dsp && processed_seconds>status_signal){
                view.recent_headroom=(processed_seconds-status_signal)/(dsp_wall-status_dsp);
                if(processed_seconds>=5)view.minimum_recent_headroom=view.minimum_recent_headroom>0 ?
                    std::min(view.minimum_recent_headroom,view.recent_headroom) : view.recent_headroom;
            }
            status_dsp=dsp_wall;status_signal=processed_seconds;
            view.received=counters.samples;view.missing=counters.missing_samples;view.host_discarded=counters.host_discarded;
            view.unknown_gaps=counters.unknown_gaps;view.overflows=counters.overflows;view.sequence_errors=counters.sequence_errors;
            view.timeouts=counters.timeouts;view.timestamp_gaps=counters.timestamp_gaps;view.segments=counters.segments;view.replay_loops=counters.replay_loops;
            view.queue_ms=(ring.head()-ring.tail())/applied.rate_hz*1000;view.capacity_ms=ring.capacity()/applied.rate_hz*1000;
            view.peak_queue_ms=ring.peak()/applied.rate_hz*1000;
            view.clock_ppm=(clock->at(double(next_center)).rate-1)*1e6;view.clock_segments=clock->segments().size();view.phase_budget=clock->phase_budget_degrees();
            const auto clock_state=clock->estimate(double(next_center));
            view.clock_sigma_ns=std::sqrt(clock_state.offset_variance)/capture.input_rate*1e9;
            view.clock_rate_sigma_ppm=std::sqrt(clock_state.velocity_variance)/capture.input_rate*1e6;
            if(archive){view.archive_bytes=archive->bytes();view.archive_queued_bytes=archive->queued_bytes();}
            for(std::size_t c=0;c<channels.size();++c){auto& v=view.channels[c];const auto& f=channels[c];
                v.receiver=status[c];v.clock=observers[c];v.clock_fit_ns=observers[c].fit_rms_samples/capture.input_rate*1e9;
                v.rows=f.rows;v.valid_rows=f.valid_rows;v.rs_good=f.good;v.rs_bad=f.bad;v.corrected_bytes=f.corrected;
                v.maps=f.maps;v.map_time=f.map_time;v.response_error=f.fit_error;
                static constexpr std::array rates{"1/2","2/3","3/4","5/6","7/8"};
                if(f.rows){v.qam=1U<<(2*(f.mode.constellation+1));v.code_rate=rates[std::clamp(f.mode.rate,0,4)];}
            }
            const double gpu=unpacking.gpu+channelizing.gpu+processing.receiver.gpu+processing.frames.gpu+processing.reconstruction.gpu+processing.response.gpu+detecting.noise.gpu+detecting.profiles.gpu+detecting.focus.gpu;
            view.gpu_ms_per_second=gpu*1000/std::max(processed_seconds,1e-9);
            view.fec_ms_per_second=(processing.fec.demap+processing.fec.viterbi+processing.fec.reed_solomon)*1000/std::max(processed_seconds,1e-9);
            diagnostics.observations(clock->take_observations());diagnostics.status(view);monitor.publish(view);
        };
        CaptureChunk chunk;
        for(;;){
            if(!ring.next(chunk,std::chrono::milliseconds(250))){show(false);if(producer_done)break;continue;}
            if(segment==UINT64_MAX && archive)archive->epoch(first_hardware_tick,applied.rate_hz);
            if(segment!=chunk.segment){
                if(segment!=UINT64_MAX){
                    run_available(true);ring.release(ring_begin+raw_end-first_raw);
                    const auto draining=std::chrono::steady_clock::now();pipeline.drain(consume,&processing);
                    dsp_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-draining).count();
                }
                segment=chunk.segment;first_raw=chunk.source_first;ring_begin=chunk.ring_first;raw_end=chunk.source_first;
                next_center=first_center(first_raw);if(archive)archive->gap(chunk);diagnostics.gap(chunk);
            }
            if(chunk.source_first!=raw_end || chunk.ring_first!=ring_begin+raw_end-first_raw)throw std::logic_error("Non-contiguous capture segment");
            raw_end+=chunk.count;run_available(false);show(false);
        }
        view.phase=CapturePhase::draining;view.message="Draining decoder, detector and archive";show(true);
        if(segment!=UINT64_MAX)run_available(true);
        const auto draining=std::chrono::steady_clock::now();
        pipeline.finish(consume,&processing);if(detector)detector->finish(publish,&detecting);
        dsp_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-draining).count();
        stopping=true;if(producer.joinable())producer.join();
        if(const auto anchor=source->time_anchor()){diagnostics.time_anchor(*anchor);if(archive)archive->time_anchor(*anchor);}
        const auto archive_drain=std::chrono::steady_clock::now();if(archive)archive->finish();
        dsp_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-archive_drain).count();
        auto summary=capture_stats(counters);summary["simulation"]=!options.simulation.empty();summary["simulation_loop"]=view.looping;summary["first_hardware_tick"]=first_hardware_tick;
        {std::lock_guard lock(failure_mutex);if(producer_failure){try{std::rethrow_exception(producer_failure);}catch(const std::exception& e){summary["error"]=e.what();}}}
        summary["actual_rate_hz"]=applied.rate_hz;summary["actual_centre_hz"]=applied.centre_hz;
        summary["device"]=applied.device;summary["serial"]=applied.serial;summary["antenna"]=applied.antenna;
        summary["actual_gain_db"]=applied.gain_db;summary["actual_bandwidth_hz"]=applied.bandwidth_hz;
        summary["actual_master_clock_hz"]=applied.master_clock_hz;
        summary["clock_source"]=applied.clock_source;summary["time_source"]=applied.time_source;
        summary["dsp_seconds"]=dsp_wall;summary["processed_seconds"]=processed_seconds;summary["processing_speed"]=processed_seconds/std::max(dsp_wall,1e-9);
        summary["receive_buffer_seconds"]=ring.capacity()/applied.rate_hz;
        summary["receive_queue_peak_seconds"]=ring.peak()/applied.rate_hz;
        summary["minimum_recent_headroom"]=view.minimum_recent_headroom;
        summary["maximum_processing_block_ms"]=view.maximum_block_ms;
        summary["rf_samples"]=view.rf.samples;summary["rf_power_sum"]=view.rf.power_sum;
        summary["rf_peak_component"]=view.rf.peak_component;summary["rf_rail_samples"]=view.rf.rail_samples;
        summary["rf_near_rail_samples"]=view.rf.near_rail_samples;
        summary["profile_stationary_subtraction"]=false;summary["display_remove_dc"]=options.remove_dc;
        summary["display_pressure_skips"]=view.display_pressure_skips;
        summary["fec_workers"]=std::min<unsigned>(centres.size(),std::max(1U,std::thread::hardware_concurrency()));
        summary["fec_worker_seconds"]=processing.fec.demap+processing.fec.viterbi+processing.fec.reed_solomon;
        summary["viterbi_worker_seconds"]=processing.fec.viterbi;
        summary["rs_worker_seconds"]=processing.fec.reed_solomon;
        summary["gpu_timestamps_enabled"]=options.profile;
        summary["range_metres"]=options.range;
        summary["channelizer_gpu_seconds"]=options.profile ? boost::json::value(channelizing.gpu) : boost::json::value();
        summary["unpack_gpu_seconds"]=options.profile ? boost::json::value(unpacking.gpu) : boost::json::value();
        summary["receiver_gpu_seconds"]=options.profile ? boost::json::value(processing.receiver.gpu) : boost::json::value();
        summary["frames_gpu_seconds"]=options.profile ? boost::json::value(processing.frames.gpu) : boost::json::value();
        summary["reconstruction_gpu_seconds"]=options.profile ? boost::json::value(processing.reconstruction.gpu) : boost::json::value();
        summary["response_gpu_seconds"]=options.profile ? boost::json::value(processing.response.gpu) : boost::json::value();
        summary["noise_gpu_seconds"]=options.profile ? boost::json::value(detecting.noise.gpu) : boost::json::value();
        summary["profiles_gpu_seconds"]=options.profile ? boost::json::value(detecting.profiles.gpu) : boost::json::value();
        summary["focus_gpu_seconds"]=options.profile ? boost::json::value(detecting.focus.gpu) : boost::json::value();
        boost::json::array stats;const auto status=pipeline.status();
        for(std::size_t c=0;c<channels.size();++c)stats.emplace_back(boost::json::object{{"id",description.signals[c].id},{"rows",channels[c].rows},
            {"rs_good",channels[c].good},{"rs_bad",channels[c].bad},{"corrected_bytes",channels[c].corrected},{"maps",channels[c].maps},
            {"resets",status[c].reacquisitions},{"locked",status[c].locked},{"cfo_hz",status[c].frequency_hz},
            {"residual_rate_ppm",status[c].residual_rate_ppm},{"timing_ns",status[c].timing_samples*125},{"timing_rms_ns",status[c].timing_rms_samples*125}});
        summary["channels"]=std::move(stats);std::ofstream file(options.output/"summary.json");file.exceptions(std::ios::failbit|std::ios::badbit);file<<boost::json::serialize(summary)<<'\n';
        {std::lock_guard lock(failure_mutex);if(producer_failure)std::rethrow_exception(producer_failure);}
        view.phase=CapturePhase::stopped;view.message="Capture complete; archive and summary closed";show(true);diagnostics.finish();
        return 0;
}
}
int capture_cli(int argc,char** argv,LiveDisplayFactory display_factory,CaptureUiRunner ui,CaptureDisplayMode display_mode){
    try {
        Options options;
        for(int i=1;i<argc;++i){
            const std::string_view key=argv[i];
            if(key=="--help"){
                std::cout<<"Usage: dvbt_capture --metadata LIVE.json --output NEW_DIRECTORY\n"
                    <<"  [--seconds 0] [--archive profiles|channels|both|none] [--no-tui] [--profile]\n"
                    <<"  [--integration 3] [--update 0.25] [--doppler 100] [--range-metres 100000]\n"
                    <<"  [--remove-dc true|false]\n"
                    <<"Simulation loops until stopped: --simulate INPUT_ci16.wav [--unpaced] [--simulate-once]\n"
                    <<"  --simulate-cyclic preserves coherence across a verified cyclic fixture's seams.\n"
                    <<"  [--inject-gap-at SECONDS --inject-gap-samples COUNT]\n"
                    <<"TUI: 1-9 select channel, v opens native ImPlot, t shows timing, ? shows help.\n"
                    <<"No raw ADC I/Q is saved. Ctrl-C drains processing and closes the archive.\n";return 0;
            }
            if(key=="--simulate-once"){options.simulate_once=true;continue;}
            if(key=="--simulate-cyclic"){options.simulate_cyclic=true;continue;}
            if(key=="--no-tui"){options.tui=false;continue;}
            if(key=="--unpaced"){options.unpaced=true;continue;}
            if(key=="--profile"){options.profile=true;continue;}
            if(i+1==argc)throw std::invalid_argument("Missing option value");const std::string_view value=argv[++i];
            if(key=="--metadata")options.metadata=value;
            else if(key=="--output")options.output=value;
            else if(key=="--archive")options.archive=value;
            else if(key=="--simulate")options.simulation=value;
            else if(key=="--seconds")options.seconds=number(value);
            else if(key=="--integration")options.cpi=number(value);
            else if(key=="--update")options.update=number(value);
            else if(key=="--doppler")options.doppler=number(value);
            else if(key=="--range-metres")options.range=number(value);
            else if(key=="--inject-gap-at")options.gap_seconds=number(value);
            else if(key=="--inject-gap-samples"){const auto n=number(value);if(n<0 || n>1e9 || std::floor(n)!=n)throw std::invalid_argument("Invalid injected sample count");options.gap_samples=std::size_t(n);}
            else if(key=="--remove-dc"){if(value!="true" && value!="false")throw std::invalid_argument("DC option requires true or false");options.remove_dc=value=="true";}
            else throw std::invalid_argument("Unknown option: "+std::string(key));
        }
        if(options.metadata.empty() || options.output.empty() || options.seconds<0 || options.gap_seconds<0 || options.range<1000 || options.range>matched_max_range_metres ||
           (options.archive!="profiles" && options.archive!="channels" && options.archive!="both" && options.archive!="none"))throw std::invalid_argument("Metadata, new output directory and valid capture options are required");
        if(options.simulation.empty() && (options.unpaced || options.gap_samples || options.simulate_once || options.simulate_cyclic))throw std::invalid_argument("Simulation controls require --simulate");
        const auto description=RecordingMetadata::read(options.metadata);
        if(!description.capture)throw std::invalid_argument("Live metadata requires a capture object");
        SignalScope signals;
        const bool interactive=options.tui && isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) && bool(ui);
        const bool display_enabled=interactive || display_mode==CaptureDisplayMode::headless_projector;
        const CaptureWork work=[&](CaptureMonitor& monitor){return run_capture(options,description,display_enabled ? display_factory : LiveDisplayFactory{},monitor);};
        if(interactive)return ui(work);
        TextMonitor monitor;return work(monitor);
    }catch(const std::exception& e){std::cerr<<"Capture error: "<<e.what()<<'\n';return 1;}
}
}
