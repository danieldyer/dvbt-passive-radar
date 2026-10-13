#include "gpu_pipeline.hpp"
#include "gpu_commands.hpp"
#include "receiver_clock.hpp"
#include "frame_decoder.hpp"
#include <cmath>
#include <thread>

namespace radar {
namespace {
void add(ComputeTimings& out,const ComputeTimings& now,const ComputeTimings& previous) {
    out.gpu+=now.gpu-previous.gpu;out.batches+=now.batches-previous.batches;
    out.gpu_timestamps_available|=now.gpu_timestamps_available;
    for(unsigned i=0;i<5;++i)out.stages[i]+=now.stages[i]-previous.stages[i];
}
}
struct GpuPipeline::Impl {
    struct Origin {unsigned segment=UINT32_MAX;double value=0;bool ready=false;};
    GpuPipelineConfig config;
    std::shared_ptr<gpu::Context> context;
    std::shared_ptr<SampleClock> clock;
    Modem tables;
    // Declared before every GPU owner: deferred query pointers remain valid
    // through normal completion and destruction during exception unwinding.
    GpuPipelineTimings timing,reported;
    GpuReceiver receiver;
    GpuFrames frames;
    GpuReconstruction reconstruction;
    GpuResponse response;
    std::vector<Origin> origins;
    std::unique_ptr<FrameDecoder> decoder;
    Impl(const GpuPipelineConfig& c,std::shared_ptr<gpu::Context> ctx,std::shared_ptr<SampleClock> shared_clock):config(c),context(std::move(ctx)),clock(std::move(shared_clock)),
        receiver(tables.pilot_pattern(),c.guards,c.max_samples,context),
        frames(c.guards,context,c.frames_per_batch),reconstruction(context),response(context),origins(c.guards.size()) {
        if(c.response_order>32 || !std::isfinite(c.reference_delay))throw std::invalid_argument("Invalid GPU response configuration");
        const unsigned count=c.fec_workers ? c.fec_workers : std::min<unsigned>(c.guards.size(),std::max(1U,std::thread::hardware_concurrency()));
        const unsigned slots=c.frame_slots ? c.frame_slots : count+2;
        decoder=std::make_unique<FrameDecoder>(count,slots,tables,context);
    }
    void render(const DecodedFrame& slot,const Consume& consume,bool measured) {
        const auto& frame=slot.frame;
        const auto decoded=slot.decoded();
        if(measured){timing.fec.demap+=slot.timing.demap;timing.fec.viterbi+=slot.timing.viterbi;timing.fec.reed_solomon+=slot.timing.reed_solomon;}
        gpu::Commands commands(context);
        const auto reference=reconstruction.reconstruct(decoded,frame.frames,tables,measured ? &timing.reconstruction : nullptr,&commands);
        auto& origin=origins[frame.channel];
        if(origin.segment!=frame.segment)origin={frame.segment,fft_window_margin+config.reference_delay,!config.automatic_origin};
        std::vector<double> offsets;offsets.reserve(frame.symbols.size());
        for(const auto& symbol:frame.symbols)offsets.push_back(symbol.source_offset_samples);
        auto fit=[&]{return response.fit(frame.carriers.measured,reference,origin.value,config.response_order,16,
                                        measured ? &timing.response : nullptr,&commands,offsets);};
        auto estimate=fit();
        if(!origin.ready) {
            commands.wait();GpuResponse::read_diagnostics(estimate);
            origin.value+=GpuResponse::direct_origin_adjustment(estimate);origin.ready=true;estimate=fit();
        }
        const auto residual=response.apply(frame.carriers.measured,reference,estimate,measured ? &timing.response : nullptr,&commands,config.residual_spectra);
        commands.wait();GpuResponse::read_diagnostics(estimate);
        receiver.pilot_variance(frame.channel,frame.symbols.back().window.sample,
            {static_cast<const float*>(estimate.innovation_variance.mapped),6817});
        consume({frame,decoded,reference,estimate,residual});
    }
    void collect_one(const Consume& consume,bool measured) {
        render(decoder->front(),consume,measured);decoder->pop();
    }
    void collect(const Consume& consume,bool all,bool measured) {
        while(!decoder->empty() && (all || decoder->ready()))collect_one(consume,measured);
    }
    void capture(const FrameSelection& selection,gpu::Commands& commands,const Consume& consume,bool measured) {
        if(decoder->full())collect_one(consume,measured);
        decoder->submit(selection,commands,measured ? &timing.frames : nullptr);
    }
    void report(GpuPipelineTimings* output) {
        if(output) {
            add(output->receiver,timing.receiver,reported.receiver);add(output->frames,timing.frames,reported.frames);
            add(output->reconstruction,timing.reconstruction,reported.reconstruction);add(output->response,timing.response,reported.response);
            output->fec.demap+=timing.fec.demap-reported.fec.demap;
            output->fec.viterbi+=timing.fec.viterbi-reported.fec.viterbi;
            output->fec.reed_solomon+=timing.fec.reed_solomon-reported.fec.reed_solomon;
        }
        reported=timing;
    }
    void detail(const GpuPipelineTimings* output) {
        if(!output)return;
        timing.receiver.detailed=output->receiver.detailed;timing.frames.detailed=output->frames.detailed;
        timing.reconstruction.detailed=output->reconstruction.detailed;timing.response.detailed=output->response.detailed;
        timing.receiver.timestamp_queries=output->receiver.timestamp_queries;timing.frames.timestamp_queries=output->frames.timestamp_queries;
        timing.reconstruction.timestamp_queries=output->reconstruction.timestamp_queries;timing.response.timestamp_queries=output->response.timestamp_queries;
    }
};
GpuPipeline::GpuPipeline(const GpuPipelineConfig& config,std::shared_ptr<gpu::Context> context,std::shared_ptr<SampleClock> clock):
    impl_(std::make_unique<Impl>(config,std::move(context),std::move(clock))){}
GpuPipeline::~GpuPipeline()=default;
std::vector<ReceiverStatus> GpuPipeline::status() const {return impl_->receiver.status();}
void GpuPipeline::reset(){
    auto& p=*impl_;if(!p.decoder->empty())throw std::logic_error("Drain the pipeline before resetting");
    p.receiver.reset();p.frames.reset();p.origins.assign(p.origins.size(),Impl::Origin{});
    if(p.clock)for(std::size_t c=0;c<p.config.guards.size();++c)p.clock->reset_observer(c);
    p.timing={};p.reported={};
}
void GpuPipeline::process(const ChannelBatch& input,const Consume& consume,GpuPipelineTimings* timing) {
    auto& p=*impl_;p.detail(timing);
    p.collect(consume,false,timing!=nullptr);
    gpu::Commands commands(p.context);
    const auto received=p.receiver.process(input,timing ? &p.timing.receiver : nullptr,&commands);
    if(p.clock)observe_timing(*p.clock,received.timing,p.config.origin,p.config.decimation);
    p.frames.select(received,[&](const FrameSelection& frame){p.capture(frame,commands,consume,timing!=nullptr);},timing ? &p.timing.frames : nullptr,&commands);
    commands.submit();p.collect(consume,false,timing!=nullptr);p.report(timing);
}
void GpuPipeline::drain(const Consume& consume,GpuPipelineTimings* timing) {
    auto& p=*impl_;p.detail(timing);p.collect(consume,true,timing!=nullptr);p.frames.wait();p.receiver.wait();p.report(timing);
}
void GpuPipeline::finish(const Consume& consume,GpuPipelineTimings* timing) {
    auto& p=*impl_;p.detail(timing);gpu::Commands commands(p.context);
    p.frames.finish_select([&](const FrameSelection& frame){p.capture(frame,commands,consume,timing!=nullptr);});
    commands.submit();p.collect(consume,true,timing!=nullptr);p.frames.wait();p.receiver.wait();p.report(timing);
}
}
