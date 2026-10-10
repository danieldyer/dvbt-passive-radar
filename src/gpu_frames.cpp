#include "gpu_frames.hpp"
#include "gpu_compute.hpp"
#include <deque>

namespace radar {
namespace {
constexpr std::uint32_t copy_code[] =
#include "carrier_copy.inc"
;
constexpr unsigned capacity=1020+2*68;
OfdmBatch copy_rows(gpu::Kernel& kernel,const OfdmBatch& source,std::span<const std::uint32_t> rows,
                   unsigned destination,unsigned size,ComputeTimings* timings,gpu::Commands* commands=nullptr) {
    if(rows.empty() || rows.size()>size || source.pilot_stride<2 || source.pilot_stride>6817 ||
       source.measured.bytes<source.count*6817*sizeof(C) || source.equalized.bytes<source.count*6817*sizeof(C) ||
       source.pilots.bytes<source.count*source.pilot_stride*sizeof(C))throw std::invalid_argument("Invalid carrier gather dimensions");
    for(auto row:rows)if(row>=source.count)throw std::invalid_argument("Carrier gather row outside source");
    kernel.bind(0,source.measured);kernel.bind(1,source.equalized);kernel.bind(2,source.pilots);
    kernel.storage(3,size*6817*sizeof(C));kernel.storage(4,size*6817*sizeof(C));
    kernel.storage(5,size*source.pilot_stride*sizeof(C));kernel.upload(6,rows);
    const auto cmd=kernel.begin(timings,2,commands);
    kernel.dispatch(cmd,kernel.primary(),std::array{static_cast<unsigned>(rows.size()),static_cast<unsigned>(source.pilot_stride),destination,size},rows.size()*6817);
    kernel.finish(cmd);
    return {kernel.view(3),kernel.view(4),kernel.view(5),rows.size(),source.pilot_stride};
}
}
struct GpuFrameStorage::Impl {
    gpu::Kernel kernel;
    explicit Impl(std::shared_ptr<gpu::Context> context):kernel(copy_code,16,{},std::move(context)){}
};
GpuFrameStorage::GpuFrameStorage(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuFrameStorage::~GpuFrameStorage()=default;
void GpuFrameStorage::wait(){impl_->kernel.wait();}
FrameBatch GpuFrameStorage::capture(const FrameSelection& selection,ComputeTimings* timings,gpu::Commands* commands) {
    auto frame=selection.frame;
    frame.carriers=copy_rows(impl_->kernel,frame.carriers,selection.rows,0,1020,timings,commands);
    return frame;
}
struct GpuFrames::Impl {
    struct Entry {std::uint32_t slot;TrackedSymbol symbol;};
    struct Channel {
        std::unique_ptr<gpu::Kernel> ring;
        OfdmBatch carriers;
        std::deque<Entry> pending,aligned;
        std::vector<Tps> frames;
        std::uint64_t written=0;
        unsigned guard,segment=0;
        bool history=false;
    };
    Modem tables;
    std::shared_ptr<gpu::Context> context;
    GpuFrameStorage staging;
    std::vector<Channel> channels;
    Impl(std::span<const unsigned> guards,std::shared_ptr<gpu::Context> context):context(std::move(context)),staging(this->context) {
        if(guards.empty() || guards.size()>64)throw std::invalid_argument("Invalid frame-buffer channel count");
        for(unsigned guard:guards) {
            if(guard!=4 && guard!=8 && guard!=16 && guard!=32)throw std::invalid_argument("Invalid frame-buffer guard interval");
            Channel c;c.guard=guard;c.ring=std::make_unique<gpu::Kernel>(copy_code,16,std::span<const std::uint32_t>{},this->context);
            channels.push_back(std::move(c));
        }
    }
    void emit(unsigned channel,bool final,const Select& consume) {
        auto& c=channels[channel];if(c.frames.empty())return;
        FrameSelection selection;auto& output=selection.frame;output.frames=c.frames;output.channel=channel;output.segment=c.segment;
        auto& rows=selection.rows;rows.reserve(c.aligned.size());output.symbols.reserve(c.aligned.size());
        for(const auto& entry:c.aligned){rows.push_back(entry.slot);output.symbols.push_back(entry.symbol);}
        output.carriers=c.carriers;
        output.first=c.history ? 68 : 0;output.last=final ? rows.size() : rows.size()-68;
        consume(selection);
        if(final){c.aligned.clear();c.frames.clear();c.history=false;}
        else {
            c.aligned.erase(c.aligned.begin(),c.aligned.end()-136);
            c.frames.erase(c.frames.begin(),c.frames.end()-2);c.history=true;
        }
    }
    void accept(unsigned channel,Entry entry,const Select& consume) {
        auto& c=channels[channel];
        if(entry.symbol.discontinuity) {
            emit(channel,true,consume);c.pending.clear();++c.segment;
        }
        c.pending.push_back(std::move(entry));
        if(c.pending.size()<68)return;
        std::array<std::uint8_t,68> bits{};
        bool indices=true;
        for(unsigned j=0;j<68;++j) {
            indices=indices && c.pending[j].symbol.modulo4==j%4;
            if(j)bits[j]=(c.pending[j].symbol.tps*std::conj(c.pending[j-1].symbol.tps)).real()<0;
        }
        Tps frame;
        if(!indices || !tables.tps_bits(bits,frame)) {
            if(!c.frames.empty()){emit(channel,true,consume);++c.segment;}
            c.pending.pop_front();return;
        }
        const int guard=c.guard==32 ? 0 : c.guard==16 ? 1 : c.guard==8 ? 2 : 3;
        if(frame.guard!=guard)throw std::runtime_error("BCH-valid TPS disagrees with receiver guard interval");
        if(!c.frames.empty()) {
            const auto& last=c.frames.back();
            if(frame.frame!=(last.frame+1)%4 || frame.constellation!=last.constellation || frame.rate!=last.rate ||
               frame.mode!=last.mode || frame.hierarchy!=last.hierarchy) {emit(channel,true,consume);++c.segment;}
        }
        if(c.frames.empty())c.pending.front().symbol.discontinuity=true;
        for(auto& row:c.pending){row.symbol.segment=c.segment;c.aligned.push_back(std::move(row));}
        c.pending.clear();c.frames.push_back(frame);
        if(c.frames.size()==15)emit(channel,false,consume);
    }
    void append(const ReceiverBatch& input,const Select& consume,ComputeTimings* timings,gpu::Commands* commands) {
        if(input.symbols.empty())return;
        if(input.symbols.size()!=input.carriers.count)throw std::invalid_argument("Receiver carrier/metadata count mismatch");
        std::vector<std::vector<std::uint32_t>> selected(channels.size());
        for(unsigned row=0;row<input.symbols.size();++row) {
            const auto& meta=input.symbols[row];
            if(meta.channel>=channels.size())throw std::invalid_argument("Receiver channel outside frame buffer");
            if(meta.ready)selected[meta.channel].push_back(row);
        }
        for(unsigned channel=0;channel<channels.size();++channel) {
            auto& c=channels[channel];
            for(std::size_t at=0;at<selected[channel].size();at+=68) {
                const auto rows=std::span<const std::uint32_t>{selected[channel]}.subspan(at,std::min<std::size_t>(68,selected[channel].size()-at));
                // At most 67 queued symbols can follow a complete batch. This
                // bounded append cannot overwrite its history before emission.
                c.carriers=copy_rows(*c.ring,input.carriers,rows,c.written%capacity,capacity,timings,commands);
                c.carriers.count=capacity;
                for(unsigned j=0;j<rows.size();++j)
                    accept(channel,{static_cast<unsigned>((c.written+j)%capacity),input.symbols[rows[j]]},consume);
                c.written+=rows.size();
            }
        }
    }
};
GpuFrames::GpuFrames(std::span<const unsigned> guards,std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(guards,std::move(context))){}
GpuFrames::~GpuFrames()=default;
void GpuFrames::wait(){for(auto& channel:impl_->channels)channel.ring->wait();impl_->staging.wait();}
void GpuFrames::select(const ReceiverBatch& input,const Select& consume,ComputeTimings* timings,gpu::Commands* commands) {
    impl_->append(input,consume,timings,commands);
}
void GpuFrames::finish_select(const Select& consume) {
    for(unsigned c=0;c<impl_->channels.size();++c){impl_->emit(c,true,consume);impl_->channels[c].pending.clear();}
}
void GpuFrames::append(const ReceiverBatch& input,const Consume& consume,ComputeTimings* timings) {
    select(input,[&](const FrameSelection& selected){consume(impl_->staging.capture(selected,timings));},timings);
}
void GpuFrames::finish(const Consume& consume,ComputeTimings* timings) {
    finish_select([&](const FrameSelection& selected){consume(impl_->staging.capture(selected,timings));});
}
}
