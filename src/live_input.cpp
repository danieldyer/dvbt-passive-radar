#include "live_input.hpp"
#include "gpu_compute.hpp"
#include <bit>
#include <chrono>
#include <limits>

namespace radar {
std::uint64_t nominal_corrected_sample(std::uint64_t n,std::uint32_t source,std::uint32_t grid){
    if(!source || !grid || n>=(std::uint64_t{1}<<53))throw std::invalid_argument("Invalid live source coordinate");
    // Do not multiply the complete ADC epoch by the rate: that overflows after
    // about 75 minutes at 64 MS/s, even though the sample index is still exact.
    return n/source*grid+(n%source)*std::uint64_t(grid)/source;
}
CaptureRing::CaptureRing(std::span<Iq16> storage):storage_(storage) {
    if(storage.size()<1024 || !std::has_single_bit(storage.size()))throw std::invalid_argument("Capture ring size must be a power of two");
}
std::span<Iq16> CaptureRing::writable(std::size_t maximum) {
    const auto h=head(),t=tail(),offset=h%capacity();
    return storage_.subspan(offset,std::min({maximum,capacity()-std::size_t(h-t),capacity()-std::size_t(offset)}));
}
bool CaptureRing::commit(std::size_t count,std::uint64_t first,std::uint64_t segment) {
    if(!count || count>writable(count).size())throw std::invalid_argument("Invalid capture-ring commit");
    std::lock_guard lock(mutex_);
    if(closed_ || chunks_.size()>=4096)return false;
    const auto h=head();chunks_.push_back({h,first,count,segment});
    peak_.store(std::max(peak(),h+count-tail()),std::memory_order_relaxed);
    head_.store(h+count,std::memory_order_release);condition_.notify_one();return true;
}
bool CaptureRing::next(CaptureChunk& chunk) {
    std::unique_lock lock(mutex_);
    condition_.wait(lock,[&]{return closed_ || !chunks_.empty();});
    if(chunks_.empty())return false;
    chunk=chunks_.front();chunks_.pop_front();return true;
}
bool CaptureRing::next(CaptureChunk& chunk,std::chrono::milliseconds timeout){
    std::unique_lock lock(mutex_);condition_.wait_for(lock,timeout,[&]{return closed_ || !chunks_.empty();});
    if(chunks_.empty())return false;chunk=chunks_.front();chunks_.pop_front();return true;
}
void CaptureRing::release(std::uint64_t position) {
    if(position<tail() || position>head())throw std::invalid_argument("Invalid capture-ring release");
    tail_.store(position,std::memory_order_release);
}
void CaptureRing::close(){std::lock_guard lock(mutex_);closed_=true;condition_.notify_all();}
CaptureTimeline::Position CaptureTimeline::receive(std::optional<std::int64_t> tick,std::size_t count,CaptureCounters& stats,bool accepted) {
    if(!count)return {};
    stats.samples+=count;
    bool boundary=pending_;pending_=false;
    if(!tick){++stats.unknown_gaps;throw std::runtime_error("SDR receive has no timestamp; stopping to preserve sample coordinates");}
    if(!epoch_)epoch_=*tick;
    if(expected_ && *tick!=*expected_) {
        boundary=true;++stats.timestamp_gaps;
        if(*tick>*expected_){stats.missing_samples+=std::uint64_t(*tick-*expected_);if(unresolved_){--stats.unknown_gaps;unresolved_=false;}}
        else {++stats.unknown_gaps;throw std::runtime_error("SDR receive timestamp moved backwards; stopping to preserve coherence");}
    }
    if(boundary && expected_ && accepted)++segment_;
    unresolved_=false;
    expected_=*tick+std::int64_t(count);stats.segments=segment_+1;
    return {std::uint64_t(*tick-*epoch_),segment_,boundary};
}
void CaptureTimeline::error(bool overflow,bool sequence,CaptureCounters& stats) {
    if(overflow)++stats.overflows;if(sequence)++stats.sequence_errors;
    pending_=true;if(!unresolved_){++stats.unknown_gaps;unresolved_=true;}
}
void CaptureTimeline::discard(std::size_t count,CaptureCounters& stats){stats.host_discarded+=count;pending_=true;}
namespace {
constexpr std::uint32_t code[]=
#include "live_unpack.inc"
;
}
struct GpuLiveInput::Impl {
    gpu::Kernel kernel;
    std::size_t ring_size,input_size;
    Impl(std::shared_ptr<gpu::Context> context,std::size_t ring,std::size_t input):
        kernel(code,16,{},std::move(context)),ring_size(ring),input_size(input) {
        if(!std::has_single_bit(ring) || ring>UINT32_MAX || input>UINT32_MAX || input>ring)
            throw std::invalid_argument("Invalid live GPU buffer sizes");
        kernel.storage(0,ring*sizeof(Iq16));kernel.storage(1,input*sizeof(C));
    }
};
GpuLiveInput::GpuLiveInput(std::shared_ptr<gpu::Context> context,std::size_t ring,std::size_t input):impl_(std::make_unique<Impl>(std::move(context),ring,input)){}
GpuLiveInput::~GpuLiveInput()=default;
std::span<Iq16> GpuLiveInput::storage() const {return {static_cast<Iq16*>(impl_->kernel.view(0).mapped),impl_->ring_size};}
gpu::BufferView GpuLiveInput::gather(std::uint64_t first,std::size_t count,ComputeTimings* timing) {
    auto& p=*impl_;if(count>p.input_size)throw std::invalid_argument("Live input exceeds FFT capacity");
    p.kernel.run(std::array{std::uint32_t(first%p.ring_size),std::uint32_t(count),std::uint32_t(p.input_size),std::uint32_t(p.ring_size-1)},p.input_size,timing);
    return p.kernel.view(1);
}
}
