#pragma once
#include "gpu_frontend.hpp"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>

namespace radar {
struct Iq16 {std::int16_t i,q;};
static_assert(sizeof(Iq16)==4);
std::uint64_t nominal_corrected_sample(std::uint64_t source_sample,std::uint32_t source_rate,std::uint32_t grid_rate);
struct CaptureChunk {std::uint64_t ring_first{},source_first{};std::size_t count{};std::uint64_t segment{};};
// A single receive producer writes directly into mapped GPU storage. The DSP
// consumer releases only samples outside its next centered-filter halo.
class CaptureRing {
    std::span<Iq16> storage_;
    std::atomic<std::uint64_t> head_{},tail_{},peak_{};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<CaptureChunk> chunks_;
    bool closed_=false;
public:
    explicit CaptureRing(std::span<Iq16> storage);
    std::span<Iq16> writable(std::size_t maximum);
    bool commit(std::size_t count,std::uint64_t source_first,std::uint64_t segment);
    bool next(CaptureChunk&);
    bool next(CaptureChunk&,std::chrono::milliseconds timeout);
    void release(std::uint64_t position);
    void close();
    std::uint64_t head() const {return head_.load(std::memory_order_acquire);}
    std::uint64_t tail() const {return tail_.load(std::memory_order_acquire);}
    std::uint64_t peak() const {return peak_.load(std::memory_order_relaxed);}
    std::size_t capacity() const {return storage_.size();}
};
struct CaptureCounters {
    std::atomic<std::uint64_t> samples{},overflows{},sequence_errors{},timeouts{},
        timestamp_gaps{},missing_samples{},host_discarded{},unknown_gaps{},segments{},replay_loops{};
};
// Timestamps are receive sample ticks relative to the first successful receive.
// An error never invents a lost-sample count. A later timestamp quantifies it.
class CaptureTimeline {
    std::optional<std::int64_t> epoch_,expected_;
    bool pending_=false,unresolved_=false;
    std::uint64_t segment_{};
public:
    struct Position {std::uint64_t first{},segment{};bool discontinuity{};};
    Position receive(std::optional<std::int64_t> tick,std::size_t count,CaptureCounters&,bool accepted=true);
    void error(bool overflow,bool sequence,CaptureCounters&);
    void discard(std::size_t count,CaptureCounters&);
    void restart(){pending_=true;} // Known synthetic boundary; not a sample loss.
};
// One inexpensive GPU gather converts the circular sc16 input to the FFT's
// cf32 view and pads inactive FFT input. There is no CPU waveform copy.
class GpuLiveInput {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    GpuLiveInput(std::shared_ptr<gpu::Context>,std::size_t ring_samples,std::size_t input_samples);
    ~GpuLiveInput();
    std::span<Iq16> storage() const;
    gpu::BufferView gather(std::uint64_t ring_first,std::size_t count,ComputeTimings* =nullptr);
};
}
