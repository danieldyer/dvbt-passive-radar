#pragma once
#include "capture_source.hpp"
#include "matched_profiles.hpp"
#include "sar_handoff.hpp"
#include <atomic>

namespace radar {
// Bounded asynchronous disk output. Only owned byte records enter its queue;
// no borrowed pipeline/GPU storage crosses the worker boundary.
class LiveArchive {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    LiveArchive(const std::filesystem::path&,const RecordingMetadata&,const AppliedCapture&,const SarCapture&,std::string_view mode);
    ~LiveArchive();
    void profiles(const SarBatch&,const MatchedProfiles&,std::span<const C> stationary,std::span<const float> variance={});
    void channels(const ChannelBatch&,std::span<const ClockSegment>);
    void gap(const CaptureChunk&);
    void epoch(std::int64_t hardware_tick,double rate_hz);
    void finish();
    std::uint64_t bytes() const;
    std::size_t queued_bytes() const;
};
}
