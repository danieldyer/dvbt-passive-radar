#pragma once
#include <array>
#include <cstdint>
namespace radar {
struct ComputeTimings {
    double read{},dispatch_wait{},readback{},gpu{};
    std::uint64_t batches{};
    bool gpu_timestamps_available=false;
    bool detailed=false;
    bool timestamp_queries=true; // Live operation can retain CPU counters without GPU queries.
    std::array<double,5> stages{}; // Stage meanings belong to the calling pipeline.
};
}
