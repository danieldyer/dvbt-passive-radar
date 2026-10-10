#pragma once
#include "gpu_buffer.hpp"
#include <cstdint>

namespace radar {
// A borrowed plane per channel, on one corrected sample grid. The producer
// owns storage until the consumer has recorded and completed its GPU use.
struct ChannelBatch {
    gpu::BufferView samples;
    std::uint64_t first{}; // Corrected channel sample, relative to capture origin.
    std::size_t count{},channels{},stride{};
};
}
