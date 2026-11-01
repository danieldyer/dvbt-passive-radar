#pragma once
#include <cstddef>
#include <memory>
#include <vulkan/vulkan_core.h>

namespace radar::gpu {
struct Context;
// Borrowed storage. The owner must outlive the view and must not resize or
// overwrite it before the consuming submission finishes. No host copy implied.
struct BufferView {
    std::shared_ptr<Context> context;
    VkBuffer buffer{};
    std::size_t bytes{};
    void *mapped{};
};
} // namespace radar::gpu
