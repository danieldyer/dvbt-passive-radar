#pragma once
#include <vkFFT.h>
#include <vulkan/vulkan.hpp>

namespace radar::gpu {
inline void configure_fft(VkFFTConfiguration &config) {
    const auto properties = vk::PhysicalDevice(*config.physicalDevice).getProperties();
    // Keep transforms exceeding Apple's 32 KiB workgroup storage in one
    // global pass. Register reuse changes decomposition, not precision.
    if (properties.vendorID == 0x106b && properties.limits.maxComputeSharedMemorySize == 32768 &&
        config.FFTdim == 1 && !config.doublePrecision && !config.halfPrecision) {
        if (config.size[0] == 8192 || config.size[0] == 6144 || config.size[0] == 5760)
            config.registerBoost = 2;
        if (config.size[0] == 12288)
            config.registerBoost = 4;
        if (config.size[0] == 6144 || config.size[0] == 12288 || config.size[0] == 5760)
            config.registerBoostNonPow2 = 1;
    }
}
} // namespace radar::gpu
