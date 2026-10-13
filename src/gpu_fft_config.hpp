#pragma once
#include <vulkan/vulkan.hpp>
#include <vkFFT.h>

namespace radar::gpu {
inline void configure_fft(VkFFTConfiguration& config) {
    const auto properties=vk::PhysicalDevice(*config.physicalDevice).getProperties();
    // An 8K complex-float transform exceeds Apple's 32 KiB shared memory.
    // VkFFT's generic Vulkan default spills it into a second global pass.
    // Two values per register group keep this transform in one pass. This
    // changes only the FFT decomposition, never the arithmetic precision.
    if(properties.vendorID==0x106b && properties.limits.maxComputeSharedMemorySize==32768 &&
       config.FFTdim==1 && config.size[0]==8192 && !config.doublePrecision && !config.halfPrecision)
        config.registerBoost=2;
}
}
