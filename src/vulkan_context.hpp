#pragma once
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace radar::gpu {
inline auto make_instance()
{
    const vk::ApplicationInfo app{"dvbt_passive_radar", 0, nullptr, 0,
                                VK_API_VERSION_1_2};
    std::vector<const char*> extensions;
    vk::InstanceCreateFlags flags;
    for (const auto& extension : vk::enumerateInstanceExtensionProperties()) {
        if (std::string_view{extension.extensionName.data()} ==
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        }
    }

    std::vector<const char*> layers;
#ifndef NDEBUG
    for (const auto& layer : vk::enumerateInstanceLayerProperties()) {
        if (std::string_view{layer.layerName.data()} == "VK_LAYER_KHRONOS_validation") {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
    std::cout << "Validation: " << (layers.empty() ? "unavailable" : "enabled") << '\n';
#endif
    return vk::createInstanceUnique(vk::InstanceCreateInfo{flags, &app, layers, extensions});
}

struct ComputeDevice {
    vk::PhysicalDevice physical;
    std::uint32_t family;
    vk::UniqueDevice device;
};

inline auto make_device(vk::Instance instance) -> ComputeDevice
{
    for (const auto physical : instance.enumeratePhysicalDevices()) {
        if (physical.getProperties().apiVersion < VK_API_VERSION_1_2) {
            continue;
        }
        const auto families = physical.getQueueFamilyProperties();
        for (std::uint32_t i = 0; i < families.size(); ++i) {
            if (!(families[i].queueFlags & vk::QueueFlagBits::eCompute)) {
                continue;
            }
            std::vector<const char*> extensions;
            for (const auto& extension : physical.enumerateDeviceExtensionProperties()) {
                // Mandatory when advertised by a portability implementation (MoltenVK).
                if (std::string_view{extension.extensionName.data()} ==
                    "VK_KHR_portability_subset") {
                    extensions.push_back("VK_KHR_portability_subset");
                }
            }
            constexpr float priority = 1.0F;
            const vk::DeviceQueueCreateInfo queue_info{{}, i, 1, &priority};
            return {physical, i, physical.createDeviceUnique(
                                    vk::DeviceCreateInfo{{}, queue_info, {}, extensions})};
        }
    }
    throw std::runtime_error("No Vulkan 1.2 compute device found");
}

struct Buffer {
    // Reverse member destruction order: destroy the buffer before freeing its memory.
    vk::UniqueDeviceMemory memory;
    vk::UniqueBuffer handle;
};

inline auto make_buffer(const ComputeDevice& gpu, vk::DeviceSize bytes) -> Buffer
{
    auto handle = gpu.device->createBufferUnique(
        vk::BufferCreateInfo{{}, bytes, vk::BufferUsageFlagBits::eStorageBuffer});
    const auto requirements = gpu.device->getBufferMemoryRequirements(*handle);
    const auto types = gpu.physical.getMemoryProperties();
    constexpr auto properties = vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent;
    for (std::uint32_t i = 0; i < types.memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits & (1U << i)) &&
            (types.memoryTypes[i].propertyFlags & properties) == properties) {
            auto memory = gpu.device->allocateMemoryUnique({requirements.size, i});
            gpu.device->bindBufferMemory(*handle, *memory, 0);
            return {std::move(memory), std::move(handle)};
        }
    }
    throw std::runtime_error("No host-visible coherent storage memory available");
}

}
