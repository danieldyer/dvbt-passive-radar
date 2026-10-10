#pragma once
#include "gpu_compute.hpp"
#include <vkFFT.h>
#include <stdexcept>
#include <string>

namespace radar::gpu {
// One persistent buffer and shape per plan. The owning kernel outlives it.
class FftPlan {
    VkDevice device_; VkPhysicalDevice physical_; VkQueue queue_;
    VkCommandPool pool_; VkFence fence_; VkBuffer buffer_;
    std::uint64_t bytes_;
    VkFFTApplication application_{};
public:
    FftPlan(Kernel& kernel,unsigned binding,std::size_t size,std::size_t batches,std::size_t offset=0)
        :device_(kernel.device()),physical_(kernel.physical()),queue_(kernel.queue()),
         pool_(kernel.pool()),fence_(kernel.fence()),buffer_(kernel.buffer(binding)),
         bytes_(kernel.view(binding).bytes) {
        VkFFTConfiguration config{};
        config.FFTdim=1;config.size[0]=size;config.numberBatches=batches;
        config.device=&device_;config.physicalDevice=&physical_;config.queue=&queue_;
        config.commandPool=&pool_;config.fence=&fence_;config.buffer=&buffer_;
        config.bufferSize=&bytes_;config.bufferOffset=offset;config.normalize=1;config.useLUT=1;
        // VkFFT uploads plan tables through the queue while initializing.
        // Share the submission lock with every other owner of that queue.
        const std::lock_guard queue_lock(kernel.context()->queue_mutex);
        const auto result=initializeVkFFT(&application_,config);
        if(result!=VKFFT_SUCCESS){deleteVkFFT(&application_);throw std::runtime_error("Compute FFT plan failed: "+std::to_string(result));}
    }
    FftPlan(const FftPlan&)=delete;
    FftPlan& operator=(const FftPlan&)=delete;
    ~FftPlan(){deleteVkFFT(&application_);}
    void append(vk::CommandBuffer command,bool inverse) {
        VkCommandBuffer raw=command;VkFFTLaunchParams launch{};launch.commandBuffer=&raw;
        const auto result=VkFFTAppend(&application_,inverse ? 1 : -1,&launch);
        if(result!=VKFFT_SUCCESS)throw std::runtime_error("Compute FFT dispatch failed: "+std::to_string(result));
        Kernel::barrier(command);
    }
};
}
