#include "gpu_residual.hpp"
#include "vulkan_context.hpp"
#include <array>
#include <chrono>
#include <cstring>
namespace radar {
namespace {
constexpr std::uint32_t spirv[] =
#include "residual.inc"
;
}
struct GpuResidual::Impl {
    vk::UniqueInstance instance=gpu::make_instance();
    gpu::ComputeDevice gpu=gpu::make_device(*instance);
    vk::Device device=*gpu.device;
    vk::UniqueCommandPool commands=device.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer,gpu.family});
    vk::UniqueDescriptorSetLayout descriptor_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::UniqueDescriptorPool descriptor_pool;
    vk::DescriptorSet descriptor;
    std::vector<vk::UniqueCommandBuffer> command;
    gpu::Buffer buffer;
    vk::DeviceSize capacity=0;
    Impl() {
        const vk::DescriptorSetLayoutBinding binding{0,vk::DescriptorType::eStorageBuffer,1,vk::ShaderStageFlagBits::eCompute};
        descriptor_layout=device.createDescriptorSetLayoutUnique({{},binding});
        const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute,0,8};
        layout=device.createPipelineLayoutUnique({{},*descriptor_layout,push});
        const auto shader=device.createShaderModuleUnique({{},sizeof(spirv),spirv});
        pipeline=device.createComputePipelineUnique({},{{},{ {},vk::ShaderStageFlagBits::eCompute,*shader,"main"},*layout}).value;
        const vk::DescriptorPoolSize size{vk::DescriptorType::eStorageBuffer,1};
        descriptor_pool=device.createDescriptorPoolUnique({{},1,size});
        descriptor=device.allocateDescriptorSets({*descriptor_pool,*descriptor_layout}).front();
        command=device.allocateCommandBuffersUnique({*commands,vk::CommandBufferLevel::ePrimary,1});
        std::cout<<"Residual compute device: "<<gpu.physical.getProperties().deviceName<<'\n';
    }
};
GpuResidual::GpuResidual() : impl_(std::make_unique<Impl>()) {}
GpuResidual::~GpuResidual()=default;
std::vector<C> GpuResidual::calculate(std::span<const C> y,std::span<const C> x,std::span<const C> h,ResidualTimings* timings) {
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    auto measure=[&](double ResidualTimings::* stage) {
        if(timings) {const auto now=Clock::now();timings->*stage+=std::chrono::duration<double>(now-mark).count();mark=now;}
    };
    if(y.size()!=x.size() || x.empty() || h.empty() || x.size()%h.size() || x.size()>UINT32_MAX)
        throw std::runtime_error("Invalid GPU residual dimensions");
    auto& p=*impl_;
    const auto bytes=(3*x.size()+h.size())*sizeof(C);
    const auto groups=(x.size()+127)/128;
    const auto limits=p.gpu.physical.getProperties().limits;
    if(bytes>limits.maxStorageBufferRange || groups>limits.maxComputeWorkGroupCount[0])
        throw std::runtime_error("Residual batch exceeds GPU limits");
    if(bytes>p.capacity) {
        // Explicitly release handle before memory; move assignment of a resource pair would free memory first.
        p.buffer.handle.reset();p.buffer.memory.reset();
        p.buffer=gpu::make_buffer(p.gpu,bytes);p.capacity=bytes;
    }
    const vk::DescriptorBufferInfo info{*p.buffer.handle,0,bytes};
    p.device.updateDescriptorSets(vk::WriteDescriptorSet{p.descriptor,0,0,vk::DescriptorType::eStorageBuffer,{},info},{});
    auto* mapped=static_cast<C*>(p.device.mapMemory(*p.buffer.memory,0,bytes));
    std::memcpy(mapped,y.data(),y.size_bytes());
    std::memcpy(mapped+x.size(),x.data(),x.size_bytes());
    std::memcpy(mapped+2*x.size(),h.data(),h.size_bytes());
    p.device.unmapMemory(*p.buffer.memory);
    measure(&ResidualTimings::upload);
    auto cmd=*p.command.front();cmd.reset();cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute,*p.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,*p.layout,0,p.descriptor,{});
    const std::array<std::uint32_t,2> dimensions{static_cast<std::uint32_t>(x.size()),static_cast<std::uint32_t>(h.size())};
    cmd.pushConstants<std::uint32_t>(*p.layout,vk::ShaderStageFlagBits::eCompute,0,dimensions);
    cmd.dispatch(static_cast<std::uint32_t>(groups),1,1);
    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eHostRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eHost,{},barrier,{},{});
    cmd.end();p.device.getQueue(p.gpu.family,0).submit(vk::SubmitInfo{{},{},cmd});p.device.waitIdle();
    measure(&ResidualTimings::dispatch_wait);
    mapped=static_cast<C*>(p.device.mapMemory(*p.buffer.memory,0,bytes));
    std::vector<C> result(mapped+2*x.size()+h.size(),mapped+3*x.size()+h.size());
    p.device.unmapMemory(*p.buffer.memory);
    measure(&ResidualTimings::readback);
    return result;
}
}
