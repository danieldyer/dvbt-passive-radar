#pragma once
#include "gpu_commands.hpp"
#include "gpu_timings.hpp"
#include "gpu_buffer.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>

namespace radar::gpu {
// Persistent unified-memory buffers and one readiness fence per command group.
// Resolve pending use before changing descriptors, uploads or command storage.
class Kernel {
    static constexpr unsigned storage_bindings=10;
    std::shared_ptr<Context> context_;
    const gpu::ComputeDevice& gpu_;
    vk::Device device_=*gpu_.device;
    vk::UniqueCommandPool pool_=device_.createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer,gpu_.family});
    vk::UniqueDescriptorSetLayout descriptors_;
    vk::UniquePipelineLayout layout_;
    vk::UniquePipeline pipeline_;
    vk::UniqueDescriptorPool descriptor_pool_;
    vk::DescriptorSet descriptor_;
    std::vector<vk::UniqueCommandBuffer> commands_;
    vk::UniqueFence fence_=device_.createFenceUnique({});
    std::array<gpu::Buffer,storage_bindings> buffers_;
    std::array<std::size_t,storage_bindings> capacities_{};
    std::array<void*,storage_bindings> mapped_{};
    std::array<BufferView,storage_bindings> external_{};
    vk::UniqueQueryPool timestamps_;
    ComputeTimings* timings_=nullptr;
    unsigned timestamp_count_=2;
    Commands* group_=nullptr;
    std::shared_ptr<detail::Submission> pending_;
public:
    Kernel(std::span<const std::uint32_t> code, unsigned push_bytes,std::span<const std::uint32_t> constants={},
           std::shared_ptr<Context> context={}) : context_(context ? std::move(context) : std::make_shared<Context>()),gpu_(context_->gpu) {
        std::array<vk::DescriptorSetLayoutBinding,storage_bindings> bindings;
        for(unsigned i=0;i<storage_bindings;++i) bindings[i]={i,vk::DescriptorType::eStorageBuffer,1,vk::ShaderStageFlagBits::eCompute};
        descriptors_=device_.createDescriptorSetLayoutUnique({{},bindings});
        const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute,0,push_bytes};
        layout_=device_.createPipelineLayoutUnique({{},*descriptors_,push});
        pipeline_=pipeline(code,constants);
        const vk::DescriptorPoolSize size{vk::DescriptorType::eStorageBuffer,storage_bindings};
        descriptor_pool_=device_.createDescriptorPoolUnique({{},1,size});
        descriptor_=device_.allocateDescriptorSets({*descriptor_pool_,*descriptors_}).front();
        commands_=device_.allocateCommandBuffersUnique({*pool_,vk::CommandBufferLevel::ePrimary,1});
    }
    ~Kernel(){wait();}
    void wait() {if(pending_){auto pending=pending_;pending->collect();pending_.reset();}}
    vk::UniquePipeline pipeline(std::span<const std::uint32_t> code,std::span<const std::uint32_t> constants={}) {
        const auto shader=device_.createShaderModuleUnique({{},code.size_bytes(),code.data()});
        std::vector<vk::SpecializationMapEntry> entries;
        for(std::uint32_t i=0;i<constants.size();++i)entries.emplace_back(i,i*sizeof(std::uint32_t),sizeof(std::uint32_t));
        const vk::SpecializationInfo specialization{static_cast<std::uint32_t>(entries.size()),entries.data(),constants.size_bytes(),constants.data()};
        return device_.createComputePipelineUnique({},{{},{ {},vk::ShaderStageFlagBits::eCompute,*shader,"main",constants.empty() ? nullptr : &specialization},*layout_}).value;
    }
    std::shared_ptr<Context> context() const {return context_;}
    VkDevice device() const {return device_;}
    VkPhysicalDevice physical() const {return gpu_.physical;}
    VkQueue queue() const {return device_.getQueue(gpu_.family,0);}
    VkCommandPool pool() const {return *pool_;}
    VkFence fence() const {return *fence_;}
    VkBuffer buffer(unsigned binding) const {return external_[binding].buffer ? external_[binding].buffer : VkBuffer(*buffers_[binding].handle);}
    BufferView view(unsigned binding) const {
        if(binding>=storage_bindings || (!external_[binding].buffer && !buffers_[binding].handle))
            throw std::logic_error("GPU buffer has not been allocated");
        if(external_[binding].buffer)return external_[binding];
        return {context_,buffer(binding),capacities_[binding],mapped_[binding]};
    }
    void bind(unsigned binding,const BufferView& view) {
        wait();
        if(binding>=storage_bindings || view.context!=context_ || !view.buffer || !view.bytes ||
           view.bytes>gpu_.physical.getProperties().limits.maxStorageBufferRange)
            throw std::invalid_argument("Invalid GPU buffer view or mismatched device context");
        const vk::DescriptorBufferInfo info{view.buffer,0,view.bytes};
        device_.updateDescriptorSets(vk::WriteDescriptorSet{descriptor_,binding,0,vk::DescriptorType::eStorageBuffer,{},info},{});
        external_[binding]=view;
    }
    vk::Pipeline primary() const {return *pipeline_;}
    vk::CommandBuffer begin(ComputeTimings* timings=nullptr,unsigned timestamp_count=2,Commands* group=nullptr) {
        wait();
        if(group && group->context()!=context_)throw std::invalid_argument("Command group device mismatch");
        group_=group;
        timings_=timings;
        timestamp_count_=timestamp_count;
        if(timings && !timestamps_ && gpu_.physical.getQueueFamilyProperties()[gpu_.family].timestampValidBits)
            timestamps_=device_.createQueryPoolUnique({{},vk::QueryType::eTimestamp,6});
        auto cmd=*commands_.front();cmd.reset();cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        if(timings_ && timestamps_) {cmd.resetQueryPool(*timestamps_,0,timestamp_count_);cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,*timestamps_,0);}
        return cmd;
    }
    void stamp(vk::CommandBuffer cmd,unsigned index) {
        if(timings_ && timestamps_ && (timestamp_count_==6 || index==timestamp_count_-1))
            cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,*timestamps_,index);
    }
    void* storage(unsigned binding,std::size_t bytes) {
        wait();
        if(binding>=storage_bindings)throw std::invalid_argument("Invalid GPU buffer binding");
        const auto limits=gpu_.physical.getProperties().limits;
        if(!bytes || bytes>limits.maxStorageBufferRange) throw std::runtime_error("GPU compute buffer exceeds device limits");
        bool changed=false;
        if(bytes>capacities_[binding]) {
            auto& b=buffers_[binding];
            if(b.memory) device_.unmapMemory(*b.memory);
            b.handle.reset(); b.memory.reset();
            b=gpu::make_buffer(gpu_,bytes); capacities_[binding]=bytes;
            mapped_[binding]=device_.mapMemory(*b.memory,0,bytes);
            changed=true;
        }
        if(changed || external_[binding].buffer) {
            const vk::DescriptorBufferInfo info{*buffers_[binding].handle,0,capacities_[binding]};
            device_.updateDescriptorSets(vk::WriteDescriptorSet{descriptor_,binding,0,vk::DescriptorType::eStorageBuffer,{},info},{});
            external_[binding]={};
        }
        return mapped_[binding];
    }
    template<class T> void upload(unsigned binding,std::span<const T> values) {
        std::memcpy(storage(binding,values.size_bytes()),values.data(),values.size_bytes());
    }
    template<class T> void dispatch(vk::CommandBuffer cmd,vk::Pipeline pipeline,const T& push,std::size_t count,std::size_t channels=1) {
        const auto limits=gpu_.physical.getProperties().limits;
        const auto groups=(count+63)/64;
        const auto x=std::min<std::size_t>(groups,limits.maxComputeWorkGroupCount[0]);
        const auto y=channels==1 ? (groups+x-1)/x : channels;
        if(y>limits.maxComputeWorkGroupCount[1] || (channels>1 && x!=groups))
            throw std::runtime_error("GPU compute dispatch exceeds device limits");
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,*layout_,0,descriptor_,{});
        cmd.pushConstants<T>(*layout_,vk::ShaderStageFlagBits::eCompute,0,push);
        cmd.dispatch(static_cast<std::uint32_t>(x),static_cast<std::uint32_t>(y),1);
    }
    template<class T> void groups(vk::CommandBuffer cmd,vk::Pipeline pipeline,const T& push,unsigned x,unsigned y=1) {
        const auto limits=gpu_.physical.getProperties().limits;
        if(!x || !y || x>limits.maxComputeWorkGroupCount[0] || y>limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("GPU compute dispatch exceeds device limits");
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute,pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,*layout_,0,descriptor_,{});
        cmd.pushConstants<T>(*layout_,vk::ShaderStageFlagBits::eCompute,0,push);
        cmd.dispatch(x,y,1);
    }
    static void barrier(vk::CommandBuffer cmd) {
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader,{},barrier,{},{});
    }
    void finish(vk::CommandBuffer cmd) {
        stamp(cmd,timestamp_count_-1);
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite|vk::AccessFlagBits::eHostRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader|vk::PipelineStageFlagBits::eHost,{},barrier,{},{});
        cmd.end();
        if(group_)pending_=group_->add(cmd,[this]{collect();});
        else {
            Commands commands(context_);pending_=commands.add(cmd,[this]{collect();});commands.wait();
        }
        group_=nullptr;
    }
    void collect() {
        if(timings_) {
            ++timings_->batches;
            if(timestamps_) {
                std::array<std::uint64_t,6> ticks{};
                const auto result=device_.getQueryPoolResults(*timestamps_,0,timestamp_count_,sizeof(ticks),ticks.data(),sizeof(ticks[0]),vk::QueryResultFlagBits::e64|vk::QueryResultFlagBits::eWait);
                if(result!=vk::Result::eSuccess)throw std::runtime_error("GPU timestamp query failed");
                timings_->gpu_timestamps_available=true;
                const auto bits=gpu_.physical.getQueueFamilyProperties()[gpu_.family].timestampValidBits;
                const auto mask=bits==64 ? UINT64_MAX : (std::uint64_t{1}<<bits)-1;
                const double period=gpu_.physical.getProperties().limits.timestampPeriod*1e-9;
                timings_->gpu+=double((ticks[timestamp_count_-1]-ticks[0])&mask)*period;
                if(timestamp_count_==6)for(unsigned i=0;i<5;++i)timings_->stages[i]+=double((ticks[i+1]-ticks[i])&mask)*period;
            }
        }
    }
    template<class T> void run(const T& push,std::size_t count,ComputeTimings* timings=nullptr) {
        const auto cmd=begin(timings);dispatch(cmd,*pipeline_,push,count);finish(cmd);
    }
};
}
