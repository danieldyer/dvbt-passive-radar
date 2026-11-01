#pragma once
#include "vulkan_context.hpp"
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace radar::gpu {
struct Context {
    struct Fence {
        vk::UniqueFence handle;
    };
    vk::UniqueInstance instance = make_instance();
    ComputeDevice gpu = make_device(*instance);
    std::mutex queue_mutex;
    std::mutex fence_mutex;
    std::vector<std::shared_ptr<Fence>> fences;
    std::shared_ptr<Fence> acquire_fence() {
        const std::lock_guard lock(fence_mutex);
        for (auto &fence : fences)
            if (fence.use_count() == 1) {
                gpu.device->resetFences(*fence->handle);
                return fence;
            }
        auto fence = std::make_shared<Fence>(Fence{gpu.device->createFenceUnique({})});
        fences.push_back(fence);
        return fence;
    }
};
namespace detail {
// A submission owns its fence. Recording resources remain owned by their kernels;
// a kernel resolves this submission before changing descriptors, uploads or storage.
class Submission {
    std::shared_ptr<Context> context_;
    std::shared_ptr<Context::Fence> fence_;
    std::mutex mutex_;
    bool submitted_ = false, complete_ = false, collected_ = false;
    std::vector<vk::CommandBuffer> commands_;
    std::vector<std::function<void()>> collect_;

  public:
    explicit Submission(std::shared_ptr<Context> context)
        : context_(std::move(context)), fence_(context_->acquire_fence()) {}
    bool submitted() const {
        return submitted_;
    } // Recording is confined to its owner thread.
    void add(vk::CommandBuffer command, std::function<void()> collect) {
        if (submitted_)
            throw std::logic_error("Cannot append to a submitted command group");
        commands_.push_back(command);
        collect_.push_back(std::move(collect));
    }
    void submit() {
        const std::lock_guard lock(mutex_);
        if (submitted_)
            return;
        const std::lock_guard queue_lock(context_->queue_mutex);
        context_->gpu.device->getQueue(context_->gpu.family, 0)
            .submit(vk::SubmitInfo{{}, {}, commands_}, *fence_->handle);
        submitted_ = true;
    }
    // Workers may wait for readiness, but query collection and resource mutation
    // stay on the recording thread. No callback or borrowed timings run here.
    void wait() {
        submit();
        const std::lock_guard lock(mutex_);
        if (complete_)
            return;
        if (context_->gpu.device->waitForFences(*fence_->handle, true, UINT64_MAX) != vk::Result::eSuccess)
            throw std::runtime_error("GPU command group fence failed");
        complete_ = true;
    }
    void collect() {
        if (collected_)
            return;
        wait();
        // Fence completion is already established. A failed query/callback must
        // propagate once, without being repeated by a destructor during unwind.
        auto callbacks = std::move(collect_);
        collected_ = true;
        for (auto &collect : callbacks)
            collect();
    }
};
} // namespace detail
class Completion {
    std::shared_ptr<detail::Submission> submission_;
    friend class Commands;
    explicit Completion(std::shared_ptr<detail::Submission> submission)
        : submission_(std::move(submission)) {}

  public:
    Completion() = default;
    void wait() const {
        if (submission_)
            submission_->wait();
    }
};
// Explicit command grouping: submit at GPU-only boundaries, wait at CPU reads.
// The owner thread must outlive any pending kernels and timing accumulators.
class Commands {
    std::shared_ptr<Context> context_;
    std::shared_ptr<detail::Submission> current_;

  public:
    explicit Commands(std::shared_ptr<Context> context) : context_(std::move(context)) {}
    ~Commands() {
        if (current_)
            current_->collect();
    }
    const std::shared_ptr<Context> &context() const {
        return context_;
    }
    std::shared_ptr<detail::Submission> add(vk::CommandBuffer command, std::function<void()> collect) {
        if (!current_ || current_->submitted())
            current_ = std::make_shared<detail::Submission>(context_);
        current_->add(command, std::move(collect));
        return current_;
    }
    Completion submit() {
        if (!current_)
            return {};
        current_->submit();
        return Completion{std::exchange(current_, {})};
    }
    void wait() {
        if (current_) {
            current_->collect();
            current_.reset();
        }
    }
};
} // namespace radar::gpu
