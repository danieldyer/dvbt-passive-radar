#pragma once
#include "complex_sample.hpp"
#include "gpu_compute.hpp"
namespace radar::gpu {
class FftPlan;
}
#include <cmath>
#include <deque>

namespace radar::video {
// Range FFTs are immutable per row. Transform each archived batch once, retain
// a GPU ring, and gather the exact sliding CPI directly into the focus stage.
class ProfileHistory {
    struct Row {
        double time, information;
        std::uint64_t index;
    };
    gpu::Kernel kernel_;
    std::unique_ptr<gpu::FftPlan> transform_;
    std::deque<Row> rows_;
    std::size_t required_, capacity_, batch_rows_ = 0, profile_size_ = 0;
    std::uint64_t cursor_ = 0, staged_ = 0, segment_ = UINT64_MAX;
    double interval_, update_, last_publish_ = -INFINITY;
    bool erasures_;
    struct Parameters {
        unsigned rows, size, capacity, first, mode;
    };

  public:
    ProfileHistory(std::shared_ptr<gpu::Context>, double integration, double interval, double update,
                   bool erasures);
    ~ProfileHistory();
    void stage(std::span<const C> moments, std::size_t count, unsigned profile_size,
               ComputeTimings * = nullptr);
    bool append(double time, double information, bool valid, std::uint64_t segment);
    std::size_t required() const {
        return required_;
    }
    double first_time() const {
        return rows_.front().time;
    }
    gpu::BufferView snapshot(std::vector<double> &information, ComputeTimings * = nullptr);
};
} // namespace radar::video
