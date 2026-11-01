#include "profile_history.hpp"
#include "gpu_fft_plan.hpp"
#include "matched_profiles.hpp"
namespace radar::video {
namespace {
constexpr std::uint32_t code[] =
#include "profile_history.inc"
    ;
} // namespace
ProfileHistory::ProfileHistory(std::shared_ptr<gpu::Context> context, double integration, double interval,
                               double update, bool erasures)
    : kernel_(code, sizeof(Parameters), {}, std::move(context)), interval_(interval), update_(update),
      erasures_(erasures) {
    if (!std::isfinite(integration) || integration < .5 || integration > 10 || !std::isfinite(interval) ||
        interval <= 0 || !std::isfinite(update) || update <= 0 || integration / interval > 10000)
        throw std::invalid_argument("Invalid profile history coordinates");
    required_ = std::size_t(std::ceil(integration / interval));
    capacity_ = required_ + 1020;
}
ProfileHistory::~ProfileHistory() = default;
void ProfileHistory::stage(std::span<const C> moments, std::size_t count, unsigned size,
                           ComputeTimings *timing) {
    if (!count || count > 1020 || !matched_profile_size_supported(size) ||
        moments.size() != 2 * count * size || cursor_ != staged_ || (profile_size_ && profile_size_ != size))
        throw std::invalid_argument("Invalid archived profile batch");
    profile_size_ = size;
    // A ring includes one complete future batch so preloading cannot overwrite
    // any row in a CPI that will be published while that batch is consumed.
    kernel_.storage(1, 2 * capacity_ * size * sizeof(C));
    kernel_.upload<C>(0, moments);
    if (batch_rows_ != count) {
        transform_.reset();
        transform_ = std::make_unique<gpu::FftPlan>(kernel_, 0, size, 2 * count);
        batch_rows_ = count;
    }
    const auto cmd = kernel_.begin(timing);
    transform_->append(cmd, false);
    kernel_.dispatch(cmd, kernel_.primary(),
                     Parameters{unsigned(count), size, unsigned(capacity_), unsigned(cursor_ % capacity_), 0},
                     moments.size());
    kernel_.finish(cmd);
    staged_ += count;
}
bool ProfileHistory::append(double time, double information, bool valid, std::uint64_t segment) {
    if (cursor_ >= staged_ || !std::isfinite(time) || !std::isfinite(information) || information <= 0)
        throw std::invalid_argument("Invalid archived profile row");
    const auto index = cursor_++;
    if (segment != segment_) {
        rows_.clear();
        segment_ = segment;
        last_publish_ = -INFINITY;
    }
    if ((!valid && !erasures_) ||
        (!rows_.empty() && std::abs(time - rows_.back().time - interval_) > interval_ * .01))
        rows_.clear();
    if (!valid && !erasures_)
        return false;
    rows_.push_back({time, valid ? information : 0, index});
    if (rows_.size() > required_)
        rows_.pop_front();
    if (rows_.size() != required_ || time - last_publish_ + interval_ * 1e-8 < update_)
        return false;
    if (std::ranges::none_of(rows_, [](const Row &r) { return r.information > 0; }))
        return false;
    last_publish_ = time;
    return true;
}
gpu::BufferView ProfileHistory::snapshot(std::vector<double> &information, ComputeTimings *timing) {
    if (rows_.size() != required_)
        throw std::logic_error("Incomplete archived coherent interval");
    information.resize(required_);
    std::vector<std::uint32_t> valid(required_);
    for (std::size_t r = 0; r < required_; ++r) {
        information[r] = rows_[r].information;
        valid[r] = information[r] > 0;
    }
    kernel_.upload<std::uint32_t>(2, valid);
    kernel_.storage(3, 2 * required_ * profile_size_ * sizeof(C));
    kernel_.run(Parameters{unsigned(required_), unsigned(profile_size_), unsigned(capacity_),
                           unsigned(rows_.front().index % capacity_), 1},
                2 * required_ * profile_size_, timing);
    return kernel_.view(3);
}
} // namespace radar::video
