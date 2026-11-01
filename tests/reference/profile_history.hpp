#pragma once
#include "complex_sample.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace radar::video {
// Offline replay of the archived detector's consecutive-row/erasure policy.
// Values already carry the capture's correction: never rotate/resample again.
class ReferenceProfileHistory {
    struct Row {
        std::vector<C> values;
        double time, information;
    };
    std::deque<Row> rows_;
    std::size_t required_;
    double interval_, update_, last_publish_ = -std::numeric_limits<double>::infinity();
    std::uint64_t segment_ = UINT64_MAX;
    bool erasures_;
    std::size_t profile_size_ = 0;
    static std::size_t required(double integration, double interval, double update) {
        if (!std::isfinite(integration) || integration < .5 || integration > 10 || !std::isfinite(interval) ||
            interval <= 0 || !std::isfinite(update) || update <= 0 || integration / interval > 10000)
            throw std::invalid_argument("Invalid profile history coordinates");
        return std::size_t(std::ceil(integration / interval));
    }

  public:
    ReferenceProfileHistory(double integration, double interval, double update, bool erasures = false)
        : required_(required(integration, interval, update)), interval_(interval), update_(update),
          erasures_(erasures) {}
    bool append(double time, double information, bool valid, std::uint64_t segment, std::span<const C> zero,
                std::span<const C> one) {
        if (!std::isfinite(time) || !std::isfinite(information) || information <= 0 ||
            (zero.size() != 1024 && zero.size() != 3072 && zero.size() != 4096) || one.size() != zero.size())
            throw std::invalid_argument("Invalid archived profile row");
        if (profile_size_ && profile_size_ != zero.size())
            throw std::invalid_argument("Archived profile size changed");
        profile_size_ = zero.size();
        if (segment != segment_) {
            rows_.clear();
            segment_ = segment;
            last_publish_ = -std::numeric_limits<double>::infinity();
        }
        if ((!valid && !erasures_) ||
            (!rows_.empty() && std::abs(time - rows_.back().time - interval_) > interval_ * .01))
            rows_.clear();
        if (!valid && !erasures_)
            return false;
        auto &row = rows_.emplace_back();
        row.time = time;
        row.information = valid ? information : 0;
        row.values.resize(2 * profile_size_);
        if (valid) {
            std::ranges::copy(zero, row.values.begin());
            std::ranges::copy(one, row.values.begin() + profile_size_);
        }
        if (rows_.size() > required_)
            rows_.pop_front();
        if (rows_.size() != required_ || time - last_publish_ + interval_ * 1e-8 < update_)
            return false;
        if (std::ranges::none_of(rows_, [](const Row &r) { return r.information > 0; }))
            return false;
        last_publish_ = time;
        return true;
    }
    std::size_t required() const {
        return required_;
    }
    double first_time() const {
        return rows_.front().time;
    }
    void snapshot(std::vector<C> &moments, std::vector<double> &information) const {
        if (rows_.size() != required_)
            throw std::logic_error("Incomplete archived coherent interval");
        moments.resize(2 * required_ * profile_size_);
        information.resize(required_);
        for (std::size_t r = 0; r < required_; ++r) {
            information[r] = rows_[r].information;
            for (unsigned moment = 0; moment < 2; ++moment)
                std::copy_n(rows_[r].values.data() + moment * profile_size_, profile_size_,
                            moments.data() + (moment * required_ + r) * profile_size_);
        }
    }
};
} // namespace radar::video
