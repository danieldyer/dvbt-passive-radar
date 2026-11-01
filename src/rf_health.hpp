#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace radar {
// Counts I/Q pairs whose largest component reaches the declared thresholds.
// These detect digital rails, not overload earlier in the analogue RF chain.
struct RfHealth {
    std::uint64_t samples{}, rail_samples{}, near_rail_samples{};
    double power_sum{}, peak_component{};
    void add(const RfHealth &other) {
        samples += other.samples;
        rail_samples += other.rail_samples;
        near_rail_samples += other.near_rail_samples;
        power_sum += other.power_sum;
        peak_component = std::max(peak_component, other.peak_component);
    }
    double rms_dbfs() const {
        return samples && power_sum > 0 ? 10 * std::log10(power_sum / samples)
                                        : -std::numeric_limits<double>::infinity();
    }
    double peak_dbfs() const {
        return peak_component > 0 ? 20 * std::log10(peak_component)
                                  : -std::numeric_limits<double>::infinity();
    }
};
} // namespace radar
