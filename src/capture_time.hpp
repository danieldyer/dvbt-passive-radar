#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace radar {
struct DeviceTimeAnchor {
    double device_seconds{};
    std::int64_t host_utc_ns{},host_monotonic_ns{};
    std::uint64_t query_uncertainty_ns{};
};
// The bracket bounds the query correspondence, not the host's absolute UTC
// error. Preserve any host-clock step in the reported uncertainty.
inline DeviceTimeAnchor device_time_anchor(double device,std::int64_t utc_before,std::int64_t utc_after,
                                           std::int64_t monotonic_before,std::int64_t monotonic_after){
    if(!std::isfinite(device) || monotonic_after<monotonic_before)throw std::invalid_argument("Invalid device time bracket");
    const auto duration=monotonic_after-monotonic_before;
    const auto step=std::abs((utc_after-utc_before)-duration);
    return {device,utc_before+(utc_after-utc_before)/2,monotonic_before+duration/2,std::uint64_t((duration+step)/2+1)};
}
template<class Read> DeviceTimeAnchor measure_device_time(Read&& read){
    using namespace std::chrono;
    auto ns=[](auto time){return duration_cast<nanoseconds>(time.time_since_epoch()).count();};
    const auto m0=ns(steady_clock::now()),u0=ns(system_clock::now());
    const double device=read();
    const auto u1=ns(system_clock::now()),m1=ns(steady_clock::now());
    return device_time_anchor(device,u0,u1,m0,m1);
}
}
