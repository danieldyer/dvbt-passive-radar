#pragma once
#include "matched_profiles.hpp"
#include "range_doppler.hpp"

namespace radar {
struct FocusOptions {
    double range_metres=30000, doppler_hz=200;
    bool remove_dc=true;
};
// Constant bistatic path rate, tied to physical transmitter RF. The phase epoch
// is the middle useful-symbol sample of the CPI; receiver CFO and per-channel
// retiming do not enter these coordinates.
class GpuFocus {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuFocus(std::shared_ptr<gpu::Context>);
    ~GpuFocus();
    void prepare(std::size_t rows);
    // Positive information weights undo per-batch amplitude normalization.
    // An empty span gives equal weights; no temporal taper is imposed.
    DopplerMap calculate(std::span<const C> moments,std::size_t rows,double interval,
                         double rf_hz,double first_window_time,FocusOptions={},ComputeTimings* =nullptr,
                         std::span<const double> information={});
};
}
