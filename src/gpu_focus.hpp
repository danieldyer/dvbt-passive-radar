#pragma once
#include "matched_profiles.hpp"
#include "range_doppler.hpp"

namespace radar {
struct FocusOptions {
    double range_metres = 30000, doppler_hz = 200;
    bool remove_dc = true;
};
// Constant bistatic path rate, tied to physical transmitter RF. The phase epoch
// is the middle useful-symbol sample of the CPI; receiver CFO and per-channel
// retiming do not enter these coordinates.
class GpuFocus {
  public:
    struct Result {
        DopplerMap map;
        gpu::BufferView values;
    };

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Result calculate_impl(std::span<const C>, const gpu::BufferView &, std::size_t, double, double, double,
                          FocusOptions, ComputeTimings *, std::span<const double>);

  public:
    explicit GpuFocus(std::shared_ptr<gpu::Context>);
    ~GpuFocus();
    void prepare(std::size_t rows, unsigned profile_size = matched_profile_size);
    // Positive information weights undo per-batch amplitude normalization.
    // An empty span gives equal weights; no temporal taper is imposed.
    DopplerMap calculate(std::span<const C> moments, std::size_t rows, double interval, double rf_hz,
                         double first_window_time, FocusOptions = {}, ComputeTimings * = nullptr,
                         std::span<const double> information = {});
    // Ready borrowed GPU output, valid until the next calculation or destruction.
    // Metadata has no host values; consumers can reduce or archive on the device.
    Result calculate_device(std::span<const C> moments, std::size_t rows, double interval, double rf_hz,
                            double first_window_time, FocusOptions = {}, ComputeTimings * = nullptr,
                            std::span<const double> information = {});
    // Same operator with the per-row forward range FFT already applied. This
    // stage boundary lets archive replay reuse unchanged rows in sliding CPIs.
    Result calculate_spectral_device(const gpu::BufferView &, std::size_t rows, double interval, double rf_hz,
                                     double first_window_time, FocusOptions = {}, ComputeTimings * = nullptr,
                                     std::span<const double> information = {});
};
} // namespace radar
