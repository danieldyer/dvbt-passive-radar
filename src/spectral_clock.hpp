#pragma once
#include "gpu_buffer.hpp"
#include "gpu_timings.hpp"
#include "dsp.hpp"
#include "sample_clock.hpp"

namespace radar {
// Common-clock channelisation: an existing forward FFT feeds twice-oversampled
// channel IFFTs. One ADC time map drives every output interpolation and NCO.
// Affine segments commute through the filter using H(rate*f - centre); windows
// crossing segment boundaries are warped before their shared forward FFT.
class SpectralClock {
public:
    SpectralClock(std::uint32_t rate,std::span<const double> centres,
                  std::span<const float> prototype,std::shared_ptr<gpu::Context> context,
                  std::uint32_t source_rate=0);
    ~SpectralClock();
    void process(std::span<const C> input,std::uint64_t input_start,
                 std::uint64_t first,std::size_t count,std::span<const ClockSegment> clock,
                 ComputeTimings* timings=nullptr);
    // Borrowed resident storage, with complete FFT padding after valid_samples.
    // Upstream writes must be submitted on this context's queue. Synchronous
    // completion protects input reuse; no host mapping or waveform copy is needed.
    void process(const gpu::BufferView& input,std::size_t valid_samples,std::uint64_t input_start,
                 std::uint64_t first,std::size_t count,std::span<const ClockSegment> clock,
                 ComputeTimings* timings=nullptr);
    std::size_t input_capacity() const;
    C* input_data(std::size_t count);
    gpu::BufferView output_buffer() const;
    std::size_t capacity() const;
    unsigned support() const;
    unsigned raw_left() const;
    unsigned raw_right() const;
    unsigned interpolation_taps() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
