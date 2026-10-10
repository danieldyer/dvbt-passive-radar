#pragma once
#include "gpu_timings.hpp"
#include "dsp.hpp"
#include "sample_clock.hpp"
#include "channel_batch.hpp"
#include <array>

namespace radar {
using FrontendTimings=ComputeTimings;
// A common ADC clock/origin for all channels. NCO phase is referenced to input
// sample zero, including after a seek. Output centres are origin + m*decimation.
class GpuChannelBank {
public:
    static constexpr std::size_t batch_size=262144;
    GpuChannelBank(WaveReader& input, std::span<const double> offsets,
                   double start_seconds, double seconds);
    GpuChannelBank(std::uint32_t input_rate,std::span<const double> offsets);
    ~GpuChannelBank();
    std::vector<std::vector<C>> next(FrontendTimings* timings=nullptr);
    ChannelBatch next_device(FrontendTimings* timings=nullptr);
    gpu::BufferView output_buffer() const;
    std::shared_ptr<gpu::Context> context() const;
    // Input includes the complete centred FIR support. Supports live/in-memory
    // streams and isolated kernel tests without a file or playback machinery.
    std::vector<std::vector<C>> process(std::span<const C> input,std::uint64_t first_center,
                                      std::size_t count,FrontendTimings* timings=nullptr);
    void set_clock(std::shared_ptr<SampleClock> clock);
    // Input coordinates are ORIGINAL ADC coordinates. The output centres are
    // on the CORRECTED grid. Every channel uses the same time map, including
    // the centre-frequency phase correction when retiming is fused with the FFT.
    std::vector<std::vector<C>> process_clocked(std::span<const C> input,std::uint64_t input_start,
        std::uint64_t first_center,std::size_t count,FrontendTimings* timings=nullptr);
    ChannelBatch process_clocked_device(std::span<const C> input,std::uint64_t input_start,
        std::uint64_t first_center,std::size_t count,FrontendTimings* timings=nullptr);
    // Resident input uses the same synthesis and common clock. Storage must
    // contain input_capacity() samples, including zero padding after valid_samples.
    // The source owner retains it until this synchronous call completes.
    ChannelBatch process_clocked_device(const gpu::BufferView& input,std::size_t valid_samples,
        std::uint64_t input_start,std::uint64_t first_center,std::size_t count,FrontendTimings* timings=nullptr);
    std::size_t input_capacity() const;
    unsigned interpolation_taps() const;
    std::size_t capacity() const;
    std::uint64_t origin{}, output_count{}, produced{};
    std::uint32_t decimation{};
    std::size_t taps() const;
    static std::vector<float> prototype(double sample_rate);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Centred, finite-support interpolation. No periodic OFDM assumption: echoes
// crossing a guard interval retain their actual samples. Output t = first + m*step.
// first is relative to input[0], including the 47 samples of left support.
class GpuResampler {
public:
    GpuResampler();
    ~GpuResampler();
    std::vector<C> process(std::span<const C> input, double first, double step, std::size_t count,FrontendTimings* timings=nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
