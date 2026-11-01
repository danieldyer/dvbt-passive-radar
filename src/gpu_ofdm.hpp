#pragma once
#include "gpu_buffer.hpp"
#include "gpu_timings.hpp"
#include "modem.hpp"

namespace radar::gpu {
class Commands;
}
namespace radar {
struct OfdmWindow {
    std::uint32_t input_offset{};
    std::uint64_t sample{};         // common corrected channel grid, never reset per batch
    double phase{}, frequency_hz{}; // exp(j*(phase - 2*pi*frequency*j/8e6))
};
enum class CodedMetric { hard_bits, likelihood_plus_7, minimax_likelihood };
struct OfdmBatch {
    gpu::BufferView measured, equalized, pilots;
    std::size_t count{}, pilot_stride{};
    gpu::BufferView coded_bits{}; // Optional bytes in convolutional-code order.
    CodedMetric coded_metric =
        CodedMetric::hard_bits; // Hard: 0/1/erasure=2; soft: 0..14, erasure=7; encoding is explicit.
    // Native acquisition FFTs have stride 8192 and explicit post-FFT phases.
    // The frame ring gathers/rotates these once into canonical 6817-carrier rows.
    std::size_t carrier_stride = 6817;
    gpu::BufferView phase_corrections{};
};
struct CpCorrelation {
    C correlation{};
    float energy{}, quality{};
};
struct IntegerFrequency {
    int bins{};
    float quality{};
};
// Batched, stateless numerical stages. Small acquisition/clock-loop control
// uses their reduced outputs; waveform and carrier arrays stay on the device.
class GpuOfdm {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    GpuOfdm(const PilotPattern &pilots, unsigned capacity, std::shared_ptr<gpu::Context> context = {});
    ~GpuOfdm();
    std::shared_ptr<gpu::Context> context() const;
    void wait();
    OfdmBatch transform(const gpu::BufferView &input, std::span<const OfdmWindow> windows,
                        ComputeTimings *timings = nullptr, gpu::Commands *commands = nullptr);
    OfdmBatch transform(std::span<const C> input, std::span<const OfdmWindow> windows,
                        ComputeTimings *timings = nullptr);
    std::vector<CpCorrelation> correlate(const gpu::BufferView &input, std::span<const std::uint32_t> offsets,
                                         unsigned guard, ComputeTimings *timings = nullptr);
    std::vector<CpCorrelation> correlate(std::span<const C> input, std::span<const std::uint32_t> offsets,
                                         unsigned guard, ComputeTimings *timings = nullptr);
    std::vector<IntegerFrequency> integer_frequency(unsigned radius = 128, ComputeTimings *timings = nullptr);
    // Compose an explicitly exported phase per window. Its application is fused
    // into the next canonical carrier gather; no full-spectrum pass is needed.
    // Payload equalization occurs once, later in the complete frame batch.
    void rotate(std::span<const double> phases);
};
} // namespace radar
