#pragma once
#include "gpu_buffer.hpp"
#include "gpu_timings.hpp"
#include "modem.hpp"

namespace radar::gpu {class Commands;}
namespace radar {
struct OfdmWindow {
    std::uint32_t input_offset{};
    std::uint64_t sample{}; // common corrected channel grid, never reset per batch
    double phase{},frequency_hz{}; // exp(j*(phase - 2*pi*frequency*j/8e6))
};
struct OfdmBatch {
    gpu::BufferView measured,equalized,pilots;
    std::size_t count{},pilot_stride{};
};
struct CpCorrelation {C correlation{};float energy{},quality{};};
struct IntegerFrequency {int bins{};float quality{};};
// Batched, stateless numerical stages. Small acquisition/clock-loop control
// uses their reduced outputs; waveform and carrier arrays stay on the device.
class GpuOfdm {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    GpuOfdm(const PilotPattern& pilots,unsigned capacity,std::shared_ptr<gpu::Context> context={});
    ~GpuOfdm();
    std::shared_ptr<gpu::Context> context() const;
    void wait();
    OfdmBatch transform(const gpu::BufferView& input,std::span<const OfdmWindow> windows,ComputeTimings* timings=nullptr,gpu::Commands* commands=nullptr);
    OfdmBatch transform(std::span<const C> input,std::span<const OfdmWindow> windows,ComputeTimings* timings=nullptr);
    std::vector<CpCorrelation> correlate(const gpu::BufferView& input,std::span<const std::uint32_t> offsets,
                                        unsigned guard,ComputeTimings* timings=nullptr);
    std::vector<CpCorrelation> correlate(std::span<const C> input,std::span<const std::uint32_t> offsets,
                                        unsigned guard,ComputeTimings* timings=nullptr);
    std::vector<IntegerFrequency> integer_frequency(unsigned radius=128,ComputeTimings* timings=nullptr);
    // Applies an explicitly exported constant phase per window. Equalization
    // already removed channel phase on the decoder branch; only Y and pilots rotate.
    void rotate(std::span<const double> phases,ComputeTimings* timings=nullptr,gpu::Commands* commands=nullptr);
    // Explicit host materialization for the CPU reference/qualification path.
    std::vector<Symbol> download(const OfdmBatch& batch,std::span<const OfdmWindow> windows) const;
};
}
