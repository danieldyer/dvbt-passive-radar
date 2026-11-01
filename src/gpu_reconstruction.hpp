#pragma once
#include "cpu_fec.hpp"
#include "gpu_buffer.hpp"
#include "gpu_timings.hpp"
#include "modem.hpp"

namespace radar::gpu {
class Commands;
}
namespace radar {
struct ReferenceBatch {
    // Canonical transmitted active carriers; no receive-window phase or channel
    // response is hidden here. Validity is uint32 per carrier on the device.
    gpu::BufferView carriers, valid;
    std::size_t count{};
};
class GpuReconstruction {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    explicit GpuReconstruction(std::shared_ptr<gpu::Context> context);
    ~GpuReconstruction();
    // Whole BCH-valid frames, starting at symbol zero. Corrected RS words retain
    // their decoded stream positions, including unavailable interleaver history.
    ReferenceBatch reconstruct(const FecBatch &packets, std::span<const Tps> frames, const Modem &tables,
                               ComputeTimings *timings = nullptr, gpu::Commands *commands = nullptr);
    Reconstruction download(const ReferenceBatch &batch) const;
};
} // namespace radar
