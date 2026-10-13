#pragma once
#include "gpu_reconstruction.hpp"

namespace radar {
struct ResponseStatistics {C cross;float reference_power,measured_power;};
struct ResponseEstimate {
    gpu::BufferView response,statistics;
    std::size_t count{};
    double reference_origin{},relative_error{};
    gpu::BufferView source_offsets; // Per-row source timing, on the common receive grid.
    gpu::BufferView innovation_variance; // Stable source-conditioned scatter; next-batch pilot weighting.
};
struct ResidualBatch {
    gpu::BufferView reference,cross,normalized,response,valid; // cross/normalized may be omitted.
    std::size_t count{};
    double reference_origin{};
};
// Estimation has no delay/all-pass parameter: magnitude determines the causal
// cepstrum, with one scalar phase alignment. Reference origin is an explicit
// input coordinate, never an extra fitted response coefficient. Per-row source
// offsets align independently clocked transmitters on the common receive grid.
// Returned buffers are borrowed until the next corresponding fit/apply call.
class GpuResponse {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpuResponse(std::shared_ptr<gpu::Context> context);
    ~GpuResponse();
    ResponseEstimate fit(const gpu::BufferView& measured,const ReferenceBatch& reference,
                         double reference_origin,unsigned order,unsigned iterations=16,ComputeTimings* timings=nullptr,gpu::Commands* commands=nullptr,
                         std::span<const double> source_offsets={});
    ResidualBatch apply(const gpu::BufferView& measured,const ReferenceBatch& reference,
                        const ResponseEstimate& estimate,ComputeTimings* timings=nullptr,gpu::Commands* commands=nullptr,
                        bool residual_spectra=true);
    // Validate and read diagnostics only after the supplied commands complete.
    static void read_diagnostics(ResponseEstimate&);
    // Optional, once-per-acquisition coordinate calibration. The caller owns
    // and exports this fixed adjustment independently of the minimum-phase fit.
    static double direct_origin_adjustment(const ResponseEstimate& estimate);
};
}
