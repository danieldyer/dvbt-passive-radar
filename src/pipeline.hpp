#pragma once
#include "modem.hpp"
#include "range_doppler.hpp"
#include <functional>

namespace radar {
struct PipelineOptions {
    std::filesystem::path input, output;
    double offset_hz=4.625e6, start_seconds=0, seconds=0;
    std::vector<double> offsets_hz;
    int guard=16;
    unsigned response_order=8;
    double reference_delay_samples=0;
    bool gpu=true, summary_only=true, automatic_origin=true, maps=true;
    bool gpu_frontend=true;
    double clock_phase_budget_degrees=3;
    MapOptions map;
};
int process_recording(const PipelineOptions& options);
// Bounded in-memory entry point for independent sample/phase-coordinate qualification.
class SampleClock;
void synchronize_samples(std::span<const C> input,int guard,const std::function<void(Symbol)>& receive,
                         std::shared_ptr<SampleClock> clock={},std::uint64_t origin=0,unsigned decimation=1);
}
