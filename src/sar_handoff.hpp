#pragma once
#include "gpu_pipeline.hpp"
#include "sample_clock.hpp"
#include <filesystem>
#include <fstream>

namespace radar {
// RF centre is a declared physical frequency, NOT recording RF + NCO + the
// receiver's tracked CFO. The latter includes the unknown receiver LO error.
struct SarChannel {
    double rf_hz{},nco_hz{};
    unsigned guard=16;
};
struct SarCapture {
    std::uint32_t input_rate{},decimation{};
    std::uint64_t origin{};
    double recording_rf_hz{},clock_phase_degrees=3;
    std::vector<SarChannel> channels;
    std::uint32_t source_rate{}; // Zero means input_rate; input_rate is the corrected coordinate grid.
    void validate() const;
};
// Synchronous, borrowed device handoff. No waveform copies or SAR processing.
// Row indices address the full batch; only [frame.first,frame.last) are new.
// Surrounding rows are reference/FEC context and must not be counted twice.
struct SarBatch {
    const SarCapture& capture;
    const SpectralBatch& spectra;
    std::vector<ClockSegment> clock;
    SarBatch(const SarCapture&,const SpectralBatch&,std::vector<ClockSegment>);
    std::uint64_t corrected_adc(std::size_t row) const;
    double seconds(std::size_t row) const;
    double original_adc(std::size_t row,double within_window=0) const;
    double carrier_rf_hz(unsigned carrier) const;
};
// Optional output boundary, independent of the numerical pipeline. Stores the
// measured spectra, canonical reference, validity and response; cross products
// can be recovered exactly from these, without storing redundant planes.
class SarWriter {
    SarCapture capture_;
    std::ofstream file_;
    std::vector<ClockSegment> clock_;
    std::uint64_t batches_=0;
public:
    SarWriter(const std::filesystem::path&,const SarCapture&);
    void write(const SarBatch&);
    void finish();
};
}
