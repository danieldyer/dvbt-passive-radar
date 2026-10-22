#pragma once
#include "sar_handoff.hpp"
#include "dsp.hpp"
#include <string>
#include <string_view>
#include <optional>

namespace radar {
// Descriptive geometry is retained verbatim for the later image formation
// consumer. Acquisition only needs declared RF, NCO and guard interval.
struct RecordingSignal {
    std::string id,name,transmitter;
    SarChannel channel;
};
struct SdrCaptureSettings {
    std::string driver="uhd";
    std::string device_args="type=b200",antenna="RX2",clock_source="internal",time_source="internal";
    unsigned channel=0;
    double sample_rate_hz{},centre_hz{},gain_db=30,bandwidth_hz=56e6,buffer_seconds=4;
    ClockControlSettings clock;
};
class RecordingMetadata {
    std::string json_;
public:
    double recording_rf_hz{};
    std::uint32_t sample_rate{};
    std::string sample_format;
    std::vector<RecordingSignal> signals;
    std::optional<SdrCaptureSettings> capture;
    static RecordingMetadata read(const std::filesystem::path&);
    void validate_wave(const WaveInfo&) const;
    void configure(SarCapture&,std::string_view signal={}) const;
    void write(const std::filesystem::path&,std::string_view signal={}) const;
    std::string_view json() const {return json_;}
};
}
