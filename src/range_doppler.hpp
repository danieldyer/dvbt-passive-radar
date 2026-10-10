#pragma once
#include "dsp.hpp"
#include <deque>
#include <future>
#include <optional>
#include <ostream>

namespace radar {
struct MapOptions {
    double range_metres=30000, doppler_hz=100, history_seconds=300;
    double integration_seconds=0, update_seconds=2, coherence_degrees=5;
    unsigned slow_decimation=4;
    bool remove_dc=false;
    // Optional lossless, length-framed RDM output. A slow consumer applies
    // backpressure instead of dropping maps; lifetime must include finish().
    std::ostream* stream=nullptr;
};
struct RangeProfile {
    // Immutable profiles are shared with the background transform. Publishing a
    // long CPI copies descriptors, not hundreds of megabytes of sample storage.
    std::shared_ptr<const std::vector<C>> values;
    double time{}, phase{}, timing{};
};
struct DopplerMap {
    std::vector<C> values; // range-major, ascending signed Doppler
    std::size_t ranges{}, frequencies{}, samples{},fft_size{};
    bool dc_removed=false;
    std::uint64_t sequence{},segment{};
    double range_step=299792458.0/8e6, frequency_start{},frequency_step{};
    double time_start{},time_end{},integration{},coherence{};
    double history_limit{},requested_integration{};
    std::string window="hann",dc_method="hann_weighted_mean",motion_model="stationary_range";
    double enbw{},phase_epoch{};
    double rf_hz{},reference_origin_samples{};
    unsigned channel{};
    std::string stationary_prediction="none";
    double background_start{},background_end{};
    std::string amplitude_units="cross_correlation";
    std::vector<float> noise_range,noise_doppler;
    std::string json() const;
    void write(std::ostream& stream) const;
    void write(const std::filesystem::path& path) const;
};
// A longest-suffix coherence measurement at both occupied-band edges, with no
// phase correction applied to the radar data. A bounded history limits memory.
std::size_t coherent_suffix(std::span<const RangeProfile> rows,double threshold,double& coherence);
DopplerMap make_doppler_map(std::span<const RangeProfile> rows,double interval,double limit,bool remove_dc=false);
struct RangeTimings { double inverse{}, filter{}; };
class RangeDoppler {
public:
    RangeDoppler(MapOptions options,double symbol_interval,std::filesystem::path output);
    ~RangeDoppler();
    void append(std::span<const C> cross,double time,double phase,double timing,bool valid,std::uint64_t segment,
                RangeTimings* timings=nullptr);
    void finish();
    std::size_t published() const {return sequence_;}
private:
    void publish(bool final);
    MapOptions options_;
    double interval_,last_publish_=-1e30;
    std::filesystem::path output_;
    Fft range_fft_{8192};
    std::vector<C> fft_buffer_,filter_ring_;
    std::vector<float> taps_;
    std::deque<RangeProfile> history_;
    std::deque<RangeProfile> diagnostics_;
    std::size_t ranges_{},input_rows_{},sequence_{};
    std::uint64_t segment_=UINT64_MAX;
    std::future<void> pending_;
};
}
