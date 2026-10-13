#pragma once
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <vector>

namespace radar {
// u(n) = n + offset + (n-start)*(rate-1). n is a corrected ADC-grid
// coordinate; u is the original recording coordinate. Segments are immutable
// once published, and meet continuously. Channel NCOs use n, never u.
struct ClockSegment {
    std::uint64_t start{};
    double offset{},rate=1;
    double offset_at(double n) const {return offset+(n-double(start))*(rate-1);}
};
struct ClockObserverStatus {
    bool active{},anchored{};
    std::size_t observations{};
    double estimated_ppm{},span_seconds{},fit_rms_samples{};
};
class SampleClock {
public:
    SampleClock(double input_rate,std::span<const double> centres,double phase_rms_degrees=3);
    ClockSegment at(double n) const;
    std::vector<ClockSegment> segments() const;
    std::vector<ClockObserverStatus> status(double corrected_sample) const;
    // Publish only beyond every input sample already used, including FIR halos.
    void update(std::uint64_t first_unprocessed);
    void schedule(std::uint64_t start,double rate); // deterministic calibration/test input
    // Differential and anchored pilot timing on the corrected grid. Static
    // propagation delays are per-channel nuisance offsets, not clock offsets.
    // The reference is the arithmetic ensemble of active transmitter clocks;
    // changing pilot precision must not redefine its phase epoch. Independent
    // transmitter clock offsets/rates remain observable relative to this grid.
    void observe_rate(std::size_t channel,double n,double relative_error);
    void observe_offset(std::size_t channel,double n,double timing_error_adc);
    void reset_observer(std::size_t channel);
    double phase_budget_degrees() const {return degrees_;}
    double timing_budget_samples() const {return budget_;}
private:
    struct Point {double n,y;};
    struct Observer {
        std::deque<Point> points;
        double rate=1,rate_time=-1,bias=0,fit_rms=0;
        bool anchored=false;
    };
    ClockSegment at_unlocked(double n) const;
    void schedule_unlocked(std::uint64_t start,double rate);
    double fs_,degrees_,budget_;
    bool reference_ready_=false;
    double reference_epoch_=0,reference_offset_=0,reference_velocity_=0,reference_acceleration_=0;
    mutable std::mutex mutex_;
    std::vector<ClockSegment> segments_{{}};
    std::vector<Observer> observers_;
};
}
