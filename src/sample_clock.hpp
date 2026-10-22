#pragma once
#include "clock_estimator.hpp"
#include <cstdint>
#include <deque>
#include <mutex>
#include <limits>
#include <span>
#include <vector>

namespace radar {
// u(n) = n + offset + (n-start)*(rate-1). n labels the corrected ADC grid;
// u labels the original ADC grid. Published segments meet continuously.
// Channel NCOs use n, never u.
struct ClockSegment {
    std::uint64_t start{};
    double offset{},rate=1;
    double offset_at(double n) const {return offset+(n-double(start))*(rate-1);}
};
struct ClockObserverStatus {
    // active reports a recent measurement, not membership in the reference.
    bool active{},anchored{};
    std::size_t observations{};
    double estimated_ppm{},span_seconds{},fit_rms_samples{};
    double measurement_sigma_samples=std::numeric_limits<double>::quiet_NaN();
    double innovation_samples{},innovation_variance{},normalized_innovation{};
    std::uint64_t downweighted{},late{};
};
// Configurable design priors, not measured TCXO specifications. The stochastic
// conventions are those of ClockNoise. Membership never changes the reference.
struct ClockControlSettings {
    ClockNoise receiver{1e-10,1e-10,1e-22},transmitter{1e-11,1e-11,0};
    double settling_seconds=.25,innovation_huber_sigma=6,unknown_sigma_samples=.01;
};
struct ClockObservation {
    enum class Kind {rate,offset,reset};
    Kind kind;std::size_t channel;
    double sample{},value{},variance{},feedback_removed{},applied_offset{},applied_rate=1;
    ClockInnovation innovation{};
    bool late=false;
};
class SampleClock {
public:
    SampleClock(double input_rate,std::span<const double> centres,double phase_rms_degrees=3,ClockControlSettings={});
    ClockSegment at(double n) const;
    std::vector<ClockSegment> segments() const;
    std::vector<ClockObserverStatus> status(double corrected_sample) const;
    // Publish only beyond every input sample already used, including FIR halos.
    void update(std::uint64_t first_unprocessed);
    void schedule(std::uint64_t start,double rate); // deterministic test input
    // These observe one common ADC grid. Static path/anchor delays are nuisance
    // offsets. Independent TX phase/rate contrasts have zero sum over the fixed
    // declared ensemble, even while some channels are unavailable.
    void observe_rate(std::size_t channel,double n,double relative_error);
    void observe_offset(std::size_t channel,double n,double timing_error_adc,
                        double variance_adc=std::numeric_limits<double>::quiet_NaN());
    void reset_observer(std::size_t channel);
    void enable_diagnostics(bool enabled=true);
    std::vector<ClockObservation> take_observations();
    ClockEstimate estimate(double corrected_sample) const;
    double phase_budget_degrees() const {return degrees_;}
    double timing_budget_samples() const {return budget_;}
private:
    struct Point {double n,y,variance;};
    struct Observer {
        std::deque<Point> points;
        double rate=1,rate_time=-1,bias=0,fit_rms=0;
        bool anchored=false;
        ClockInnovation innovation;
        std::uint64_t downweighted{},late{};
    };
    ClockSegment at_unlocked(double n) const;
    void schedule_unlocked(std::uint64_t start,double rate);
    double fs_,degrees_,budget_;
    ClockControlSettings settings_;
    ClockEstimator estimator_;
    mutable std::mutex mutex_;
    std::vector<ClockSegment> segments_{{}};
    std::vector<Observer> observers_;
    std::vector<ClockObservation> pending_,diagnostics_;
    bool diagnostics_enabled_=false;
};
}
