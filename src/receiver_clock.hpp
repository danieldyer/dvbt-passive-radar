#pragma once
#include "gpu_receiver.hpp"
#include "sample_clock.hpp"

namespace radar {
// The controller, rather than the demodulator, converts channel observations
// into the common ADC coordinates and updates the ensemble clock in order.
inline void observe_timing(SampleClock& clock,std::span<const TimingObservation> observations,
                           std::uint64_t origin,unsigned decimation) {
    if(!decimation)throw std::invalid_argument("Zero clock-observation decimation");
    for(const auto& observation:observations) {
        const double n=double(origin)+observation.sample*decimation;
        switch(observation.kind) {
        case TimingObservation::Kind::reset:clock.reset_observer(observation.channel);break;
        case TimingObservation::Kind::rate:clock.observe_rate(observation.channel,n,observation.value);break;
        case TimingObservation::Kind::offset:clock.observe_offset(observation.channel,n,observation.value*decimation);break;
        }
    }
}
}
