#pragma once
#include "dsp.hpp"
#include <numbers>

// Deliberately scalar, double-accumulating oracle for the GPU convolution.
// The input starts half a prototype before first_center. Each output evaluates
// the centred FIR at its physical ADC coordinate, including the NCO at EACH tap.
// No GPU coefficient modulation, oscillator table or tiled indexing is reused.
inline std::vector<radar::C> channel_reference(std::span<const radar::C> input,
    std::span<const float> taps,double frequency,double fs,std::uint64_t first_center,
    unsigned decimation,std::size_t count) {
    std::vector<radar::C> result(count);
    for(std::size_t m=0;m<count;++m) {
        std::complex<double> value{};
        for(std::size_t j=0;j<taps.size();++j) {
            const double n=double(first_center)+double(m*decimation+j)-double(taps.size()/2);
            value+=std::complex<double>(input[m*decimation+j])*double(taps[j])*
                std::polar(1.0,-2*std::numbers::pi*std::remainder(n*frequency/fs,1.0));
        }
        result[m]=static_cast<radar::C>(value);
    }
    return result;
}
