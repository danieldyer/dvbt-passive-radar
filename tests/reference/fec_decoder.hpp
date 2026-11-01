#pragma once
#include "cpu_fec.hpp"
#include "gpu_ofdm.hpp"
#include "modem.hpp"
#include "viterbi.hpp"
#include <cmath>
#include <complex>
namespace radar {
template <unsigned Bits> std::uint8_t hard_qam(std::complex<float> z) {
    static_assert(Bits == 2 || Bits == 4 || Bits == 6);
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        return 255;
    constexpr unsigned axis = Bits / 2, half = 1U << (axis - 1);
    const float scale = std::sqrt(Bits == 2 ? 2.F : Bits == 4 ? 10.F : 42.F);
    unsigned label = 0;
    for (unsigned a = 0; a < 2; ++a) {
        const float x = a ? z.imag() : z.real();
        const auto rank =
            static_cast<unsigned>(std::clamp(float(half) - std::abs(x) * scale * .5F, 0.F, float(half - 1)));
        const auto gray = rank ^ (rank >> 1);
        label |= unsigned(x < 0) << a;
        for (unsigned j = 0; j < axis - 1; ++j)
            label |= ((gray >> (axis - 2 - j)) & 1U) << (2 + 2 * j + a);
    }
    return static_cast<std::uint8_t>(label);
}

namespace validation {
// Input adapters and classical hard/uniform decoding are test references only.
// Compact likelihoods always exercise the actual production decoder.
class FecDecoder {
    CpuFec production_;
    ReferenceViterbi viterbi_;
    std::vector<std::uint8_t> bits_;
    std::vector<std::uint32_t> indices_;
    std::vector<fec::RsResult> packets_;
    int constellation_ = -1;
    void configure(std::size_t, const Tps &, const Modem &);
    void demap(std::span<const C>, unsigned, unsigned, std::size_t);
    FecBatch finish(std::span<const std::uint8_t>, const Tps &, FecTimings *, std::size_t, CodedMetric);

  public:
    FecBatch decode(std::span<const Symbol>, const Tps &, const Modem &, FecTimings * = nullptr);
    FecBatch decode(const OfdmBatch &, const Tps &, const Modem &, FecTimings * = nullptr, std::size_t = 0);
};
} // namespace validation
} // namespace radar
