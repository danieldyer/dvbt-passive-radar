#pragma once
#include "fec.hpp"
#include <complex>
#include <cmath>
namespace radar {
// Label bit 0 is the first transmitted QAM bit (I sign), then Q sign,
// then alternating Gray-coded I/Q magnitude bits. 255 denotes an erasure.
template<unsigned Bits>
std::uint8_t hard_qam(std::complex<float> z) {
    static_assert(Bits==2 || Bits==4 || Bits==6);
    if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))return 255;
    constexpr unsigned axis=Bits/2,half=1U<<(axis-1);
    const float scale=std::sqrt(Bits==2 ? 2.F : Bits==4 ? 10.F : 42.F);
    unsigned label=0;
    for(unsigned a=0;a<2;++a) {
        const float x=a ? z.imag() : z.real();
        const auto rank=static_cast<unsigned>(std::clamp(float(half)-std::abs(x)*scale*.5F,0.F,float(half-1)));
        const auto gray=rank^(rank>>1);
        label|=unsigned(x<0)<<a;
        for(unsigned j=0;j<axis-1;++j)label|=((gray>>(axis-2-j))&1U)<<(2+2*j+a);
    }
    return static_cast<std::uint8_t>(label);
}
struct Symbol;struct Tps;struct OfdmBatch;class Modem;
// Elapsed seconds in CPU-only work; no GPU submission or wait is included.
struct FecTimings {double demap{},viterbi{},reed_solomon{};};
struct FecBatch {
    // Borrowed until the next decode; corrected RS words retain their coordinates.
    std::span<const fec::RsResult> packets;
    std::size_t count{},trellis_bits{};
};
class CpuFec {
    fec::Viterbi viterbi_;
    std::vector<std::uint8_t> bits_;
    std::vector<std::uint32_t> indices_;
    std::vector<fec::RsResult> packets_;
    int constellation_=-1;
    void configure(std::size_t count,const Tps&,const Modem&);
    void demap(std::span<const std::complex<float>>,unsigned modulo4,unsigned bits,std::size_t symbol);
    FecBatch finish(std::span<const std::uint8_t>,const Tps&,FecTimings*);
public:
    FecBatch decode(std::span<const Symbol>,const Tps&,const Modem&,FecTimings* =nullptr);
    // OFDM submissions must have completed and the buffers must be host coherent.
    FecBatch decode(const OfdmBatch&,const Tps&,const Modem&,FecTimings* =nullptr);
};
}
