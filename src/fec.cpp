#include "fec.hpp"
namespace radar::fec {
namespace {
constexpr auto costs=[]() consteval {
    std::array<std::array<std::uint8_t,32>,16> table{};
    for(unsigned mask=0;mask<4;++mask)for(unsigned observed=0;observed<4;++observed)
        for(unsigned p=0;p<32;++p)table[mask*4+observed][p]=std::popcount((DvbtCode::branches[2*p]^observed)&mask);
    return table;
}();
struct Metrics {
    std::array<std::uint8_t,64> metric{};
    // Inlining also keeps the metric array from escaping through the tail path.
    [[gnu::always_inline]] std::uint64_t advance(const std::uint8_t* cost,unsigned weight) {
        std::array<std::uint8_t,64> next;
        std::array<std::uint8_t,32> decisions;
        for(unsigned p=0;p<32;++p) {
            const auto branch=cost[p];
            const auto complement=static_cast<std::uint8_t>(weight-branch);
            const auto lo0=static_cast<std::uint8_t>(metric[p]+branch);
            const auto hi0=static_cast<std::uint8_t>(metric[p+32]+complement);
            const auto lo1=static_cast<std::uint8_t>(metric[p]+complement);
            const auto hi1=static_cast<std::uint8_t>(metric[p+32]+branch);
            next[2*p]=std::min(lo0,hi0);next[2*p+1]=std::min(lo1,hi1);
            decisions[p]=(hi0<lo0 ? 1U<<(2*p%8) : 0) | (hi1<lo1 ? 2U<<(2*p%8) : 0);
        }
        // Each byte contains two survivor bits in disjoint positions. Two
        // pairwise sums pack four butterflies into each output byte. Keep these
        // short loops vectorized: eager unrolling otherwise scalarizes packing.
        std::array<std::uint8_t,16> pairs;
#if defined(__clang__)
#pragma clang loop vectorize(enable) unroll(disable)
#endif
        for(unsigned i=0;i<16;++i)pairs[i]=decisions[2*i]+decisions[2*i+1];
        std::array<std::uint8_t,8> packed;
#if defined(__clang__)
#pragma clang loop vectorize(enable) unroll(disable)
#endif
        for(unsigned i=0;i<8;++i)packed[i]=pairs[2*i]+pairs[2*i+1];
        auto result=std::bit_cast<std::uint64_t>(packed);
        if constexpr(std::endian::native==std::endian::big)result=std::byteswap(result);
        metric=next;return result;
    }
    [[gnu::always_inline]] void normalize() {
        std::uint8_t minimum=255;for(auto value:metric)minimum=std::min(minimum,value);
        for(auto& value:metric)value-=minimum;
    }
    std::array<std::uint8_t,64> values() const {return metric;}
};

}

template<unsigned Rate>
std::vector<std::uint8_t> Viterbi::decode_rate(std::span<const std::uint8_t> bits) {
    constexpr auto pattern=punctures[Rate];
    constexpr unsigned period=puncture_lengths[Rate]/2;
    constexpr unsigned sent=[] {unsigned n=0;for(auto bit:punctures[Rate])n+=bit;return n;}();
    if(bits.size()%sent)throw std::invalid_argument("Incomplete puncture cycle");
    const auto count=bits.size()/sent*period;
    survivors_.resize(count);
    Metrics metrics;std::size_t at=0,t=0;
    // Keep both template layers in the time loop. Clang otherwise outlines a
    // puncture cycle and reloads captured metrics/counters on every cycle.
    auto advance=[&]<std::size_t Phase> [[gnu::always_inline]] () {
        unsigned observed=0,mask=0;
        if constexpr(pattern[2*Phase]){const auto bit=bits[at++];if(bit<2){observed=bit;mask=1;}}
        if constexpr(pattern[2*Phase+1]){const auto bit=bits[at++];if(bit<2){observed|=bit<<1;mask|=2;}}
        survivors_[t++]=metrics.advance(costs[mask*4+observed].data(),(mask&1)+(mask>>1));
    };
    // Normalize outside the puncture-cycle loop. In at most 96 steps the
    // common metric can grow by 192; the six-step spread is at most 12.
    // Thus every byte metric is <=204, including arbitrary erasures/errors.
    constexpr unsigned block=period*(96/period);
    while(count-t>=block) {
        const auto end=t+block;
        while(t<end)[&]<std::size_t... P> [[gnu::always_inline]] (std::index_sequence<P...>){(advance.template operator()<P>(),...);}(std::make_index_sequence<period>{});
        metrics.normalize();
    }
    while(at<bits.size()) {
        [&]<std::size_t... P> [[gnu::always_inline]] (std::index_sequence<P...>){(advance.template operator()<P>(),...);}(std::make_index_sequence<period>{});
        metrics.normalize();
    }

    const auto final=metrics.values();
    unsigned state=static_cast<unsigned>(std::min_element(final.begin(),final.end())-final.begin());
    std::vector<std::uint8_t> result(count/8);
    auto predecessor=[&]{--t;const auto bit=state&1U;state=(state>>1)|(((survivors_[t]>>state)&1U)<<5);return bit;};
    while(t%8)predecessor(); // Partial terminal byte still contributes to the path.
    while(t) {
        const auto byte=t/8-1;
        result[byte]=[&]<std::size_t... I>(std::index_sequence<I...>){
            std::uint8_t value=0;((value|=predecessor()<<I),...);return value;
        }(std::make_index_sequence<8>{});
    }
    return result;
}
std::vector<std::uint8_t> Viterbi::decode(std::span<const std::uint8_t> bits,unsigned rate) {
    switch(rate) {
    case 0:return decode_rate<0>(bits);
    case 1:return decode_rate<1>(bits);
    case 2:return decode_rate<2>(bits);
    case 3:return decode_rate<3>(bits);
    case 4:return decode_rate<4>(bits);
    default:throw std::invalid_argument("Invalid DVB-T code rate");
    }
}
}
