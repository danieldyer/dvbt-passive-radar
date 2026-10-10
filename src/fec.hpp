#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <utility>
#include <span>
#include <stdexcept>
#include <vector>

namespace radar::fec {
// DVB-T K=7. Registers shift toward the older bits; the incoming bit is bit 0.
template<unsigned K, unsigned G0, unsigned G1>
struct Convolutional {
    static_assert(K>=2 && K<=7 && G0<(1U<<K) && G1<(1U<<K));
    static constexpr unsigned states=1U<<(K-1);
    static constexpr std::uint8_t symbol(unsigned reg) {
        return (std::popcount(reg&G0)&1U) | ((std::popcount(reg&G1)&1U)<<1);
    }
    static constexpr auto branches=[] {
        std::array<std::uint8_t,states> a{};
        for(unsigned s=0;s<states;++s)a[s]=symbol(s);return a;
    }();
};
using DvbtCode=Convolutional<7,0x4f,0x6d>;
inline constexpr std::array<std::array<unsigned,14>,5> punctures{{
    {1,1},{1,1,0,1},{1,1,0,1,1,0},{1,1,0,1,1,0,0,1,1,0},{1,1,0,1,0,1,0,1,1,0,0,1,1,0}}};
inline constexpr std::array<unsigned,5> puncture_lengths{2,4,6,10,14};
inline std::span<const unsigned> puncture_pattern(unsigned rate) {
    if(rate>=5)throw std::invalid_argument("Invalid DVB-T code rate");
    return {punctures[rate].data(),puncture_lengths[rate]};
}
// A bit is 0 or 1; 2 means an erased observation (including a non-finite carrier).
// Full survivor traceback, unknown initial/final state, no guessed termination.
class Viterbi {
    std::vector<std::uint64_t> survivors_;
    template<unsigned Rate> std::vector<std::uint8_t> decode_rate(std::span<const std::uint8_t>);
public:
    std::vector<std::uint8_t> decode(std::span<const std::uint8_t> bits,unsigned rate);
};

template<unsigned Primitive=0x11d>
struct Gf256 {
    static_assert(Primitive>=256 && Primitive<512);
    struct Tables {std::array<std::uint8_t,510> exp{};std::array<unsigned,256> log{};};
    static constexpr Tables table=[]() consteval {
        Tables t;unsigned x=1;std::array<bool,256> seen{};
        for(unsigned i=0;i<255;++i) {
            if(!x || seen[x])throw "GF polynomial is not primitive";
            seen[x]=true;t.exp[i]=x;t.log[x]=i;x<<=1;if(x&256)x^=Primitive;
        }
        for(unsigned i=255;i<510;++i)t.exp[i]=t.exp[i-255];return t;
    }();
    static constexpr std::uint8_t mul(unsigned a,unsigned b) {
        return a && b ? table.exp[table.log[a]+table.log[b]] : 0;
    }
    static constexpr std::uint8_t div(unsigned a,unsigned b) {
        assert(b!=0);
        return a ? table.exp[table.log[a]+255-table.log[b]] : 0;
    }
    static constexpr std::uint8_t alpha(unsigned exponent) {return table.exp[exponent%255];}
};
// Shortened systematic RS over GF(256), first root alpha^0 (DVB-T).
// The fixed dimensions expose parity arithmetic to SLP vectorization.
template<unsigned N=204,unsigned Parity=16,class Field=Gf256<>>
struct ReedSolomon {
    static_assert(N<=255 && Parity<N && Parity%2==0);
    using Word=std::array<std::uint8_t,N>;
    using Remainder=std::array<std::uint8_t,Parity>;
    static constexpr auto generator=[]() consteval {
        std::array<std::uint8_t,Parity+1> g{};g[0]=1;
        for(unsigned i=0;i<Parity;++i) {
            for(unsigned j=i+1;j>0;--j)g[j]=g[j-1]^Field::mul(g[j],Field::alpha(i));
            g[0]=Field::mul(g[0],Field::alpha(i));
        }
        return g; // ascending powers; highest coefficient is one
    }();
    static constexpr auto feedback=[]() consteval {
        std::array<Remainder,256> result{};
        for(unsigned b=0;b<256;++b)for(unsigned i=0;i<Parity;++i)
            result[b][i]=Field::mul(b,generator[Parity-1-i]);
        return result;
    }();
    static Remainder remainder(std::span<const std::uint8_t> bytes) {
        if constexpr(Parity==16) {
            // Two integer lanes express the whole LFSR shift/XOR. This avoids
            // Clang scalarizing the sixteen overlapping byte assignments.
            static constexpr auto words=[]() consteval {
                std::array<std::array<std::uint64_t,2>,256> result{};
                for(unsigned b=0;b<256;++b)for(unsigned i=0;i<16;++i)
                    result[b][i/8]|=std::uint64_t(feedback[b][i])<<(8*(i%8));
                return result;
            }();
            // Four-byte slicing removes three dependent table lookups per
            // group. Each table is one byte's contribution after 0..3 shifts.
            constexpr unsigned slices=4;
            static constexpr auto table=[]() consteval {
                std::array<std::array<std::array<std::uint64_t,2>,256>,slices> t{};
                t[0]=words;
                for(unsigned i=1;i<slices;++i)for(unsigned b=0;b<256;++b) {
                    const auto lo=t[i-1][b][0],hi=t[i-1][b][1];
                    const auto& f=words[lo&255];
                    t[i][b]={((lo>>8)|(hi<<56))^f[0],(hi>>8)^f[1]};
                }
                return t;
            }();
            std::uint64_t lo=0,hi=0;
            std::size_t offset=0;
            for(;offset+slices<=bytes.size();offset+=slices) {
                using Chunk=std::uint32_t;
                std::array<std::uint8_t,slices> chunk;
                std::copy_n(bytes.data()+offset,slices,chunk.data());
                auto input=std::bit_cast<Chunk>(chunk);
                if constexpr(std::endian::native==std::endian::big)input=std::byteswap(input);
                const auto data=lo^input;
                lo=(lo>>32)|(hi<<32);hi>>=32;
                [&]<std::size_t... I>(std::index_sequence<I...>) {
                    ((lo^=table[slices-1-I][(data>>(8*I))&255][0],hi^=table[slices-1-I][(data>>(8*I))&255][1]),...);
                }(std::make_index_sequence<slices>{});
            }
            for(;offset<bytes.size();++offset) {
                const auto& f=words[bytes[offset]^static_cast<std::uint8_t>(lo)];
                lo=((lo>>8)|(hi<<56))^f[0];hi=(hi>>8)^f[1];
            }
            if constexpr(std::endian::native==std::endian::big){lo=std::byteswap(lo);hi=std::byteswap(hi);}
            return std::bit_cast<Remainder>(std::array{lo,hi});
        }
        Remainder r{};
        for(auto b:bytes) {
            const auto& f=feedback[b^r[0]];
            for(unsigned i=0;i<Parity-1;++i)r[i]=r[i+1]^f[i];
            r.back()=f.back();
        }
        return r;
    }
    static bool valid(const Word& word) {
        const auto r=remainder(word);return std::all_of(r.begin(),r.end(),[](auto x){return x==0;});
    }
    static void encode(Word& word) {
        const auto parity=remainder(std::span{word}.first(N-Parity));
        std::copy(parity.begin(),parity.end(),word.begin()+N-Parity);
    }
    // Returns corrected symbol count, or -1. Invalid words are left unchanged.
    static int decode(Word& word) {
        if(valid(word))return 0;
        Remainder syndrome{};
        for(unsigned i=0;i<Parity;++i)for(auto b:word)syndrome[i]=Field::mul(syndrome[i],Field::alpha(i))^b;
        std::array<std::uint8_t,Parity+1> c{},b{};c[0]=b[0]=1;
        unsigned degree=0,shift=1;std::uint8_t previous=1;
        for(unsigned n=0;n<Parity;++n) {
            auto discrepancy=syndrome[n];
            for(unsigned i=1;i<=degree;++i)discrepancy^=Field::mul(c[i],syndrome[n-i]);
            if(!discrepancy){++shift;continue;}
            const auto old=c;const auto scale=Field::div(discrepancy,previous);
            for(unsigned i=0;i+shift<=Parity;++i)c[i+shift]^=Field::mul(scale,b[i]);
            if(2*degree<=n){degree=n+1-degree;b=old;previous=discrepancy;shift=1;}else ++shift;
        }
        if(!degree || degree>Parity/2)return -1;
        Remainder omega{};
        for(unsigned i=0;i<Parity;++i)for(unsigned j=0;j<=degree && j<=i;++j)
            omega[i]^=Field::mul(syndrome[i-j],c[j]);
        auto corrected=word;unsigned roots=0;
        for(unsigned position=0;position<N;++position) {
            const auto x=Field::alpha(N-1-position),z=Field::div(1,x);
            auto value=c[degree];for(unsigned i=degree;i-->0;)value=Field::mul(value,z)^c[i];
            if(value)continue;
            std::uint8_t derivative=0,power=1,numerator=0;
            for(unsigned i=1;i<=degree;++i){if(i&1)derivative^=Field::mul(c[i],power);power=Field::mul(power,z);}
            for(unsigned i=Parity;i-->0;)numerator=Field::mul(numerator,z)^omega[i];
            if(!derivative || ++roots>degree)return -1;
            corrected[position]^=Field::div(Field::mul(x,numerator),derivative);
        }
        // Searching only transmitted positions rejects roots in the shortened prefix.
        if(roots!=degree || !valid(corrected))return -1;
        word=corrected;return static_cast<int>(roots);
    }
};
using DvbtRs=ReedSolomon<>;
struct RsResult {DvbtRs::Word bytes{};bool valid=false;unsigned corrected=0;};
}
