#include "fec.hpp"
extern "C" {
#include <gnuradio/fec/rs.h>
}
#include <iostream>
#include <random>
#include <numeric>
using namespace radar::fec;
void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
// Deliberately scalar 32-bit reference with independent polynomial evaluation.
std::vector<std::uint8_t> reference(std::span<const std::uint8_t> bits,unsigned rate) {
    const auto pattern=puncture_pattern(rate);const auto count=bits.size()/std::count(pattern.begin(),pattern.end(),1U)*pattern.size()/2;
    std::vector<std::array<unsigned,64>> history(count);std::array<unsigned,64> metrics{},next{};
    std::size_t at=0;
    for(std::size_t t=0;t<count;++t) {
        std::array<unsigned,2> y{2,2};for(unsigned j=0;j<2;++j)if(pattern[(2*t+j)%pattern.size()])y[j]=bits[at++];
        next.fill(UINT32_MAX);
        for(unsigned s=0;s<64;++s)for(unsigned b=0;b<2;++b) {
            const auto reg=2*s+b;unsigned cost=metrics[s];
            for(unsigned j=0;j<2;++j){unsigned parity=0;for(unsigned i=0;i<7;++i)parity^=((reg>>i)&1)*(((j ? 0x6dU:0x4fU)>>i)&1);cost+=y[j]<2 && y[j]!=parity;}
            const auto dest=reg%64;if(cost<next[dest]){next[dest]=cost;history[t][dest]=s;}
        }
        metrics=next;
    }
    unsigned state=std::min_element(metrics.begin(),metrics.end())-metrics.begin();std::vector<std::uint8_t> bytes(count/8);
    for(auto t=count;t-->0;){if(t/8<bytes.size())bytes[t/8]|=(state&1)<<(7-t%8);state=history[t][state];}return bytes;
}
int main(){try {
    std::mt19937 rng(82741);Viterbi v;
    for(unsigned rate=0;rate<5;++rate)for(unsigned length: {1U,7U,64U,257U,4096U}) {
        const auto p=puncture_pattern(rate);std::vector<std::uint8_t> y(length*std::count(p.begin(),p.end(),1U));
        for(auto& b:y)b=rng()%3;
        check(v.decode(y,rate)==reference(y,rate),"Viterbi differs from scalar exact ML oracle");
    }
    // Exercise both sides of every new normalization-block boundary. Include
    // worst-case hard observations, complete erasures and noncanonical erasures.
    for(unsigned rate=0;rate<5;++rate) {
        const auto p=puncture_pattern(rate);const unsigned cycles=96/(p.size()/2);
        for(unsigned length:{cycles-1,cycles,cycles+1,2*cycles-1,2*cycles,2*cycles+1,4097U}) {
            std::vector<std::uint8_t> y(length*std::count(p.begin(),p.end(),1U));
            for(unsigned kind=0;kind<5;++kind) {
                for(auto& b:y)b=kind<3 ? kind : kind==3 ? rng()%2 : (rng()%3==2 ? 255 : rng()%2);
                check(v.decode(y,rate)==reference(y,rate),"Viterbi normalization boundary differs from oracle");
            }
        }
    }
    check(v.decode({},0).empty(),"Empty trellis");
    bool rejected=false;
    try{const std::array<std::uint8_t,1> incomplete{};v.decode(incomplete,2);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"Incomplete puncture cycle accepted");
    // Polynomial long division is independent of the sliced remainder tables.
    // Check every tail size and unaligned starting address, beyond packet sizes.
    std::vector<std::uint8_t> input(1032);for(auto& b:input)b=rng();
    for(unsigned offset=0;offset<8;++offset)for(unsigned length=0;length<=1024;++length) {
        const auto bytes=std::span{input}.subspan(offset,length);
        std::vector<std::uint8_t> dividend(bytes.begin(),bytes.end());dividend.resize(length+16);
        for(unsigned i=0;i<length;++i) {
            const auto lead=dividend[i];
            for(unsigned j=1;j<=16;++j)dividend[i+j]^=Gf256<>::mul(lead,DvbtRs::generator[16-j]);
        }
        const auto actual=DvbtRs::remainder(bytes);
        check(std::equal(actual.begin(),actual.end(),dividend.begin()+length),"Sliced RS remainder differs from polynomial division");
    }
    void* rs=init_rs_char(8,0x11d,0,1,16);check(rs,"libfec init");
    std::array<unsigned,204> positions;std::iota(positions.begin(),positions.end(),0);
    for(unsigned trial=0;trial<500;++trial) {
        DvbtRs::Word word{};std::array<unsigned char,255> legacy{};
        for(unsigned i=0;i<188;++i)legacy[51+i]=word[i]=rng();
        encode_rs_char(rs,legacy.data(),legacy.data()+239);DvbtRs::encode(word);
        check(std::equal(word.begin(),word.end(),legacy.begin()+51),"RS encoder parity differs from libfec");
        check(DvbtRs::valid(word),"Encoded RS word failed parity");
        for(unsigned errors=0;errors<=9;++errors) {
            auto damaged=word;std::shuffle(positions.begin(),positions.end(),rng);
            for(unsigned i=0;i<errors;++i)damaged[positions[i]]^=1+rng()%255;
            const auto original=damaged;const auto actual=DvbtRs::decode(damaged);
            std::fill(legacy.begin(),legacy.begin()+51,0);std::copy(original.begin(),original.end(),legacy.begin()+51);
            const auto expected=decode_rs_char(rs,legacy.data(),nullptr,0);
            if(errors<=8)check(actual==int(errors) && damaged==word,"RS failed correctable word");
            if(actual>=0)check(expected>=0 && std::equal(damaged.begin(),damaged.end(),legacy.begin()+51),"RS correction differs from libfec");
            else check(damaged==original,"Failed RS mutated input");
        }
    }
    std::array<unsigned char,255> nonshortened{};nonshortened[3]=123;
    for(unsigned i=51;i<239;++i)nonshortened[i]=rng();
    encode_rs_char(rs,nonshortened.data(),nonshortened.data()+239);
    DvbtRs::Word shortened;std::copy_n(nonshortened.begin()+51,204,shortened.begin());
    const auto unchanged=shortened;
    check(DvbtRs::decode(shortened)<0 && shortened==unchanged,"RS accepted a root in the shortened prefix");
    free_rs_char(rs);std::cout<<"Viterbi exact oracle, all punctures/erasures/ties, RS independent parity and 0..9 errors: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
