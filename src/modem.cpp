// DVB-T mappings follow ETSI EN 300 744 and GNU Radio gr-dtv (GPL-3.0-or-later).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "modem.hpp"
#include "cpu_fec.hpp"
#include "dvbt_configure.h"
#include "dvbt_pilot_gen.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace radar {
unsigned qam_bits(const Tps& t) {
    if (t.mode!=1 || t.hierarchy!=0 || t.constellation<0 || t.constellation>2 || t.rate<0 || t.rate>4)
        throw std::runtime_error("Reconstruction currently supports non-hierarchical 8K DVB-T only");
    return 2+2*t.constellation;
}
namespace {
constexpr int bins=6817, payload=6048;
constexpr std::array<int,6> offsets{0,63,105,42,21,84};
std::vector<C> constellation(int m) {
    const int size=1<<m, axis=m/2, steps=(1<<axis)/2-1;
    const float norm=1/std::sqrt(m==2 ? 2.0F : m==4 ? 10.0F : 42.0F);
    std::vector<C> points(size);
    for (int i=0;i<size;++i) {
        const int q=(i>>(m-2))&3;
        int x=(i>>(axis-1))&((1<<(axis-1))-1), y=i&((1<<(axis-1))-1);
        const int xv=1+2*(steps-x), yv=1+2*(steps-y);
        const int v=(((x>>1)^x)<<(axis-1))+((y>>1)^y);
        x=0; y=0;
        for (int j=0;j<axis-1;++j) { x|=((v>>(1+2*j))&1)<<j; y|=((v>>(2*j))&1)<<j; }
        const int label=(q<<(m-2))+(x<<(axis-1))+y;
        points[label]=norm*C{static_cast<float>((q>>1) ? -xv : xv),static_cast<float>((q&1) ? -yv : yv)};
    }
    return points;
}
std::vector<int> symbol_permutation() {
    constexpr int p[]{7,1,4,2,9,6,8,10,0,3,11,5};
    std::vector<int> h;
    int reg=0;
    for (int i=0;i<8192;++i) {
        if (i==2) reg=1;
        else if (i>2) reg=(reg>>1)|(((reg^(reg>>1)^(reg>>4)^(reg>>6))&1)<<11);
        int r=(i&1)<<12;
        for (int k=0;k<12;++k) r|=((reg>>k)&1)<<p[k];
        if (r<payload) h.push_back(r);
    }
    if (h.size()!=payload) throw std::runtime_error("DVB-T permutation generation failed");
    return h;
}
struct Encoded { std::vector<std::uint8_t> bits, valid; };
Encoded convolutional_encode(std::span<const std::uint8_t> bytes,
                             std::span<const std::uint8_t> valid, int rate) {
    const auto pattern=fec::puncture_pattern(rate);
    // The convolutional code is linear: XOR the previous-state contribution
    // with the current byte contribution. Two small tables replace per-bit parity.
    struct Tables {
        std::array<std::uint16_t,64> state{};
        std::array<std::uint16_t,256> byte{};
        static constexpr std::uint16_t encode(unsigned reg,unsigned value) {
            std::uint16_t word=0;
            for(unsigned bit=0;bit<8;++bit) {
                reg=((reg<<1)|((value>>(7-bit))&1U))&127U;
                word|=static_cast<std::uint16_t>((fec::DvbtCode::symbol(reg)&1U)<<(2*bit));
                word|=static_cast<std::uint16_t>(((fec::DvbtCode::symbol(reg)>>1)&1U)<<(2*bit+1));
            }
            return word;
        }
        constexpr Tables() {
            for(unsigned i=0;i<state.size();++i)state[i]=encode(i,0);
            for(unsigned i=0;i<byte.size();++i)byte[i]=encode(0,i);
        }
    };
    static constexpr Tables table;
    const auto phases=pattern.size()/std::gcd(pattern.size(),std::size_t{16});
    std::array<std::array<unsigned,16>,7> positions{};
    std::array<unsigned,7> counts{};
    for(std::size_t phase=0;phase<phases;++phase)for(unsigned bit=0;bit<16;++bit)
        if(pattern[(phase*16+bit)%pattern.size()])positions[phase][counts[phase]++]=bit;
    const auto total=bytes.size()/phases*std::accumulate(counts.begin(),counts.end(),std::size_t{})+
        std::accumulate(counts.begin(),counts.begin()+bytes.size()%phases,std::size_t{});
    Encoded result;result.bits.resize(total);result.valid.resize(total);
    std::size_t at=0,phase=0;unsigned state=0;bool previous_good=false;
    for(std::size_t i=0;i<bytes.size();++i) {
        const auto word=table.state[state]^table.byte[bytes[i]];
        const bool good=valid.empty() || valid[i];
        for(unsigned j=0;j<counts[phase];++j) {
            const auto bit=positions[phase][j];
            result.bits[at]=static_cast<std::uint8_t>((word>>bit)&1U);
            // After seven input bits the previous byte can no longer affect validity.
            result.valid[at++]=good && (previous_good || bit>=12);
        }
        state=bytes[i]&63U;previous_good=good;
        if(++phase==phases)phase=0;
    }
    return result;
}
}

std::vector<C> qam_constellation(int bits) {
    if(bits!=2 && bits!=4 && bits!=6)throw std::invalid_argument("Invalid QAM bit count");
    return constellation(bits);
}

struct Modem::Impl {
    gr::dl5eu::dvbt_configure config;
    gr::dl5eu::dvbt_pilot_gen pilots;
    std::vector<int> h=symbol_permutation();
    std::vector<char> prbs;
    explicit Impl(int guard) : config(gr::dtv::MOD_64QAM,gr::dtv::NH,gr::dtv::C3_4,gr::dtv::C3_4,
        guard==4 ? gr::dtv::GI_1_4 : guard==8 ? gr::dtv::GI_1_8 : guard==16 ? gr::dtv::GI_1_16 : gr::dtv::GI_1_32,gr::dtv::T8k),pilots(config),prbs(pilots.pilot_values()) {}
};
Modem::Modem(int guard) : impl_(std::make_unique<Impl>(guard)) {}
Modem::~Modem() = default;
PilotPattern Modem::pilot_pattern() const {
    PilotPattern result;
    for(unsigned k=0;k<6817;++k)result.values[k]=impl_->prbs[k] ? -4.0F/3 : 4.0F/3;
    for(int i=0;i<impl_->pilots.cpilot_carriers_size();++i)result.continual.push_back(impl_->pilots.cpilot_carriers()[i]);
    for(int i=0;i<impl_->pilots.tps_carriers_size();++i)result.tps.push_back(impl_->pilots.tps_carriers()[i]);
    return result;
}
std::vector<int> Modem::data_carriers(int modulo4) const {
    std::array<bool,bins> excluded{};
    for (int i=0;i<impl_->pilots.cpilot_carriers_size();++i) excluded[impl_->pilots.cpilot_carriers()[i]]=true;
    for (int i=0;i<impl_->pilots.tps_carriers_size();++i) excluded[impl_->pilots.tps_carriers()[i]]=true;
    for (int i=3*modulo4;i<bins;i+=12) excluded[i]=true;
    std::vector<int> result;
    for (int i=0;i<bins;++i) if (!excluded[i]) result.push_back(i);
    if (result.size()!=payload) throw std::runtime_error("Invalid pilot partition");
    return result;
}
bool Modem::tps(std::span<const Symbol> frame, Tps& result) const {
    if (frame.size()!=68) return false;
    std::array<std::uint8_t,68> decoded{};
    for (std::size_t s=1;s<68;++s) {
        double sum=0;
        for (int j=0;j<impl_->pilots.tps_carriers_size();++j) {
            const auto k=impl_->pilots.tps_carriers()[j];
            sum+=(frame[s].equalized[k]*std::conj(frame[s-1].equalized[k])).real();
        }
        decoded[s]=sum<0;
    }
    return tps_bits(decoded,result);
}
bool Modem::tps_bits(std::span<const std::uint8_t> decoded,Tps& result) const {
    if(decoded.size()!=68 || std::any_of(decoded.begin(),decoded.end(),[](auto v){return v>1;}))return false;
    std::copy(decoded.begin(),decoded.end(),result.bits.begin());
    auto bits=[&](int begin,int end) { int value=0; for(int i=begin;i<=end;++i)value=value*2+result.bits[i]; return value; };
    const auto sync=bits(1,16);
    if (sync!=0x35ee && sync!=0xca11) return false;
    std::deque<char> b(result.bits.begin(),result.bits.end());
    if (impl_->pilots.verify_bch_code(b)) return false;
    result.frame=bits(23,24); result.constellation=bits(25,26); result.hierarchy=bits(27,29);
    result.rate=bits(30,32); result.guard=bits(36,37); result.mode=bits(38,39);
    return (result.frame%2 ? 0xca11 : 0x35ee)==sync;
}

std::vector<std::uint32_t> Modem::payload_bit_indices(const Tps& mode) const {
    const unsigned m=qam_bits(mode);
    std::vector<std::uint32_t> result(4*payload*m);
    for(unsigned mod=0;mod<4;++mod) {
        const auto carriers=data_carriers(mod);
        std::vector<unsigned> inverse(payload);
        for(unsigned q=0;q<payload;++q)inverse[impl_->h[q]]=q;
        for(unsigned i=0;i<payload;++i)for(unsigned bit=0;bit<m;++bit) {
            const unsigned e=bit/(m/2)+2*(bit%(m/2));
            const unsigned w=i/126*126+(i%126+126-offsets[e])%126;
            const unsigned source=mod&1 ? inverse[w] : impl_->h[w];
            result[(mod*payload+i)*m+bit]=8*carriers[source]+e;
        }
    }
    return result;
}

Reconstruction Modem::reconstruct(std::span<const Symbol> symbols,const Tps& mode,ReconstructionTimings* timings,CpuFec* decoder) const {
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    auto measure=[&](double ReconstructionTimings::* stage) {
        if(timings) {const auto now=Clock::now();timings->*stage+=std::chrono::duration<double>(now-mark).count();mark=now;}
    };
    const int m=qam_bits(mode);
    const auto points=constellation(m);
    std::vector<std::uint8_t> corrected,good;
    Reconstruction result;
    CpuFec local;
    FecTimings fec_timing;
    const auto decoded=(decoder ? *decoder : local).decode(symbols,mode,*this,timings ? &fec_timing : nullptr);
    corrected.resize(decoded.trellis_bits/8);good.resize(corrected.size());
    for(std::size_t p=0;p<decoded.count;++p) {
        const auto& packet=decoded.packets[p];++result.packets;
        result.bad_packets+=!packet.valid;result.corrected_bytes+=packet.corrected;
        std::array<std::uint8_t,188> data{};std::copy_n(packet.bytes.begin(),188,data.begin());
        result.corrected_packets.push_back(data);result.packet_valid.push_back(packet.valid);
        for(std::size_t j=0;j<204;++j) {
            const auto i=2244+p*204+j,destination=i-(11-i%12)*204;
            corrected[destination]=packet.bytes[j];good[destination]=packet.valid;
        }
    }
    if(timings) {
        timings->demap+=fec_timing.demap;timings->viterbi+=fec_timing.viterbi;
        timings->reed_solomon+=fec_timing.reed_solomon;mark=Clock::now();
    }
    const auto recoded=convolutional_encode(corrected,good,mode.rate);
    measure(&ReconstructionTimings::reencode);
    result.carriers.resize(symbols.size()*bins);
    result.valid.resize(symbols.size()*bins);
    std::vector<std::uint8_t> labels(payload), labels_valid(payload);
    Tps frame_tps{};
    int tps_sign=1;
    for(std::size_t s=0;s<symbols.size();++s) {
        if(s%68==0) {
            tps_sign=1;
            if(s+68<=symbols.size() && !tps(symbols.subspan(s,68),frame_tps))
                throw std::runtime_error("Cannot regenerate TPS from a frame without BCH validation");
        } else if(frame_tps.bits[s%68]) tps_sign=-tps_sign;
        for(int b=0;b<payload;b+=126) {
            std::uint8_t streams[6][126]{}, validity[6][126]{};
            for(int i=0;i<126;++i) for(int bit=0;bit<m;++bit) {
                const auto at=(s*payload+b+i)*static_cast<std::size_t>(m)+bit;
                const int e=bit/(m/2)+2*(bit%(m/2));
                if (at<recoded.bits.size()) { streams[e][i]=recoded.bits[at]; validity[e][i]=recoded.valid[at]; }
            }
            for(int w=0;w<126;++w) {
                labels[b+w]=0; labels_valid[b+w]=1;
                for(int e=0;e<m;++e) {
                    labels[b+w]=static_cast<std::uint8_t>((labels[b+w]<<1)|streams[e][(w+offsets[e])%126]);
                    labels_valid[b+w]&=validity[e][(w+offsets[e])%126];
                }
            }
        }
        const auto carriers=data_carriers(symbols[s].modulo4);
        for(int q=0;q<payload;++q) {
            const int src=(symbols[s].modulo4&1) ? impl_->h[q] : q;
            const int dst=(symbols[s].modulo4&1) ? q : impl_->h[q];
            result.carriers[s*bins+carriers[dst]]=points[labels[src]];
            result.valid[s*bins+carriers[dst]]=labels_valid[src];
        }
        // Pilots are known; regenerate TPS DBPSK from BCH-checked bits and each carrier's PRBS seed.
        for(int k=0;k<bins;++k) if (result.carriers[s*bins+k]==C{}) {
            result.carriers[s*bins+k]=C{(impl_->prbs[k] ? -1.0F : 1.0F)*4/3,0};
            result.valid[s*bins+k]=1;
        }
        for(int j=0;j<impl_->pilots.tps_carriers_size();++j) {
            const auto k=impl_->pilots.tps_carriers()[j];
            result.carriers[s*bins+k]=C{static_cast<float>(tps_sign*(impl_->prbs[k] ? -1 : 1)),0};
            if((s/68)*68+68>symbols.size())
                result.valid[s*bins+k]=0;
        }
    }
    measure(&ReconstructionTimings::remap);
    return result;
}

std::vector<C> Modem::synthesize(std::span<const std::uint8_t> packets,const Tps& mode,std::size_t count) const {
    const int m=qam_bits(mode);
    if (packets.size()%188) throw std::runtime_error("Randomized packet input must be a multiple of 188 bytes");
    std::vector<std::uint8_t> rs(packets.size()/188*204);
    for(std::size_t p=0;p<packets.size()/188;++p) {
        fec::DvbtRs::Word code{};
        std::copy_n(packets.begin()+static_cast<std::ptrdiff_t>(p*188),188,code.begin());
        fec::DvbtRs::encode(code);
        std::copy(code.begin(),code.end(),rs.begin()+static_cast<std::ptrdiff_t>(p*204));
    }
    std::vector<std::uint8_t> outer(rs.size());
    for(std::size_t i=0;i<rs.size();++i) if(i>=(i%12)*204) outer[i]=rs[i-(i%12)*204];
    const auto encoded=convolutional_encode(outer,{},mode.rate);
    if(encoded.bits.size()<count*payload*static_cast<std::size_t>(m)) throw std::runtime_error("Insufficient packets for requested symbols");
    const auto points=constellation(m);
    std::vector<C> out(count*bins);
    std::vector<std::uint8_t> labels(payload);
    std::array<std::uint8_t,68> tps_bits{};
    int tps_sign=1;
    for(std::size_t s=0;s<count;++s) {
        if(s%68==0) {
            tps_bits.fill(0);tps_sign=1;
            auto set=[&](int first,int last,int value) { for(int bit=last;bit>=first;--bit) {tps_bits[bit]=value&1;value>>=1;} };
            const int frame=(mode.frame+static_cast<int>(s/68))%4;
            set(1,16,frame%2 ? 0xca11 : 0x35ee);set(17,22,0x17);set(23,24,frame);
            set(25,26,mode.constellation);set(27,29,mode.hierarchy);set(30,32,mode.rate);
            set(33,35,mode.rate);set(36,37,mode.guard);set(38,39,mode.mode);
            unsigned reg=0;
            for(int bit=1;bit<=53;++bit) {
                const unsigned feedback=(tps_bits[bit]^reg)&1;
                reg=(reg>>1)^(feedback*0x3bb0U);
            }
            for(int bit=0;bit<14;++bit)tps_bits[54+bit]=(reg>>bit)&1;
        } else if(tps_bits[s%68])tps_sign=-tps_sign;
        for(int b=0;b<payload;b+=126) {
            std::uint8_t streams[6][126]{};
            for(int i=0;i<126;++i) for(int bit=0;bit<m;++bit)
                streams[bit/(m/2)+2*(bit%(m/2))][i]=encoded.bits[(s*payload+b+i)*static_cast<std::size_t>(m)+bit];
            for(int w=0;w<126;++w) {
                labels[b+w]=0;
                for(int e=0;e<m;++e) labels[b+w]=static_cast<std::uint8_t>((labels[b+w]<<1)|streams[e][(w+offsets[e])%126]);
            }
        }
        const auto carriers=data_carriers(static_cast<int>(s%4));
        for(int q=0;q<payload;++q) {
            const int src=(s&1) ? impl_->h[q] : q, dst=(s&1) ? q : impl_->h[q];
            out[s*bins+carriers[dst]]=points[labels[src]];
        }
        for(int k=0;k<bins;++k) if(out[s*bins+k]==C{}) out[s*bins+k]=C{(impl_->prbs[k] ? -1.0F : 1.0F)*4/3,0};
        for(int j=0;j<impl_->pilots.tps_carriers_size();++j) {
            const auto k=impl_->pilots.tps_carriers()[j];
            out[s*bins+k]=C{static_cast<float>(tps_sign*(impl_->prbs[k] ? -1 : 1)),0};
        }
    }
    return out;
}
} // namespace radar
