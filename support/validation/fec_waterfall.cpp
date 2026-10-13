// Known-payload qualification of the complete hard-decision FEC chain.
// Optional GPU mode includes the production FFT/pilot equaliser, with exact
// window timing, to separate equalisation floors from decoder implementation.
#include "cpu_fec.hpp"
#include "dsp.hpp"
#include "gpu_ofdm.hpp"
#include "gpu_frames.hpp"
#include "modem.hpp"
#include <array>
#include <charconv>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <string_view>

using namespace radar;
namespace {
constexpr unsigned bins=6817,fft=8192,rows=204,history=68;
constexpr double tau=2*std::numbers::pi;
// ETSI EN 300 744 Annex B, path amplitude, delay in us, phase in radians.
constexpr std::array<std::array<double,3>,20> paths{{
    {.057662,1.003019,4.855121},{.176809,5.422091,3.419109},{.407163,.518650,5.864470},
    {.303585,2.751772,2.215894},{.258782,.602895,3.758058},{.061831,1.016585,5.430202},
    {.150340,.143556,3.952093},{.051534,.153832,1.093586},{.185074,3.324866,5.775198},
    {.400967,1.935570,.154459},{.295723,.429948,5.928383},{.350825,3.228872,3.053023},
    {.262909,.848831,.628578},{.225894,.073883,2.128544},{.170996,.203952,1.099463},
    {.149723,.194207,3.462951},{.240140,.924450,3.664773},{.116587,1.381320,2.833799},
    {.221155,.640512,3.334290},{.259730,1.368671,.393889}}};
std::array<C,bins> channel(std::string_view kind) {
    std::array<C,bins> h;
    double power=0;for(auto p:paths)power+=p[0]*p[0];
    double energy=0;
    for(unsigned k=0;k<bins;++k){
        const double frequency=(double(k)-3408)*8e6/fft;
        std::complex<double> value=1;
        if(kind=="f1" || kind=="p1"){
            value=kind=="f1" ? std::sqrt(10*power) : 0;
            for(auto p:paths)value+=std::polar(p[0],-p[2]-tau*frequency*p[1]*1e-6);
        }else if(kind=="echo")value+=std::polar(.8,-tau*frequency*20e-6+.7);
        else if(kind!="awgn" && kind!="burst")throw std::invalid_argument("Unknown channel model");
        h[k]=static_cast<C>(value);energy+=std::norm(value);
    }
    for(auto& z:h)z/=float(std::sqrt(energy/bins));
    return h;
}
struct Counts {
    std::uint64_t words{},rejected{},wrong{},corrected{},edge_words{},edge_rejected{},bits{},bit_errors{};
    std::uint64_t coded_bits{},coded_errors{},trellis_bits{},trellis_errors{},beyond_radius{},radius_violation{};
    void add(const FecBatch& decoded,std::span<const std::uint8_t> bytes,std::size_t first_packet){
        for(std::size_t p=0;p<decoded.count;++p){
            const auto& word=decoded.packets[p];++words;rejected+=!word.valid;corrected+=word.corrected;
            const auto expected=bytes.subspan((first_packet+p)*188,188);
            wrong+=word.valid && !std::equal(expected.begin(),expected.end(),word.bytes.begin());
            for(unsigned j=0;j<188;++j)bit_errors+=std::popcount(unsigned(word.bytes[j]^expected[j]));
            bits+=188*8;
            if(p<16 || p+16>=decoded.count){++edge_words;edge_rejected+=!word.valid;}
        }
    }
    // Deliberately separate diagnostic path. Production does not pay for this
    // second traceback or known-truth comparisons. Count every bit, including
    // both unknown-state trellis boundaries, rather than trimming a waterfall.
    void inspect(std::span<const Symbol> symbols,const Tps& mode,Modem& modem,
                 std::span<const C> transmitted,std::span<const std::uint8_t> payload,
                 std::size_t first_packet,const FecBatch& decoded) {
        const unsigned m=2+2*mode.constellation;
        const auto indices=modem.payload_bit_indices(mode);
        std::vector<std::uint8_t> received(symbols.size()*6048*m);
        auto label=[&](C z){return m==2 ? hard_qam<2>(z) : m==4 ? hard_qam<4>(z) : hard_qam<6>(z);};
        for(unsigned row=0;row<symbols.size();++row)for(unsigned i=0;i<6048*m;++i){
            const auto index=indices[symbols[row].modulo4*6048*m+i];
            const auto value=label(symbols[row].equalized[index/8]);
            auto& bit=received[row*6048*m+i];bit=value==255 ? 2 : (value>>(index%8))&1;
            coded_errors+=bit!=((label(transmitted[row*bins+index/8])>>(index%8))&1);
        }
        coded_bits+=received.size();
        fec::Viterbi viterbi;const auto bytes=viterbi.decode(received,mode.rate);
        std::vector<std::uint8_t> rs(payload.size()/188*204);
        for(std::size_t p=0;p<payload.size()/188;++p){
            fec::DvbtRs::Word word{};std::copy_n(payload.begin()+p*188,188,word.begin());fec::DvbtRs::encode(word);
            std::copy(word.begin(),word.end(),rs.begin()+p*204);
        }
        for(std::size_t i=0;i<bytes.size();++i){
            const auto global=first_packet*204+i;
            trellis_errors+=std::popcount(unsigned(bytes[i]^rs[global-(global%12)*204]));
        }
        trellis_bits+=bytes.size()*8;
        for(std::size_t p=0;p<decoded.count;++p){
            unsigned errors=0;
            for(unsigned j=0;j<204;++j){const auto i=2244+p*204+j;
                errors+=bytes[i-(11-i%12)*204]!=rs[(first_packet+p)*204+j];}
            beyond_radius+=errors>8;
            radius_violation+=errors<=8 && (!decoded.packets[p].valid ||
                !std::equal(rs.begin()+(first_packet+p)*204,rs.begin()+(first_packet+p+1)*204,decoded.packets[p].bytes.begin()));
        }
    }
};
unsigned number(std::string_view s){unsigned n;auto [end,error]=std::from_chars(s.data(),s.data()+s.size(),n);
    if(error!=std::errc{} || end!=s.data()+s.size())throw std::invalid_argument("Invalid unsigned argument");return n;}
}
int main(int argc,char** argv){try{
    unsigned blocks=1;int selected_qam=-1,selected_rate=-1;bool gpu=false,clean=false,inspect=false,frame_pilots=false;
    std::string model="awgn";std::vector<double> levels;
    for(int i=1;i<argc;++i){const std::string_view key=argv[i];
        if(key=="--gpu"){gpu=true;continue;}if(key=="--clean"){clean=true;continue;}if(key=="--inspect"){inspect=true;continue;}
        if(key=="--frame-pilots"){gpu=true;frame_pilots=true;continue;}
        if(++i==argc)throw std::invalid_argument("Missing argument");const std::string_view value=argv[i];
        if(key=="--blocks")blocks=number(value);else if(key=="--qam")selected_qam=number(value);
        else if(key=="--rate")selected_rate=number(value);else if(key=="--model")model=value;
        else if(key=="--snr")levels.push_back(std::stod(std::string(value)));
        else throw std::invalid_argument("Usage: fec_waterfall [--gpu] [--clean] [--inspect] [--blocks N] [--qam 0..2] [--rate 0..4] [--model awgn|f1|p1|echo|burst] [--snr DB ...]");
    }
    if(!blocks || selected_qam>2 || selected_rate>4)throw std::invalid_argument("Invalid sweep settings");
    Modem modem;CpuFec decoder;const auto h=channel(model);
    std::unique_ptr<GpuOfdm> ofdm;if(gpu)ofdm=std::make_unique<GpuOfdm>(modem.pilot_pattern(),rows);
    std::unique_ptr<GpuFrameStorage> storage;if(frame_pilots)storage=std::make_unique<GpuFrameStorage>(ofdm->context());
    Fft transform(fft);
    std::cout<<"model,receiver,qam,rate,esn0_db,blocks,packets,rejected,wrong_accepted,corrected_bytes,edge_packets,edge_rejected,payload_bit_errors,payload_bits,coded_bits,coded_errors,trellis_bits,trellis_errors,beyond_radius,radius_violation\n";
    for(int qam=0;qam<3;++qam)for(int rate=0;rate<5;++rate){
        if((selected_qam>=0 && qam!=selected_qam)||(selected_rate>=0 && rate!=selected_rate))continue;
        Tps mode;mode.mode=1;mode.guard=1;mode.constellation=qam;mode.rate=rate;
        const auto pattern=fec::puncture_pattern(rate);const auto sent=std::count(pattern.begin(),pattern.end(),1U);
        const unsigned bits=2+2*qam;
        const auto source_bytes=(history+rows)*6048*bits/sent*(pattern.size()/2)/8;
        const auto first_packet=history*6048*bits/sent*(pattern.size()/2)/8/204;
        std::vector<double> sweep=levels;
        if(sweep.empty()){
            if(clean)sweep={100};
            else for(double db=2;db<=14;db+=1)sweep.push_back(db+qam*6+(model=="p1" ? 7 : 0));
        }
        for(double snr:sweep){
            std::mt19937 random(817331+qam*101+rate*1009);std::normal_distribution<float> normal;
            const float sigma=clean ? 0 : std::sqrt(std::pow(10.,-snr/10)/2);
            Counts total;
            for(unsigned trial=0;trial<blocks;++trial){
                std::vector<std::uint8_t> bytes((source_bytes/204+16)*188);
                for(auto& b:bytes)b=random();for(std::size_t p=0;p<bytes.size()/188;++p)bytes[p*188]=p%8 ? 0x47 : 0xb8;
                const auto tx=modem.synthesize(bytes,mode,history+rows);
                std::vector<Symbol> symbols(rows);std::vector<C> samples(gpu ? rows*fft : 0);
                std::vector<OfdmWindow> windows(rows);
                for(unsigned row=0;row<rows;++row){
                    auto& symbol=symbols[row];symbol.modulo4=row%4;symbol.equalized.resize(bins);
                    std::vector<C> spectrum(gpu ? fft : 0);
                    for(unsigned k=0;k<bins;++k){
                        const float burst=model=="burst" && row%68==23 ? 10 : 1;
                        const C y=h[k]*tx[(history+row)*bins+k]+sigma*burst*C{normal(random),normal(random)};
                        symbol.equalized[k]=y/h[k];
                        if(gpu)spectrum[(k+fft-3408)%fft]=y*static_cast<C>(std::polar(1.,-tau*(int(k)-3408)*fft_window_margin/fft));
                    }
                    if(gpu){transform.transform(spectrum,true);std::copy(spectrum.begin(),spectrum.end(),samples.begin()+row*fft);windows[row]={row*fft,row*8704ULL,0,0};}
                }
                FecBatch decoded;
                if(gpu){auto batch=ofdm->transform(samples,windows);
                    if(frame_pilots){FrameSelection selection;selection.frame.carriers=batch;selection.frame.frames.assign(rows/68,mode);
                        for(unsigned row=0;row<rows;++row)selection.rows.push_back(row);
                        batch=storage->capture(selection).carriers;
                    }
                    decoded=decoder.decode(batch,mode,modem);
                    if(inspect)symbols=ofdm->download(batch,windows);
                }else decoded=decoder.decode(symbols,mode,modem);
                total.add(decoded,bytes,first_packet);
                if(inspect)total.inspect(symbols,mode,modem,std::span{tx}.subspan(history*bins),bytes,first_packet,decoded);
            }
            std::cout<<model<<','<<(frame_pilots ? "gpu_frame_pilots" : gpu ? "gpu_pilots" : "known_channel")<<','<<(1U<<bits)<<','<<rate<<','<<snr<<','<<blocks<<','
                <<total.words<<','<<total.rejected<<','<<total.wrong<<','<<total.corrected<<','<<total.edge_words<<','<<total.edge_rejected<<','
                <<total.bit_errors<<','<<total.bits<<','<<total.coded_bits<<','<<total.coded_errors<<','<<total.trellis_bits<<','
                <<total.trellis_errors<<','<<total.beyond_radius<<','<<total.radius_violation<<'\n'<<std::flush;
            if(total.radius_violation)throw std::runtime_error("RS failed within its guaranteed correction radius");
            if(clean && !gpu && (total.rejected || total.wrong))throw std::runtime_error("Noiseless known-channel FEC failure");
        }
    }
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
