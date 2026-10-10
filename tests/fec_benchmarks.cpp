#include "modem.hpp"
#include "cpu_fec.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
using namespace radar;
int main(int argc,char** argv){try {
    const bool curve=argc>1 && std::string_view(argv[1])=="--curve";
    const unsigned count=curve ? 204 : 1020;
    Modem modem;CpuFec fec;std::mt19937 rng(1092381);std::normal_distribution<float> normal;
    std::vector<std::uint8_t> bytes(25000*188);for(auto& b:bytes)b=rng();for(unsigned i=0;i<25000;++i)bytes[i*188]=i%8 ? 0x47 : 0xb8;
    std::cout<<"qam,esn0_db,packets,rejected,wrong,demap_ms_per_s,viterbi_ms_per_s,rs_ms_per_s\n";
    for(int qam=0;qam<3;++qam) {
        Tps mode;mode.mode=1;mode.constellation=qam;mode.rate=2;mode.guard=2;
        const auto tx=modem.synthesize(bytes,mode,count);
        for(double snr:curve ? std::vector<double>{6,8,10,12,20} : std::vector<double>{20}) {
            snr+=qam*6;const float sigma=std::sqrt(std::pow(10.,-snr/10)/2);
            std::vector<Symbol> symbols(count);
            for(unsigned s=0;s<count;++s){auto& r=symbols[s];r.modulo4=s%4;r.equalized.assign(tx.begin()+s*6817,tx.begin()+(s+1)*6817);for(auto& z:r.equalized)z+=sigma*C{normal(rng),normal(rng)};}
            fec.decode(symbols,mode,modem);FecTimings timing;FecBatch decoded;
            const unsigned repetitions=curve ? 1 : 5;
            for(unsigned run=0;run<repetitions;++run)decoded=fec.decode(symbols,mode,modem,&timing);
            unsigned fail=0,wrong=0;
            for(unsigned p=16;p+16<decoded.count;++p){const auto& r=decoded.packets[p];fail+=!r.valid;wrong+=r.valid && !std::equal(r.bytes.begin(),r.bytes.begin()+188,bytes.begin()+p*188);}
            const double scale=1000/(count*8704/8e6)/repetitions;
            std::cout<<(1<<(2+2*qam))<<','<<snr<<','<<decoded.count-32<<','<<fail<<','<<wrong<<','<<timing.demap*scale<<','<<timing.viterbi*scale<<','<<timing.reed_solomon*scale<<'\n'<<std::flush;
            if(wrong || (!curve && fail))throw std::runtime_error("Known-truth FEC qualification failed");
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
