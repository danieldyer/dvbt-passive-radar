#include "gpu_reconstruction.hpp"
#include "gpu_ofdm.hpp"
#include "gpu_compute.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <random>

using namespace radar;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(int argc,char** argv) {
    try {
        const bool benchmark=argc>1 && std::string_view(argv[1])=="--benchmark";
        const unsigned count=benchmark ? 1020 : 136,packet_count=benchmark ? 22000 : 3000;
        Modem modem;std::mt19937 rng(91378);
        std::vector<std::uint8_t> bytes(packet_count*188);
        for(auto& b:bytes)b=static_cast<std::uint8_t>(rng());
        for(unsigned p=0;p<packet_count;++p)bytes[p*188]=p%8 ? 0x47 : 0xb8;
        auto context=std::make_shared<gpu::Context>();
        auto samples=gpu::make_buffer(context->gpu,count*6817*sizeof(C));
        auto stats=gpu::make_buffer(context->gpu,count*2*sizeof(C));
        auto* x=static_cast<C*>(context->gpu.device->mapMemory(*samples.memory,0,count*6817*sizeof(C)));
        auto* v=static_cast<C*>(context->gpu.device->mapMemory(*stats.memory,0,count*2*sizeof(C)));
        const gpu::BufferView values{context,*samples.handle,count*6817*sizeof(C),x};
        const OfdmBatch input{values,values,{context,*stats.handle,count*2*sizeof(C),v},count,2};
        CpuFec fec;GpuReconstruction gpu(context);ComputeTimings timing;
        if(benchmark)std::cout<<"qam,gpu_reconstruction_ms_per_input_second\n";
        for(int modulation=0;modulation<3;++modulation)for(int rate=0;rate<5;++rate) {
            if(benchmark && rate!=2)continue;
            Tps mode;mode.mode=1;mode.constellation=modulation;mode.rate=rate;mode.guard=1;
            const auto tx=modem.synthesize(bytes,mode,count);std::copy(tx.begin(),tx.end(),x);
            std::vector<Symbol> symbols(count);
            for(unsigned s=0;s<count;++s) {
                symbols[s].equalized.assign(tx.begin()+s*6817,tx.begin()+(s+1)*6817);
                symbols[s].measured=symbols[s].equalized;symbols[s].modulo4=s%4;symbols[s].noise_variance=.01F;
                v[s*2]={std::bit_cast<float>(s%4),1};v[s*2+1]={.01F,1};
            }
            std::vector<Tps> frames(count/68);
            for(unsigned j=0;j<frames.size();++j)check(modem.tps(std::span{symbols}.subspan(j*68,68),frames[j]),"Synthetic TPS did not pass BCH");
            const auto decoded=fec.decode(input,mode,modem);
            if(benchmark) {
                gpu.reconstruct(decoded,frames,modem);ComputeTimings warm;
                for(unsigned repeat=0;repeat<5;++repeat)gpu.reconstruct(decoded,frames,modem,&warm);
                std::cout<<(1U<<(2+2*modulation))<<','<<warm.gpu/5/(count*8704.0/8e6)*1000<<'\n';
                continue;
            }
            const auto actual=gpu.download(gpu.reconstruct(decoded,frames,modem,&timing));
            const auto reference=modem.reconstruct(symbols,mode);
            check(actual.valid==reference.valid,"GPU reconstruction mask differs from CPU reference");
            for(unsigned i=0;i<tx.size();++i)if(actual.valid[i])
                check(std::abs(actual.carriers[i]-tx[i])<1e-6 && actual.carriers[i]==reference.carriers[i],
                    "GPU re-encoding or QAM/pilot/TPS remapping changed a valid carrier");
            // Invalidate entire RS words, including the first and last. Build
            // an independent serial bit-validity oracle, then forward-interleave
            // its transmitted bits. This exercises byte and trellis boundaries.
            auto words=std::vector(decoded.packets.begin(),decoded.packets.end());
            const FecBatch erased{words,decoded.count,decoded.trellis_bits};
            for(unsigned p=0;p<decoded.count;++p)if(p%17==0 || p+1==decoded.count)words[p].valid=0;
            std::vector<std::uint8_t> valid_bytes(decoded.trellis_bits/8);
            for(unsigned p=0;p<decoded.count;++p)for(unsigned j=0;j<204;++j) {
                const unsigned at=2244+p*204+j;
                valid_bytes[at-(11-at%12)*204]=words[p].valid;
            }
            const auto pattern=fec::puncture_pattern(rate);std::vector<std::uint8_t> transmitted;
            unsigned history=0;
            for(unsigned t=0;t<decoded.trellis_bits;++t) {
                history=((history<<1)|valid_bytes[t/8])&127;
                for(unsigned b=0;b<2;++b)if(pattern[(2*t+b)%pattern.size()])transmitted.push_back(history==127);
            }
            auto mask=std::vector<std::uint8_t>(count*6817,1);const auto forward=modem.payload_bit_indices(mode);
            const unsigned m=2+2*modulation;
            for(unsigned s=0;s<count;++s)for(unsigned b=0;b<6048*m;++b)
                mask[s*6817+forward[(s%4)*6048*m+b]/8]&=transmitted[s*6048*m+b];
            const auto damaged=gpu.download(gpu.reconstruct(erased,frames,modem));
            check(damaged.valid==mask,"Erased RS word failed to invalidate every dependent QAM carrier");
            check(std::count(mask.begin(),mask.end(),0)>std::count(actual.valid.begin(),actual.valid.end(),0),"Erasure test did not exercise interior packets");
            auto broken=frames;broken[1].bits[53]^=1;bool rejected=false;
            try{gpu.reconstruct(decoded,broken,modem);}catch(const std::invalid_argument&){rejected=true;}
            check(rejected,"GPU reconstruction accepted TPS without BCH validation");
            std::cout<<"QAM="<<(1U<<m)<<" rate="<<rate<<": carriers, TPS, boundaries and erasure masks PASS\n";
        }
        context->gpu.device->unmapMemory(*stats.memory);context->gpu.device->unmapMemory(*samples.memory);
        if(!benchmark)std::cout<<"Reconstruction GPU seconds over all fifteen cases="<<timing.gpu<<'\n';
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
