#include "gpu_frontend.hpp"
#include "receiver_clock.hpp"
#include "gpu_receiver.hpp"
#include "gpu_compute.hpp"
#include "cpu_fec.hpp"
#include "gpu_frames.hpp"
#include "gpu_reconstruction.hpp"
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv) {
    try {
        Modem modem;const auto pattern=modem.pilot_pattern();
        if(argc>1) {
            WaveReader wave(argv[1]);const std::array centres{4625000.0};
            GpuChannelBank bank(wave,centres,0,argc>2 ? std::stod(argv[2]) : 2);
            auto clock=std::make_shared<SampleClock>(56e6,centres);bank.set_clock(clock);
            const std::array guards{16U};
            GpuReceiver receiver(pattern,guards,bank.capacity(),bank.context());
            std::vector<Symbol> symbols;unsigned seen=0;ComputeTimings timing;
            for(;;) {
                const auto input=bank.next_device();if(!input.count)break;
                auto output=receiver.process(input,&timing);observe_timing(*clock,output.timing,bank.origin,bank.decimation);auto host=receiver.download(output);
                for(unsigned j=0;j<host.size();++j) {
                    ++seen;
                    if(output.symbols[j].ready)symbols.push_back(std::move(host[j]));
                }
            }
            Tps tps;unsigned first=0;
            while(first+68<=symbols.size() && !modem.tps(std::span{symbols}.subspan(first,68),tps))++first;
            check(first+68<=symbols.size(),"Recorded stream failed TPS acquisition");
            CpuFec fec;std::size_t packets=0,failed=0;
            while(first+68<=symbols.size()) {
                const auto count=std::min<std::size_t>(1020,(symbols.size()-first)/68*68);
                check(modem.tps(std::span{symbols}.subspan(first,68),tps),"Recorded stream lost TPS alignment");
                const auto decoded=modem.reconstruct(std::span{symbols}.subspan(first,count),tps,nullptr,&fec);
                packets+=decoded.packets;failed+=decoded.bad_packets;first+=count;
            }
            std::cout<<"Recording: seen="<<seen<<" ready="<<symbols.size()<<" TPS="<<tps.constellation<<'/'<<tps.rate
                <<" packets="<<packets<<" failed="<<failed<<" GPU seconds="<<timing.gpu<<'\n';
            check(failed==0,"Recorded streaming GPU receiver failed RS");
            return 0;
        }
        constexpr unsigned count=1360,length=8704,maximum=131071;
        constexpr std::uint64_t epoch=(1ULL<<40)+37;
        constexpr double tau=2*std::numbers::pi;
        const std::array frequencies{35585.25,-23137.125};
        const std::array delays{73U,221U};
        Tps mode;mode.mode=1;mode.constellation=0;mode.rate=2;mode.guard=1;
        std::mt19937 rng(9581);std::vector<std::uint8_t> bytes(8000*188);
        for(auto& b:bytes)b=static_cast<std::uint8_t>(rng());
        for(unsigned j=0;j<8000;++j)bytes[j*188]=j%8 ? 0x47 : 0xb8;
        const auto tx=modem.synthesize(bytes,mode,count);
        std::vector<C> base(count*length),work(8192);Fft fft(8192);
        for(unsigned s=0;s<count;++s) {
            std::fill(work.begin(),work.end(),C{});
            for(unsigned k=0;k<6817;++k)work[(k+8192-3408)%8192]=tx[s*6817+k];
            fft.transform(work,true);
            std::copy(work.end()-512,work.end(),base.begin()+s*length);
            std::copy(work.begin(),work.end(),base.begin()+s*length+512);
        }
        std::array<std::vector<C>,2> source;
        for(unsigned c=0;c<2;++c) {
            source[c].resize(base.size()+delays[c]);
            for(unsigned n=delays[c];n<source[c].size();++n)
                source[c][n]=base[n-delays[c]]*static_cast<C>(std::polar(1.0-.2*c,tau*frequencies[c]*n/8e6+.71*c));
        }
        auto context=std::make_shared<gpu::Context>();
        auto storage=gpu::make_buffer(context->gpu,2*maximum*sizeof(C));
        auto* mapped=static_cast<C*>(context->gpu.device->mapMemory(*storage.memory,0,2*maximum*sizeof(C)));
        const gpu::BufferView input{context,*storage.handle,2*maximum*sizeof(C),mapped};
        const std::array guards{16U,16U};
        GpuReceiver receiver(pattern,guards,maximum,context);
        GpuFrames frames(guards,context);CpuFec resident_fec;GpuReconstruction reconstruction(context);
        std::array<std::uint64_t,2> last_emitted{};std::array<unsigned,2> emitted{},batches{};
        auto consume=[&](const FrameBatch& frame) {
            const auto decoded=resident_fec.decode(frame.carriers,frame.frames.front(),modem);
            const auto reference=reconstruction.reconstruct(decoded,frame.frames,modem);
            const auto* values=static_cast<const C*>(reference.carriers.mapped);
            const auto* valid=static_cast<const std::uint32_t*>(reference.valid.mapped);
            for(unsigned row=0;row<frame.symbols.size();++row) {
                const auto source=(frame.symbols[row].window.sample-epoch-delays[frame.channel])/length;
                check(source<count,"Resident frame lost symbol coordinates");
                for(unsigned k=0;k<6817;++k)if(valid[row*6817+k])
                    check(std::abs(values[row*6817+k]-tx[source*6817+k])<1e-6,"Resident receiver/FEC/reconstruction changed a valid carrier");
            }
            for(std::size_t row=frame.first;row<frame.last;++row) {
                const auto sample=frame.symbols[row].window.sample;
                if(emitted[frame.channel])check(sample-last_emitted[frame.channel]==length,"Frame overlap duplicated or dropped emitted symbols");
                last_emitted[frame.channel]=sample;++emitted[frame.channel];
            }
            ++batches[frame.channel];
        };
        std::array<std::vector<Symbol>,2> output;
        double max_error=0;unsigned block=0;
        for(std::size_t first=0;first<base.size();) {
            // Include blocks smaller than one symbol and changes in dispatch shape.
            const auto n=std::min<std::size_t>(block%3==0 ? 137 : block%3==1 ? 65537 : maximum,base.size()-first);
            for(unsigned c=0;c<2;++c)std::copy_n(source[c].data()+first,n,mapped+c*maximum);
            const auto batch=receiver.process({input,epoch+first,n,2,maximum});
            frames.append(batch,consume);
            auto host=receiver.download(batch);
            for(unsigned row=0;row<host.size();++row) {
                const auto& meta=batch.symbols[row];const auto& w=meta.window;
                const auto at=w.sample-epoch;const unsigned c=meta.channel;
                for(unsigned j=0;j<8192;++j)work[j]=source[c][at+j]*static_cast<C>(std::polar(1.0,
                    std::remainder(w.phase,tau)-tau*w.frequency_hz*j/8e6));
                fft.transform(work,false);
                double error=0,power=0;
                for(unsigned k=0;k<6817;++k) {
                    const C expected=work[(k+8192-3408)%8192];
                    error+=std::norm(expected-host[row].measured[k]);power+=std::norm(expected);
                }
                max_error=std::max(max_error,std::sqrt(error/power));
                check(host[row].sample_step==1,"Receiver independently retimed a channel");
                if(meta.ready)output[c].push_back(std::move(host[row]));
            }
            first+=n;++block;
        }
        frames.finish(consume);
        for(unsigned c=0;c<2;++c)check(batches[c]>=2 && emitted[c]>=1020,"Resident frame ring/overlap path was not exercised");
        check(max_error<3e-6,"Streaming GPU FFT/phase metadata disagrees with CPU waveform oracle");
        CpuFec fec;
        for(unsigned c=0;c<2;++c) {
            check(output[c].size()>200,"Streaming receiver did not retain lock");
            check(output[c].front().discontinuity,"Initial acquisition lacks a discontinuity");
            for(unsigned j=1;j<output[c].size();++j) {
                check(!output[c][j].discontinuity,"Unexpected reacquisition in clean stream");
                check(output[c][j].sample-output[c][j-1].sample==length,"Symbol coordinates jump across batches");
            }
            Tps tps;unsigned first=0;
            while(first+68<=output[c].size() && !modem.tps(std::span{output[c]}.subspan(first,68),tps))++first;
            check(first+68<=output[c].size(),"Streaming receiver failed TPS BCH");
            const auto n=std::min<std::size_t>(1020,(output[c].size()-first)/68*68);
            const auto decoded=modem.reconstruct(std::span{output[c]}.subspan(first,n),tps,nullptr,&fec);
            check(decoded.bad_packets==0,"Streaming receiver failed FEC");
            std::cout<<"Channel "<<c<<": "<<output[c].size()<<" ready symbols, "<<decoded.packets<<" RS packets passed\n";
        }
        // A missing input interval cannot be represented as phase continuity.
        for(unsigned c=0;c<2;++c)std::copy_n(source[c].data(),maximum,mapped+c*maximum);
        const auto gap=receiver.process({input,epoch+base.size()+1000,maximum,2,maximum});
        for(const auto& row:gap.symbols)check(!row.ready && row.segment>0,"Gap reused stale acquisition/phase state");
        std::array<unsigned,2> after_gap{};
        auto consume_after_gap=[&](const FrameBatch& frame) {
            check(frame.segment>1 && frame.symbols[frame.first].discontinuity,
                  "Gap or failed TPS reused stale frame continuity");
            const auto packets=resident_fec.decode(frame.carriers,frame.frames.front(),modem).packets;
            for(const auto& packet:packets)check(packet.valid,"Reacquired resident frame failed FEC");
            after_gap[frame.channel]+=frame.last-frame.first;
        };
        frames.append(gap,consume_after_gap);
        const auto gap_epoch=epoch+base.size()+1000;
        for(std::size_t first=maximum;first<410*length;) {
            const auto n=std::min<std::size_t>(maximum,410*length-first);
            for(unsigned c=0;c<2;++c)std::copy_n(source[c].data()+first,n,mapped+c*maximum);
            auto batch=receiver.process({input,gap_epoch+first,n,2,maximum});
            // Corrupt signalling only in one channel; its invalid frame must
            // split that channel's segment without disturbing the other one.
            for(auto& row:batch.symbols)if(row.channel==0 &&
                (row.window.sample-gap_epoch-delays[0])/length==300)row.tps=-row.tps;
            frames.append(batch,consume_after_gap);first+=n;
        }
        frames.finish(consume_after_gap);
        check(after_gap[0]==136 && after_gap[1]==204,"TPS failure lost extra frames or crossed channel boundaries");
        context->gpu.device->unmapMemory(*storage.memory);
        std::cout<<"Independent CFOs, arbitrary blocks, large epoch, gap: PASS; phase/FFT RMS error="<<max_error<<'\n';
    } catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
