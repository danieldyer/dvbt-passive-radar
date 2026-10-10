#include "modem.hpp"
#include "fractional_resampler.hpp"
#include "gpu_residual.hpp"
#include "cpu_fec.hpp"
#include <gnuradio/blocks/vector_source.h>
#include <gnuradio/blocks/vector_sink.h>
#include <gnuradio/dtv/dvbt_inner_coder.h>
#include <gnuradio/dtv/dvbt_reed_solomon_enc.h>
#include <gnuradio/dtv/dvbt_convolutional_interleaver.h>
#include <gnuradio/dtv/dvbt_bit_inner_interleaver.h>
#include <gnuradio/dtv/dvbt_symbol_inner_interleaver.h>
#include <gnuradio/dtv/dvbt_map.h>
#include <gnuradio/top_block.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <chrono>
#include <stdexcept>

using namespace radar;
void check(bool condition,const char* message) { if(!condition)throw std::runtime_error(message); }
int main(int argc,char** argv) {
    try {
        CpuFec fec;
        if(argc>1) {
            GpuResidual gpu;
            for(const std::size_t count : {320,8192,8192*3}) {
                std::vector<C> y(count),x(count),h(64);
                for(std::size_t i=0;i<count;++i) { y[i]={static_cast<float>(std::sin(i*.13)),static_cast<float>(std::cos(i*.01))};x[i]={static_cast<float>(std::cos(i*.17)),static_cast<float>(std::sin(i*.37))}; }
                for(std::size_t k=0;k<h.size();++k)h[k]={static_cast<float>(1+.2*std::cos(k)),static_cast<float>(.1*std::sin(k))};
                const auto actual=gpu.calculate(y,x,h);
                for(std::size_t i=0;i<count;++i)check(std::abs(actual[i]-(y[i]-h[i%h.size()]*x[i])*std::conj(x[i]))<2e-6,"GPU residual disagrees with CPU");
            }
            std::cout<<"GPU residual PASS\n";return 0;
        }
        const auto taps=Channelizer::prototype(56e6);
        double ripple=0,stop=0,phase=0;
        for(int bin=0;bin<=5600;++bin) {
            const double f=bin*5000.0;
            std::complex<double> response{};
            for(std::size_t k=0;k<taps.size();++k)
                response+=static_cast<double>(taps[k])*std::polar(1.0,-2*std::numbers::pi*f*(static_cast<double>(k)-512)/56e6);
            if(f<=3.328125e6) { ripple=std::max(ripple,std::abs(std::abs(response)-1));phase=std::max(phase,std::abs(std::arg(response))); }
            if(f>=4e6)stop=std::max(stop,std::abs(response));
        }
        std::cout<<"FIR passband ripple="<<ripple<<" phase="<<phase<<" stopband="<<20*std::log10(stop)<<" dB\n";
        check(ripple<1e-5 && phase<1e-10 && stop<1e-5,"FIR passband/alias rejection failed");
        FractionalResampler resampler;
        std::array<C,96> tone;
        double interp_error=0;
        for(double f : {-0.416015625,0.0,0.416015625}) for(int p=0;p<=100;++p) {
            const double mu=p/101.0;
            for(int i=0;i<96;++i)tone[i]=std::polar(1.0F,static_cast<float>(2*std::numbers::pi*f*(i-47)));
            interp_error=std::max(interp_error,std::abs(std::complex<double>(resampler.interpolate(tone.data(),mu))-std::polar(1.0,2*std::numbers::pi*f*mu)));
        }
        std::cout<<"Fractional resampler outer-carrier max error="<<interp_error<<'\n';
        check(interp_error<5e-5,"Fractional resampler amplitude/phase distortion");
        // Exercise WAV I/O, absolute NCO phase, compensated FIR delay, and overlap boundaries together.
        const auto temporary=std::filesystem::temp_directory_path()/
            ("dvbt-channelizer-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".wav");
        struct Cleanup { std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove(path,e);} } cleanup{temporary};
        for(const double offset : {4.625e6,-9.5e6}) {
            {
                constexpr std::uint32_t n=180000;
                std::ofstream wave(temporary,std::ios::binary);
                auto put=[&](auto v){wave.write(reinterpret_cast<const char*>(&v),sizeof(v));};
                wave.write("RIFF",4);put(std::uint32_t{36+n*8});wave.write("WAVEfmt ",8);put(std::uint32_t{16});
                put(std::uint16_t{3});put(std::uint16_t{2});put(std::uint32_t{56000000});put(std::uint32_t{448000000});
                put(std::uint16_t{8});put(std::uint16_t{32});wave.write("data",4);put(std::uint32_t{n*8});
                for(std::uint32_t i=0;i<n;++i) {
                    // Asymmetric in-band tones distinguish translation from reflection/conjugation.
                    const auto z=static_cast<C>(std::polar(1.0,2*std::numbers::pi*(offset+.5e6)*i/56e6)+
                        std::polar(.4,2*std::numbers::pi*(offset-1.25e6)*i/56e6+.2)+
                        std::polar(.5,2*std::numbers::pi*(offset+7e6)*i/56e6));
                    put(z);
                }
            }
            WaveReader wave(temporary);
            Channelizer channel(wave,offset,0,1);
            auto expected_sample=[](std::uint64_t position) {
                return std::polar(1.0,2*std::numbers::pi*.5e6*position/56e6)+
                    std::polar(.4,-2*std::numbers::pi*1.25e6*position/56e6+.2);
            };
            std::size_t sample=0,blocks=0;
            double channel_error=0;
            for(;;) {
                const auto block=channel.next();if(block.empty())break;++blocks;
                for(auto value:block) {
                    const auto expected=expected_sample(channel.origin+sample*7);
                    channel_error=std::max(channel_error,std::abs(std::complex<double>(value)-expected));++sample;
                }
            }
            check(blocks>=3 && channel_error<3e-6,"FFT channelizer NCO/overlap/sample origin mismatch");
            std::cout<<"Centred channelizer offset="<<offset<<" max complex error="<<channel_error<<" across "<<blocks<<" blocks\n";
            Channelizer sought(wave,offset,.002,.0005);
            const auto after_seek=sought.next();
            for(std::size_t i=0;i<after_seek.size();++i) {
                const auto expected=expected_sample(sought.origin+i*7);
                check(std::abs(std::complex<double>(after_seek[i])-expected)<3e-6,"Seeking reset the NCO phase or sample origin");
            }
        }
        // Finite-sample validation must cover both I/Q lanes and vector tails.
        for(const std::size_t at:{0U,15U,32U})for(unsigned component:{0U,1U})
            for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
                auto set=[&](float value) {
                    std::fstream change(temporary,std::ios::binary|std::ios::in|std::ios::out);
                    change.seekp(static_cast<std::streamoff>(44+at*8+component*4));
                    change.write(reinterpret_cast<const char*>(&value),sizeof(value));
                };
                set(bad);bool rejected=false;
                try {WaveReader invalid(temporary);std::array<C,33> values;invalid.read(0,values);}
                catch(const std::runtime_error& e){rejected=std::string_view(e.what())=="Non-finite input I/Q sample";}
                check(rejected,"WAV validation accepted non-finite I/Q");set(0);
            }
        Modem modem;
        // Compare the new slicer against a full 2-D distance search over GNU
        // Radio's independent constellation, including decision boundaries.
        double metric_error=0;
        for(int modulation=0;modulation<3;++modulation) {
            const int bits=2+2*modulation,points=1<<bits;
            std::vector<std::uint8_t> labels(6048);
            for(std::size_t i=0;i<labels.size();++i)labels[i]=static_cast<std::uint8_t>(i%points);
            auto input=gr::blocks::vector_source_b::make(labels,false,6048);
            auto mapper=gr::dtv::dvbt_map::make(6048,static_cast<gr::dtv::dvb_constellation_t>(modulation),gr::dtv::NH,gr::dtv::T8k,1);
            auto output=gr::blocks::vector_sink_c::make(6048);
            auto graph=gr::make_top_block("independent_hard_qam_test");
            graph->connect(input,0,mapper,0);graph->connect(mapper,0,output,0);graph->run();
            const auto mapped=output->data();check(mapped.size()>=static_cast<std::size_t>(points),"Independent QAM mapper produced too little data");
            auto slice=[&](C z){return bits==2 ? hard_qam<2>(z) : bits==4 ? hard_qam<4>(z) : hard_qam<6>(z);};
            for(int re=-64;re<=64;++re)for(int im=-64;im<=64;++im) {
                const C z{re*.025F,im*.025F};float nearest=std::numeric_limits<float>::infinity();
                for(int label=0;label<points;++label)nearest=std::min(nearest,std::norm(z-mapped[label]));
                const auto actual=slice(z);unsigned label=0;
                for(int bit=0;bit<bits;++bit)label|=((actual>>bit)&1U)<<(bits-1-bit);
                metric_error=std::max(metric_error,double(std::norm(z-mapped[label])-nearest));
            }
            for(auto x:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
                check(slice({x,0})==255 && slice({0,x})==255,"Non-finite QAM was not erased");
        }
        check(metric_error<2e-6,"Hard QAM slicer disagrees with independent nearest-point search");
        std::cout<<"Independent hard QAM distance error="<<metric_error<<'\n';
        Tps mode;mode.mode=1;mode.constellation=2;mode.rate=2;mode.guard=1;
        constexpr std::size_t count=16;
        std::vector<std::uint8_t> bytes(400*188);
        std::mt19937 rng(321);
        for(auto& x:bytes)x=static_cast<std::uint8_t>(rng());
        for(std::size_t i=0;i<400;++i)bytes[i*188]=(i%8==0 ? 0xb8 : 0x47);
        const auto tx=modem.synthesize(bytes,mode,count);
        // Independent implementation: compare every transmitted payload point to GNU Radio's encoder.
        auto source=gr::blocks::vector_source_b::make(bytes,false,188*8);
        auto rs=gr::dtv::dvbt_reed_solomon_enc::make(2,8,0x11d,255,239,8,51,8);
        auto outer=gr::dtv::dvbt_convolutional_interleaver::make(136,12,17);
        auto inner=gr::dtv::dvbt_inner_coder::make(1,6048,gr::dtv::MOD_64QAM,gr::dtv::NH,gr::dtv::C3_4);
        auto bit_interleaver=gr::dtv::dvbt_bit_inner_interleaver::make(6048,gr::dtv::MOD_64QAM,gr::dtv::NH,gr::dtv::T8k);
        auto symbol_interleaver=gr::dtv::dvbt_symbol_inner_interleaver::make(6048,gr::dtv::T8k,1);
        auto mapper=gr::dtv::dvbt_map::make(6048,gr::dtv::MOD_64QAM,gr::dtv::NH,gr::dtv::T8k,1);
        auto sink=gr::blocks::vector_sink_c::make(6048);
        auto graph=gr::make_top_block("independent_encoder_test");
        graph->connect(source,0,rs,0);graph->connect(rs,0,outer,0);graph->connect(outer,0,inner,0);
        graph->connect(inner,0,bit_interleaver,0);graph->connect(bit_interleaver,0,symbol_interleaver,0);
        graph->connect(symbol_interleaver,0,mapper,0);graph->connect(mapper,0,sink,0);graph->run();
        const auto independent=sink->data();
        check(independent.size()>=count*6048,"GNU Radio independent TX produced too little data");
        for(std::size_t s=0;s<count;++s) {
            const auto carriers=modem.data_carriers(static_cast<int>(s%4));
            for(std::size_t k=0;k<6048;++k)
                check(std::abs(independent[s*6048+k]-tx[s*6817+carriers[k]])<1e-6,"Independent GNU Radio TX mapping disagrees");
        }
        std::cout<<"Independent GNU Radio TX agrees on all payload carriers\n";
        std::vector<Symbol> symbols(count);
        for(std::size_t s=0;s<count;++s) {
            symbols[s].modulo4=static_cast<int>(s%4);
            symbols[s].equalized.assign(tx.begin()+static_cast<std::ptrdiff_t>(s*6817),tx.begin()+static_cast<std::ptrdiff_t>((s+1)*6817));
            symbols[s].measured=symbols[s].equalized;symbols[s].noise_variance=.01F;
        }
        auto reconstructed=modem.reconstruct(symbols,mode,nullptr,&fec);
        std::size_t compared=0;
        for(std::size_t s=1;s+2<count;++s)for(auto k:modem.data_carriers(static_cast<int>(s%4))) {
            const auto i=s*6817+k;
            check(reconstructed.valid[i],"Valid noiseless payload was rejected");
            check(std::abs(reconstructed.carriers[i]-tx[i])<1e-6,"FEC reconstruction changed noiseless payload");
            ++compared;
        }
        std::cout<<"Reconstruction: "<<reconstructed.packets<<" packets, "<<reconstructed.bad_packets<<" failures; "<<compared<<" exact QAM symbols\n";
        check(reconstructed.bad_packets<12,"RS roundtrip failed");
        // Destroy two symbols: FEC failures must propagate to validity masks and retain positions.
        for(std::size_t s=6;s<8;++s)for(auto& z:symbols[s].equalized)z=std::polar(1.0F,static_cast<float>(rng())/static_cast<float>(rng.max())*6.2831853F);
        auto damaged=modem.reconstruct(symbols,mode,nullptr,&fec);
        check(damaged.bad_packets>0 && damaged.carriers.size()==tx.size(),"Uncorrectable FEC must be flagged without time-axis compaction");
        check(std::count(damaged.valid.begin(),damaged.valid.end(),0)>std::count(reconstructed.valid.begin(),reconstructed.valid.end(),0),"FEC validity was not propagated");
        for(int constellation=0;constellation<3;++constellation)for(int rate=0;rate<5;++rate) {
            mode.constellation=constellation;mode.rate=rate;
            const auto encoded=modem.synthesize(bytes,mode,count);
            for(std::size_t s=0;s<count;++s) {
                symbols[s].equalized.assign(encoded.begin()+static_cast<std::ptrdiff_t>(s*6817),encoded.begin()+static_cast<std::ptrdiff_t>((s+1)*6817));
                symbols[s].measured=symbols[s].equalized;symbols[s].noise_variance=.01F;
            }
            const auto decoded=modem.reconstruct(symbols,mode,nullptr,&fec);
            for(std::size_t s=4;s+4<count;++s)for(auto k:modem.data_carriers(static_cast<int>(s%4))) {
                const auto i=s*6817+k;
                check(decoded.valid[i] && std::abs(decoded.carriers[i]-encoded[i])<1e-6,"DVB-T constellation/code-rate roundtrip failed");
            }
        }
        std::cout<<"All 15 non-hierarchical constellation/code-rate combinations pass\n";
        std::vector<C> x(6817, C{1,0}),y(6817);
        for(std::size_t k=0;k<6817;++k)y[k]=C{2,.3F}*std::exp(C{.2F,0}*std::polar(1.0F,static_cast<float>(-2*std::numbers::pi*(static_cast<double>(k)-3408)/8192)));
        auto fit=fit_minimum_phase(y,x,6817,1,32);
        std::cout<<"Minimum-phase fit relative error="<<fit.relative_error<<'\n';
        check(fit.relative_error<1e-6,"Minimum-phase fit failed");
        for(std::size_t k=0;k<6817;++k)y[k]=std::polar(1.0F,static_cast<float>(-2*std::numbers::pi*(static_cast<double>(k)-3408)*8/8192));
        const auto delay_fit=fit_minimum_phase(y,x,6817,8,32);
        C mean{};for(auto value:delay_fit.response)mean+=value;
        mean/=6817.0F;
        for(auto value:delay_fit.response)check(std::abs(value-mean)<1e-4,"Minimum-phase magnitude estimator fitted an all-pass delay");
        std::cout<<"PASS\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n';return 1; }
}
