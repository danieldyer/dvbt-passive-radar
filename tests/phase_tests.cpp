#include "receiver_clock.hpp"
#include "pipeline.hpp"
#include "fractional_resampler.hpp"
#include "gpu_frontend.hpp"
#include "gpu_receiver.hpp"
#include "gpu_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi,fs=8e6;
// Closed-form DFT of a finite complex sinusoid. Independent of the FFT/resampler code.
std::complex<double> dirichlet(double cycles) {
    cycles=std::remainder(cycles,1.0);
    if(std::abs(cycles)<1e-13)return {8192,0};
    return std::polar(1.0,std::numbers::pi*8191*cycles)*
        (std::sin(std::numbers::pi*8192*cycles)/std::sin(std::numbers::pi*cycles));
}
}
int main(int argc,char** argv) {
    try {
        const bool pipeline_test=argc>1 && std::string_view(argv[1])=="--pipeline";
        const bool streaming=pipeline_test || (argc>1 && std::string_view(argv[1])=="--receiver");
        const std::size_t nsymbols=streaming ? 1020 : 204, length=8704;
        constexpr double ratio=1.000005, delay=173.375, cfo=85.25, initial_phase=.731;
        constexpr double nco_offset=500000;
        constexpr double echo_delay=37.25, echo_doppler=29.0, echo_gain=.08;
        Modem modem;Tps mode;mode.mode=1;mode.constellation=0;mode.rate=2;mode.guard=1;
        std::mt19937 rng(9871);
        const auto packets=streaming ? 7000U : 1400U;
        std::vector<std::uint8_t> bytes(packets*188);
        for(auto& value:bytes)value=static_cast<std::uint8_t>(rng());
        for(std::size_t p=0;p<packets;++p)bytes[p*188]=(p%8 ? 0x47 : 0xb8);
        const auto carriers=modem.synthesize(bytes,mode,nsymbols);
        std::vector<Symbol> signalling(68);
        for(std::size_t frame=0;frame<3;++frame) {
            for(std::size_t s=0;s<68;++s)
                signalling[s].equalized.assign(carriers.begin()+static_cast<std::ptrdiff_t>((frame*68+s)*6817),carriers.begin()+static_cast<std::ptrdiff_t>((frame*68+s+1)*6817));
            Tps decoded;
            if(!modem.tps(signalling,decoded) || decoded.frame!=static_cast<int>(frame) || decoded.mode!=1 || decoded.rate!=2 || decoded.constellation!=0)
                throw std::runtime_error("TPS signalling/BCH/frame reconstruction failed");
        }
        std::vector<C> base(nsymbols*length),work(8192);
        Fft fft(8192);
        for(std::size_t s=0;s<nsymbols;++s) {
            std::fill(work.begin(),work.end(),C{});
            for(std::size_t k=0;k<6817;++k)work[(k+8192-3408)%8192]=carriers[s*6817+k];
            fft.transform(work,true);
            std::copy(work.end()-512,work.end(),base.begin()+static_cast<std::ptrdiff_t>(s*length));
            std::copy(work.begin(),work.end(),base.begin()+static_cast<std::ptrdiff_t>(s*length+512));
        }
        FractionalResampler source_interpolator;
        auto sample=[&](double at) {
            const auto integer=static_cast<std::int64_t>(std::floor(at));
            if(integer<47 || integer+49>=static_cast<std::int64_t>(base.size()))return C{};
            return source_interpolator.interpolate(base.data()+integer-47,at-integer);
        };
        std::vector<C> received(base.size());
        for(std::size_t n=0;n<received.size();++n) {
            const double u=(static_cast<double>(n)-delay)/ratio;
            const auto direct=sample(u);
            const auto echo=static_cast<float>(echo_gain)*sample(u-echo_delay)*static_cast<C>(std::polar(1.0,tau*echo_doppler*n/fs));
            received[n]=(direct+echo)*static_cast<C>(std::polar(1.0,tau*(cfo+nco_offset)*n/fs+initial_phase));
        }
        // Include the actual WAV -> absolute NCO -> centred filter implementation.
        // The 7:1 case is checked independently by dsp_tests; here 1:1 isolates the
        // phase bookkeeping through the entire receive chain with an analytic oracle.
        const auto path=std::filesystem::temp_directory_path()/
            ("dvbt-phase-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".wav");
        struct Cleanup { std::filesystem::path path;~Cleanup(){std::error_code e;std::filesystem::remove(path,e);} } cleanup{path};
        {
            std::ofstream file(path,std::ios::binary);
            auto put=[&](auto value){file.write(reinterpret_cast<const char*>(&value),sizeof(value));};
            const auto size=static_cast<std::uint32_t>(received.size()*8);
            file.write("RIFF",4);put(size+36);file.write("WAVEfmt ",8);put(std::uint32_t{16});
            put(std::uint16_t{3});put(std::uint16_t{2});put(std::uint32_t{8000000});put(std::uint32_t{64000000});
            put(std::uint16_t{8});put(std::uint16_t{32});file.write("data",4);put(size);
            file.write(reinterpret_cast<const char*>(received.data()),size);
        }
        received.clear();
        WaveReader wave(path);std::uint64_t origin;
        std::shared_ptr<SampleClock> clock;
        if(!streaming && argc>1) {
            const std::array centres{nco_offset};
            clock=std::make_shared<SampleClock>(fs,centres,3);clock->schedule(64,ratio);
            GpuChannelBank channel(wave,centres,0,1);channel.set_clock(clock);origin=channel.origin;
            for(;;) {const auto block=channel.next();if(block.empty())break;received.insert(received.end(),block[0].begin(),block[0].end());}
        } else if(!streaming) {
            Channelizer channel(wave,nco_offset,0,1);origin=channel.origin;
            for(;;) {const auto block=channel.next();if(block.empty())break;received.insert(received.end(),block.begin(),block.end());}
        }
        std::size_t checked=0,output_symbols=0;
        double error_energy=0,expected_energy=0,max_error=0;
        std::vector<double> clock_errors;
        auto verify=[&](Symbol symbol) {
            symbol.sample+=static_cast<double>(origin);
            // The common warp precedes the NCO, so the latter uses the corrected
            // coordinate. Recover both coordinates independently for this oracle.
            const double corrected_sample=symbol.sample,corrected_step=symbol.sample_step;
            if(clock) {
                const auto s=clock->at(symbol.sample);
                // The closed-form oracle assumes one affine clock segment per
                // FFT. Other windows remain processed, but use no false oracle.
                if(streaming && clock->at(symbol.sample+8192).start!=s.start)return;
                if(streaming && symbol.sample>fs*.7)
                    clock_errors.push_back(s.offset_at(symbol.sample)-(ratio-1)*symbol.sample);
                symbol.sample+=s.offset_at(symbol.sample);symbol.sample_step*=s.rate;
            }
            ++output_symbols;
            if(output_symbols<50 || output_symbols%13)return;
            const double u=(symbol.sample-delay)/ratio;
            const auto tx_symbol=static_cast<std::size_t>(std::floor(u/length));
            if(tx_symbol>=nsymbols)return;
            const double local=u-static_cast<double>(tx_symbol*length+512);
            const double tx_step=symbol.sample_step/ratio;
            if(local-echo_delay< -460 || local+8191*tx_step>8191)return;
            // Compare actual un-equalized output against the exact DFT, including its echo.
            // The expected signal knows the injected clock, CFO and Doppler, independently
            // of the synchronizer's estimates, and uses only its exported applied coordinates.
            for(int k_out=70;k_out<6750;k_out+=397) {
                const int j=k_out-3408;
                std::complex<double> expected{};
                for(int k_in=0;k_in<6817;++k_in) {
                    const int k=k_in-3408;
                    const auto amplitude=std::complex<double>(carriers[tx_symbol*6817+k_in])/8192.0;
                    for(int path=0;path<2;++path) {
                        const double offset=path ? echo_delay : 0;
                        const double frequency=cfo+(path ? echo_doppler : 0);
                        const double phase=tau*k*(local-offset)/8192+
                            tau*(frequency*symbol.sample+nco_offset*(symbol.sample-corrected_sample))/fs+initial_phase+symbol.phase;
                        const double increment=k*tx_step/8192+(frequency*symbol.sample_step+nco_offset*(symbol.sample_step-corrected_step))/fs-symbol.frequency_hz/fs;
                        expected+=amplitude*(path ? echo_gain : 1.0)*std::polar(1.0,phase)*dirichlet(increment-j/8192.0);
                    }
                }
                const double error=std::abs(std::complex<double>(symbol.measured[k_out])-expected);
                error_energy+=error*error;expected_energy+=std::norm(expected);max_error=std::max(max_error,error);++checked;
            }
        };
        if(streaming) {
            const std::array centres{nco_offset};
            clock=std::make_shared<SampleClock>(fs,centres,3);
            GpuChannelBank channel(wave,centres,0,double(wave.info.samples)/fs);channel.set_clock(clock);origin=channel.origin;
            const std::array guards{16U};
            if(pipeline_test) {
                GpuPipeline pipeline({{16U},channel.capacity(),origin,1,8,0,true},channel.context(),clock);
                auto consume=[&](const SpectralBatch& output) {
                    const auto* raw=static_cast<const C*>(output.frame.carriers.measured.mapped);
                    const auto* x=static_cast<const C*>(output.residual.reference.mapped);
                    const auto* h=static_cast<const C*>(output.residual.response.mapped);
                    const auto* cross=static_cast<const C*>(output.residual.cross.mapped);
                    const auto* valid=static_cast<const std::uint32_t*>(output.residual.valid.mapped);
                    for(std::size_t row=output.frame.first;row<output.frame.last;++row) {
                        const auto& w=output.frame.symbols[row].window;
                        Symbol symbol;symbol.sample=double(w.sample);symbol.sample_step=1;symbol.phase=w.phase;symbol.frequency_hz=w.frequency_hz;
                        symbol.measured.resize(6817);
                        for(unsigned k=0;k<6817;++k) {
                            const auto at=row*6817+k;
                            // Invert the exported product where its reference is
                            // valid. Feed the recovered measurement to the same
                            // independent continuous-waveform Doppler oracle.
                            symbol.measured[k]=valid[at] ? cross[at]/std::conj(x[at])+h[k]*x[at] : raw[at];
                            if(std::abs(symbol.measured[k]-raw[at])>1e-6)
                                throw std::runtime_error("Spectral handoff lost the underlying complex measurement");
                        }
                        verify(std::move(symbol));
                    }
                };
                for(;;){const auto input=channel.next_device();if(!input.count)break;pipeline.process(input,consume);}
                pipeline.finish(consume);
            } else {
                GpuReceiver receiver(modem.pilot_pattern(),guards,channel.capacity(),channel.context());
                for(;;) {
                    const auto input=channel.next_device();if(!input.count)break;
                    const auto output=receiver.process(input);observe_timing(*clock,output.timing,origin,1);auto symbols=receiver.download(output);
                    for(unsigned j=0;j<symbols.size();++j)if(output.symbols[j].ready)verify(std::move(symbols[j]));
                }
            }
            if(clock_errors.size()<100)throw std::runtime_error("Insufficient settled clock observations");
            double mean=0,error=0;
            for(double e:clock_errors)mean+=e/clock_errors.size();
            for(double e:clock_errors)error+=(e-mean)*(e-mean)/clock_errors.size();
            const double degrees=std::sqrt(error)*6656250/fs*360;
            std::cout<<"Closed-loop timing phase RMS across occupied band="<<degrees<<" degrees; final clock ppm="
                <<(clock->segments().back().rate-1)*1e6<<'\n';
            if(degrees>3)throw std::runtime_error("Streaming clock tracking exceeds three-degree phase budget");
        } else synchronize_samples(received,16,verify,clock,origin);
        const double relative=std::sqrt(error_energy/expected_energy);
        std::cout<<"Analytic phase-coordinate check: "<<checked<<" carriers, "<<output_symbols
            <<" symbols; relative RMS="<<relative<<" maximum error="<<max_error<<'\n';
        if(checked<100 || relative>8e-5 || max_error>4e-4)
            throw std::runtime_error("Resampling/NCO/FFT sample or phase bookkeeping disagrees with analytic Doppler/clock model");
        std::cout<<"PASS: 5 ppm sample-clock error, fractional delay, CFO and delayed Doppler echo retained\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
