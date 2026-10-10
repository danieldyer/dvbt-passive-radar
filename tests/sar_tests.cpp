#include "gpu_frontend.hpp"
#include "sar_handoff.hpp"
#include "fractional_resampler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
namespace {
constexpr double tau=2*std::numbers::pi,fo=8e6;
constexpr unsigned bins=6817,nfft=8192,length=8704;
void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<class F> void rejects(F f,const char* message){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,message);}
struct Temporary {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("dvbt-sar-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Temporary(){std::error_code e;std::filesystem::remove(path,e);}
};
template<class T> T read(std::istream& file){T x{};file.read(reinterpret_cast<char*>(&x),sizeof(x));check(bool(file),"Truncated archive");return x;}
void coordinates() {
    SarCapture capture{24'000'000,3,(1ULL<<42)+9,600e6,3,{{596e6+35.25,-4e6,16},{604e6-71.125,4e6,16}}};
    FrameBatch frame;frame.channel=1;frame.segment=4;frame.first=0;frame.last=2;frame.symbols.resize(2);frame.frames.resize(1);
    frame.carriers.count=2;
    for(unsigned i=0;i<2;++i) {
        auto& s=frame.symbols[i];s.window.sample=7000+i*length;s.window.phase=.7+i*.03;s.window.frequency_hz=-784.625;
        s.channel=1;s.segment=4;s.modulo4=i;
    }
    std::vector<C> y(2*bins,C{.4f,-.7f}),x(2*bins,C{1,-1}),h(bins,C{.8f,.3f});
    std::vector<std::uint32_t> valid(2*bins,1);valid[17]=0;
    auto view=[](auto& v){return gpu::BufferView{{},{},v.size()*sizeof(v[0]),v.data()};};
    frame.carriers.measured=view(y);
    SpectralBatch spectra{frame,{}, {view(x),view(valid),2},{view(h),{},2,128.375,.001},{{},{},{},{},{},2,128.375}};
    const auto crossing=capture.origin+frame.symbols[0].window.sample*3+1000;
    std::vector<ClockSegment> clock{{0,0,1},{crossing,0,1.000005}};
    SarBatch batch(capture,spectra,clock);
    check(batch.corrected_adc(0)==capture.origin+21000,"Lost integer corrected coordinate");
    check(batch.original_adc(0,100)==double(capture.origin+21300),"Clock segment chosen too early");
    check(std::abs(batch.original_adc(0,8000)-(double(capture.origin+45000)+23000*.000005))<.001,
          "Clock transition inside FFT window was lost");
    check(batch.carrier_rf_hz(3408)==604e6-71.125,"Receiver CFO contaminated physical RF frequency");
    check(batch.carrier_rf_hz(0)==604e6-71.125-3328125,"Wrong signed carrier frequency");
    rejects([&]{auto bad=clock;bad[1].offset=1;SarBatch ignored(capture,spectra,bad);},"Accepted discontinuous clock");
    rejects([&]{batch.corrected_adc(2);},"Accepted unavailable row");
    rejects([&]{batch.carrier_rf_hz(6817);},"Accepted out-of-band carrier");
    Temporary file;
    SarWriter writer(file.path,capture);writer.write(batch);writer.finish();
    rejects([&]{SarWriter duplicate(file.path,capture);},"Overwrote an existing archive");
    // Parse independently with byte counts specified in docs, not C++ structs.
    std::ifstream input(file.path,std::ios::binary);char magic[8];input.read(magic,8);
    check(std::memcmp(magic,"DVBTSAR1",8)==0,"Wrong archive magic");
    check(read<std::uint32_t>(input)==24'000'000,"Wrong archived sample rate");
    input.seekg(48+2*24);char tag[4];input.read(tag,4);
    check(std::memcmp(tag,"CLCK",4)==0 && read<std::uint64_t>(input)==52,"Wrong clock record layout");
    check(read<std::uint32_t>(input)==2,"Missing clock segments");
    check(read<std::uint64_t>(input)==0 && read<double>(input)==0 && read<double>(input)==1,"Wrong initial clock");
    check(read<std::uint64_t>(input)==crossing && read<double>(input)==0 && read<double>(input)==1.000005,"Wrong clock update");
    input.read(tag,4);check(std::memcmp(tag,"SPEC",4)==0,"Missing spectra");
    const auto size=read<std::uint64_t>(input),begin=std::uint64_t(input.tellg());
    check(read<std::uint32_t>(input)==1 && read<std::uint32_t>(input)==4 && read<std::uint32_t>(input)==2,"Wrong batch identity");
    input.seekg(begin+40+2*32+92);
    const auto response=read<C>(input);check(response==h[0],"Response serialization changed complex order");
    input.seekg(begin+40+2*32+92+8*bins);
    check(read<C>(input)==y[0],"Measurement serialization changed complex order");
    input.seekg(begin+40+2*32+92+8*bins+16*2*bins+4*17);
    check(read<std::uint32_t>(input)==0,"Invalid reference became valid");
    input.seekg(begin+size);input.read(tag,4);
    check(std::memcmp(tag,"DONE",4)==0 && read<std::uint64_t>(input)==8 && read<std::uint64_t>(input)==1,"Archive framing/termination failed");
    std::cout<<"Physical coordinates, large epoch, in-window warp, RF/CFO distinction, archive: PASS\n";
}
struct ConsumerFailure : std::runtime_error {ConsumerFailure():std::runtime_error("injected consumer failure") {}};
void multichannel(bool bounded=false,bool failure=false,bool mixed=false) {
    constexpr unsigned symbols=1360,fs=24'000'000;
    constexpr double ratio=1.000005,drift=.0000002,receiver_lo_error=713.5;
    const std::array<double,2> offsets{-4e6,4e6},tx_offset{35.25,-71.125},epoch{73.375,3521.125},phase{.42,-1.12};
    const std::array<double,2> delay_samples{37.25,700.75},doppler{29,-43};
    const C reflectivity=std::polar(.04f,.23f);
    SarCapture capture{fs,3,0,600e6,3,{{600e6+offsets[0]+tx_offset[0],offsets[0],16},
                                    {600e6+offsets[1]+tx_offset[1],offsets[1],16}}};
    Modem modem;Tps mode;mode.mode=1;mode.constellation=0;mode.rate=2;mode.guard=1;
    std::mt19937 rng(398421);
    std::array<std::vector<C>,2> transmitted,base;
    Fft fft(nfft);std::vector<C> work(nfft);
    for(unsigned c=0;c<2;++c) {
        auto channel_mode=mode;if(mixed && c==1)channel_mode.constellation=2;
        const unsigned packet_count=8000U*(1+unsigned(channel_mode.constellation));
        std::vector<std::uint8_t> bytes(packet_count*188);
        for(auto& b:bytes)b=static_cast<std::uint8_t>(rng());
        for(unsigned p=0;p<packet_count;++p)bytes[p*188]=p%8 ? 0x47 : 0xb8;
        transmitted[c]=modem.synthesize(bytes,channel_mode,symbols);base[c].resize(symbols*length);
        for(unsigned s=0;s<symbols;++s) {
            std::fill(work.begin(),work.end(),C{});
            for(unsigned k=0;k<bins;++k)work[(k+nfft-3408)%nfft]=transmitted[c][s*bins+k];
            fft.transform(work,true);
            std::copy(work.end()-512,work.end(),base[c].begin()+s*length);
            std::copy(work.begin(),work.end(),base[c].begin()+s*length+512);
        }
    }
    FractionalResampler interpolation;
    auto sample=[&](std::span<const C> data,double at) {
        const auto index=std::int64_t(std::floor(at));
        if(index<47 || index+49>=std::int64_t(data.size()))return C{};
        return interpolation.interpolate(data.data()+index-47,at-index);
    };
    const auto raw_count=std::uint32_t(3*symbols*length);
    Temporary file;
    {
        std::ofstream out(file.path,std::ios::binary);auto put=[&](auto v){out.write(reinterpret_cast<const char*>(&v),sizeof(v));};
        out.write("RIFF",4);put(raw_count*8+36);out.write("WAVEfmt ",8);put(std::uint32_t{16});
        put(std::uint16_t{3});put(std::uint16_t{2});put(std::uint32_t{fs});put(std::uint32_t{fs*8});
        put(std::uint16_t{8});put(std::uint16_t{32});out.write("data",4);put(raw_count*8);
        std::vector<C> block(65536);
        for(unsigned first=0;first<raw_count;first+=unsigned(block.size())) {
            const auto count=std::min(unsigned(block.size()),raw_count-first);
            for(unsigned j=0;j<count;++j) {
                const double nominal=double(first+j)/fs;
                // Inverse of ADC coordinate Fs*(ratio*t + drift*t^2/2).
                const double t=2*nominal/(ratio+std::sqrt(ratio*ratio+2*drift*nominal));
                C value{};
                for(unsigned c=0;c<2;++c) {
                    const auto rf=capture.channels[c].rf_hz;
                    const double delay=delay_samples[c]/fo-doppler[c]/rf*t;
                    const auto direct=sample(base[c],t*fo-epoch[c]);
                    const auto echo=reflectivity*sample(base[c],(t-delay)*fo-epoch[c])*
                        static_cast<C>(std::polar(1.0,-tau*rf*delay));
                    value+=(direct+echo)*static_cast<C>(std::polar(1.0-.2*c,
                        tau*(offsets[c]+tx_offset[c]-receiver_lo_error)*t+phase[c]));
                }
                block[j]=value;
            }
            out.write(reinterpret_cast<const char*>(block.data()),count*sizeof(C));
        }
    }
    std::cerr<<"Independent two-transmitter IQ fixture generated\n";
    WaveReader wave(file.path);auto clock=std::make_shared<SampleClock>(fs,offsets,3);
    GpuChannelBank bank(wave,offsets,0,double(raw_count)/fs);bank.set_clock(clock);capture.origin=bank.origin;
    GpuPipelineConfig config{{16U,16U},bank.capacity(),bank.origin,3,8,0,true};
    if(bounded){config.fec_workers=2;config.frame_slots=2;}
    GpuPipeline pipeline(config,bank.context(),clock);
    std::array<std::complex<double>,2> correlation{},wrong_rf{};
    std::array<double,2> energy{},phase_square{},phase_max{},row_magnitudes{};
    std::array<unsigned,2> observations{},batches{},checked{},outside_guard{};
    std::array<std::uint64_t,2> last{};
    double max_reference_error=0;
    auto consume=[&](const SpectralBatch& s) {
        if(failure)throw ConsumerFailure{};
        SarBatch batch(capture,s,clock->segments());const auto& f=s.frame;const unsigned c=f.channel;
        ++batches[c];
        const auto* x=static_cast<const C*>(s.transmitted.carriers.mapped);
        const auto* valid=static_cast<const std::uint32_t*>(s.transmitted.valid.mapped);
        const auto* aligned=static_cast<const C*>(s.residual.reference.mapped);
        const auto* normalized=static_cast<const C*>(s.residual.normalized.mapped);
        std::vector<C> reference(f.symbols.size()*length);
        std::vector<bool> good(f.symbols.size(),true);
        for(unsigned row=0;row<f.symbols.size();++row) {
            const double t=batch.original_adc(row)/fs;
            const double physical=2*t/(ratio+std::sqrt(ratio*ratio+2*drift*t));
            const auto tx_row=std::int64_t(std::floor((physical*fo-epoch[c])/length));
            check(tx_row>=0 && tx_row<symbols,"Reference symbol epoch out of range");
            std::fill(work.begin(),work.end(),C{});
            for(unsigned k=0;k<bins;++k) {
                if(valid[row*bins+k]) {
                    max_reference_error=std::max(max_reference_error,double(std::abs(x[row*bins+k]-transmitted[c][tx_row*bins+k])));
                    ++checked[c];
                } else good[row]=false;
                work[(k+nfft-3408)%nfft]=x[row*bins+k];
            }
            fft.transform(work,true);
            std::copy(work.end()-512,work.end(),reference.begin()+row*length);
            std::copy(work.begin(),work.end(),reference.begin()+row*length+512);
        }
        for(std::size_t row=f.first;row<f.last;++row) {
            if(last[c])check(f.symbols[row].window.sample-last[c]==length,"Handoff duplicated/dropped rows at a batch boundary");
            last[c]=f.symbols[row].window.sample;
            if(row==0 || row+1>=f.symbols.size() || !good[row-1] || !good[row] || !good[row+1] ||
               batch.seconds(row)<.8 || row%13)continue;
            const auto rf=batch.carrier_rf_hz(3408);
            for(unsigned j=0;j<nfft;++j) {
                const double t=batch.seconds(row)+j/fo;
                const double delay=delay_samples[c]/fo-doppler[c]/rf*t;
                const double at=row*length+512+j-s.estimate.reference_origin-delay*fo;
                work[j]=sample(reference,at)*static_cast<C>(std::polar(1.0,-tau*rf*delay));
            }
            fft.transform(work);
            std::complex<double> row_correlation{};
            for(unsigned k=0;k<bins;++k) {
                const auto q=std::complex<double>(work[(k+nfft-3408)%nfft]);
                const auto e=std::complex<double>(normalized[row*bins+k]*aligned[row*bins+k]);
                row_correlation+=e*std::conj(q);energy[c]+=std::norm(q);
                // A deliberately wrong RF assignment includes the receiver LO
                // error. It must NOT give the same coherent target phase.
                const double delay=delay_samples[c]/fo-doppler[c]/rf*batch.seconds(row);
                wrong_rf[c]+=e*std::conj(q*std::polar(1.0,tau*receiver_lo_error*delay));
            }
            correlation[c]+=row_correlation;row_magnitudes[c]+=std::abs(row_correlation);
            const double error=std::arg(row_correlation/std::complex<double>(reflectivity))*180/std::numbers::pi;
            phase_square[c]+=error*error;phase_max[c]=std::max(phase_max[c],std::abs(error));
            ++observations[c];if(delay_samples[c]>512)++outside_guard[c];
        }
    };
    for(;;){const auto input=bank.next_device();if(!input.count)break;pipeline.process(input,consume);}
    pipeline.finish(consume);
    check(max_reference_error<1e-6,"Multi-transmitter FEC/reconstruction disagrees with known transmitted data");
    std::complex<double> sum{};double magnitudes=0;
    for(unsigned c=0;c<2;++c) {
        check(batches[c]>=2 && observations[c]>=20 && checked[c]>100000,"Insufficient multi-channel/batch-boundary coverage");
        const auto recovered=correlation[c]/energy[c],truth=std::complex<double>(reflectivity);
        const double phase_error=std::arg(recovered/truth)*180/std::numbers::pi;
        const double phase_rms=std::sqrt(phase_square[c]/observations[c]);
        std::cout<<"Channel "<<c<<": observations="<<observations[c]<<" beyond_guard="<<outside_guard[c]
            <<" amplitude="<<std::abs(recovered)<<" phase_error_deg="<<phase_error<<" phase_rms_deg="<<phase_rms
            <<" phase_max_deg="<<phase_max[c]<<" temporal_coherence="<<std::abs(correlation[c])/row_magnitudes[c]<<" wrong_RF_phase_deg="
            <<std::arg(wrong_rf[c]/energy[c]/truth)*180/std::numbers::pi<<'\n';
        check(std::abs(phase_error)<3,"Physical excess-path phase exceeds the 3 degree budget");
        check(phase_rms<3,"Per-window phase RMS exceeds the 3 degree budget");
        check(std::abs(std::abs(recovered)/std::abs(truth)-1)<.05,"Moving-echo amplitude lost in handoff");
        sum+=recovered;magnitudes+=std::abs(recovered);
    }
    check(outside_guard[1]>=20,"No actual reference-symbol boundary crossing tested");
    check(std::abs(std::arg(wrong_rf[1]/std::complex<double>(reflectivity)))*180/std::numbers::pi>10,
          "Physical RF/CFO distinction test is insensitive");
    check(std::abs(sum)/magnitudes>std::cos(3*std::numbers::pi/180),"Distinct transmitter outputs do not add coherently");
    std::cout<<"Raw IQ to coherent multi-transmitter templates: PASS; clock segments="<<clock->segments().size()
        <<" coherent amplitude fraction="<<std::abs(sum)/magnitudes<<'\n';
}
}
int main(int argc,char** argv) {
    const std::string_view mode=argc>1 ? argv[1] : "";
    try{
        if(mode.starts_with("--gpu"))multichannel(mode=="--gpu-bounded" || mode=="--gpu-mixed",mode=="--gpu-failure",mode=="--gpu-mixed");else coordinates();
        if(mode=="--gpu-failure")throw std::runtime_error("Consumer failure path was not exercised");
    }
    catch(const ConsumerFailure&){if(mode!="--gpu-failure")return 1;std::cout<<"Pending worker/GPU ownership survives consumer exception: PASS\n";}
    catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
