#include "recording_metadata.hpp"
#include "modem.hpp"
#include "dsp.hpp"
#include "live_input.hpp"
#include <cmath>
#include <iostream>
#include <random>

// Independent OFDM RF fixture. It is support tooling, never a production
// capture mode. At 64 MS/s the VHF carriers lie exactly on the common FFT grid.
int main(int argc,char** argv){
    try{
        if(argc!=4){std::cerr<<"Usage: simulate_capture METADATA.json OUTPUT.wav SECONDS\n";return 1;}
        const auto metadata=radar::RecordingMetadata::read(argv[1]);const double seconds=std::stod(argv[3]);
        if(seconds<=0 || seconds>15)throw std::invalid_argument("Fixture duration must be 0..15 seconds");
        constexpr unsigned fs=64'000'000,nfft=65536,prefix=4096,length=nfft+prefix,bins=6817;
        const auto symbols=unsigned(std::ceil(seconds*fs/length));const auto samples=std::uint64_t(symbols)*length;
        if(samples*4+36>UINT32_MAX)throw std::invalid_argument("Fixture exceeds RIFF size");
        radar::Modem modem;radar::Tps mode;mode.mode=1;mode.constellation=2;mode.rate=2;mode.guard=1;
        const unsigned packets=unsigned(std::ceil(symbols*6048.0*6*.75/(204*8)))+256;
        std::mt19937 random(819923);
        std::vector<std::vector<radar::C>> transmitters;
        for(const auto& signal:metadata.signals){
            if(signal.channel.guard!=16 || std::abs(signal.channel.nco_hz)>28e6 ||
               std::abs(signal.channel.nco_hz/976.5625-std::round(signal.channel.nco_hz/976.5625))>1e-9)
                throw std::invalid_argument("Fixture requires GI 1/16 and integer carrier offsets within 64 MS/s");
            std::vector<std::uint8_t> bytes(packets*188);for(auto& byte:bytes)byte=std::uint8_t(random());
            for(unsigned p=0;p<packets;++p)bytes[p*188]=p%8 ? 0x47 : 0xb8;
            transmitters.push_back(modem.synthesize(bytes,mode,symbols));
        }
        std::ofstream file(argv[2],std::ios::binary|std::ios::out|std::ios::noreplace);file.exceptions(std::ios::badbit|std::ios::failbit);
        auto put=[&](auto value){file.write(reinterpret_cast<const char*>(&value),sizeof(value));};
        file.write("RIFF",4);put(std::uint32_t(samples*4+36));file.write("WAVEfmt ",8);put(std::uint32_t{16});
        put(std::uint16_t{1});put(std::uint16_t{2});put(std::uint32_t{fs});put(std::uint32_t{fs*4});put(std::uint16_t{4});put(std::uint16_t{16});
        file.write("data",4);put(std::uint32_t(samples*4));
        std::vector<radar::C> work(nfft),block(length);std::vector<radar::Iq16> quantized(length);
        radar::Fft fft(nfft);std::normal_distribution<float> noise(0,2e-6f);
        for(unsigned s=0;s<symbols;++s){
            std::fill(work.begin(),work.end(),radar::C{});
            for(std::size_t c=0;c<transmitters.size();++c){
                const auto offset=int(std::llround(metadata.signals[c].channel.nco_hz/976.5625));
                for(unsigned k=0;k<bins;++k)work[(int(k)-3408+offset+int(nfft))%int(nfft)]+=transmitters[c][s*bins+k];
            }
            fft.transform(work,true);std::copy(work.end()-prefix,work.end(),block.begin());std::copy(work.begin(),work.end(),block.begin()+prefix);
            for(unsigned i=0;i<length;++i){
                const auto value=block[i]*40.0f+radar::C(noise(random),noise(random));
                quantized[i]={std::int16_t(std::round(std::clamp(value.real()*32768,-32768.F,32767.F))),
                    std::int16_t(std::round(std::clamp(value.imag()*32768,-32768.F,32767.F)))};
            }
            file.write(reinterpret_cast<const char*>(quantized.data()),std::streamsize(quantized.size()*sizeof(radar::Iq16)));
        }
        std::cout<<"Synthesized "<<metadata.signals.size()<<" independent 64-QAM DVB-T channels, "<<samples<<" samples at "<<fs<<" S/s\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
