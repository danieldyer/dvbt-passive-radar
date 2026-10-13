#include "recording_metadata.hpp"
#include "gpu_frontend.hpp"
#include "gpu_ofdm.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// CP-only timing check on a fixed common clock: no decoder/pilot-clock feedback.
int main(int argc,char** argv) {
    try {
        if(argc!=5)throw std::invalid_argument("cp_probe INPUT.wav METADATA.json START_SECONDS SECONDS");
        std::ostream output(std::cout.rdbuf());std::cout.rdbuf(std::cerr.rdbuf());
        const auto metadata=radar::RecordingMetadata::read(argv[2]);
        radar::WaveReader wave(argv[1]);metadata.validate_wave(wave.info);
        std::vector<double> centres;
        for(const auto& s:metadata.signals)centres.push_back(s.channel.nco_hz);
        radar::GpuChannelBank bank(wave,centres,std::stod(argv[3]),std::stod(argv[4]));
        bank.set_clock(std::make_shared<radar::SampleClock>(bank.coordinate_rate(),centres));
        radar::Modem modem;radar::GpuOfdm ofdm(modem.pilot_pattern(),1,bank.context());
        std::vector<double> previous(centres.size());std::vector<bool> initialized(centres.size());
        output<<"time_s,channel,cp_phase_samples,unwrapped_phase_samples,quality\n"<<std::setprecision(12);
        for(;;) {
            const auto input=bank.next_device();if(!input.count)break;
            for(unsigned c=0;c<centres.size();++c) {
                const auto guard=metadata.signals[c].channel.guard;
                const unsigned length=8192+8192/guard;
                if(input.count<2*length)continue;
                std::vector<std::uint32_t> positions;
                for(unsigned j=0;j<=length;j+=4)positions.push_back(c*input.stride+j);
                const auto scores=ofdm.correlate(input.samples,positions,guard);
                const auto best=std::max_element(scores.begin(),scores.end(),[](auto a,auto b){return a.quality<b.quality;});
                const auto index=positions[best-scores.begin()]-c*input.stride;
                const double sample=double(bank.origin)/bank.decimation+input.first+index;
                const double phase=std::fmod(sample,length);
                const double unwrapped=initialized[c] ? previous[c]+std::remainder(phase-previous[c],length) : phase;
                previous[c]=unwrapped;initialized[c]=true;
                output<<sample/8e6<<','<<metadata.signals[c].id<<','<<phase<<','<<unwrapped<<','<<best->quality<<'\n';
            }
        }
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
