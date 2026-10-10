#include "gpu_frontend.hpp"
#include "sar_handoff.hpp"
#include <charconv>
#include <cmath>
#include <iostream>
#include <string_view>

namespace {
double number(std::string_view s) {
    double x{};auto [end,error]=std::from_chars(s.data(),s.data()+s.size(),x);
    if(error!=std::errc{} || end!=s.data()+s.size() || !std::isfinite(x))throw std::invalid_argument("Invalid number: "+std::string(s));
    return x;
}
radar::SarChannel channel(std::string_view s) {
    const auto comma=s.find(',');
    if(comma==s.npos)throw std::invalid_argument("--channel requires RF_HZ,NCO_HZ[,GUARD_DENOMINATOR]");
    const auto next=s.find(',',comma+1);
    const double guard=next==s.npos ? 16 : number(s.substr(next+1));
    if(guard!=4 && guard!=8 && guard!=16 && guard!=32)throw std::invalid_argument("Invalid guard denominator");
    return {number(s.substr(0,comma)),number(s.substr(comma+1,next==s.npos ? next : next-comma-1)),unsigned(guard)};
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string_view(argv[1])=="--help") {
            std::cout<<"Usage: dvbt_sar INPUT.wav --output FILE.sar --recording-rf-hz HZ\n"
                <<"       --channel RF_HZ,NCO_HZ[,GUARD_DENOMINATOR] (repeat per channel)\n"
                <<"       [--start 0] [--seconds 0] [--response-order 8] [--clock-phase-degrees 3]\n"
                <<"       [--reference-delay-samples 0] (explicit fixed origin; otherwise acquisition estimates it)\n"
                <<"Unpaced Vulkan raw-IQ to complex SAR spectra, 7 MHz non-hierarchical 8K DVB-T.\n"
                <<"RF_HZ is the declared transmitter centre; NCO_HZ is relative to recording centre.\n"
                <<"No independent per-channel resampling, map processing or video output.\n";
            return argc<2 ? 1 : 0;
        }
        radar::SarCapture capture;radar::GpuPipelineConfig config;
        std::filesystem::path output;double start=0,seconds=0;
        for(int i=2;i<argc;++i) {
            const std::string_view key=argv[i];
            if(i+1==argc)throw std::invalid_argument("Missing option value");
            const std::string_view value=argv[++i];
            if(key=="--output")output=value;
            else if(key=="--channel")capture.channels.push_back(channel(value));
            else if(key=="--recording-rf-hz")capture.recording_rf_hz=number(value);
            else if(key=="--start")start=number(value);
            else if(key=="--seconds")seconds=number(value);
            else if(key=="--clock-phase-degrees")capture.clock_phase_degrees=number(value);
            else if(key=="--reference-delay-samples") {config.reference_delay=number(value);config.automatic_origin=false;}
            else if(key=="--response-order") {
                const auto n=number(value);
                if(n<0 || n>32 || std::floor(n)!=n)throw std::invalid_argument("Response order must be 0..32");
                config.response_order=unsigned(n);
            } else throw std::invalid_argument("Unknown option: "+std::string(key));
        }
        if(output.empty() || start<0 || seconds<0)throw std::invalid_argument("Output and a nonnegative interval are required");
        radar::WaveReader wave(argv[1]);capture.input_rate=wave.info.sample_rate;capture.decimation=capture.input_rate/8'000'000;
        capture.validate();
        if(start>=double(wave.info.samples)/capture.input_rate)throw std::invalid_argument("Start is beyond recording");
        if(seconds==0)seconds=double(wave.info.samples)/capture.input_rate-start;
        std::vector<double> centres;
        for(const auto& c:capture.channels){centres.push_back(c.nco_hz);config.guards.push_back(c.guard);}
        auto clock=std::make_shared<radar::SampleClock>(capture.input_rate,centres,capture.clock_phase_degrees);
        radar::GpuChannelBank bank(wave,centres,start,seconds);bank.set_clock(clock);
        capture.origin=bank.origin;config.origin=bank.origin;config.decimation=bank.decimation;config.max_samples=bank.capacity();
        radar::GpuPipeline pipeline(config,bank.context(),clock);
        radar::SarWriter writer(output,capture);
        std::vector<std::size_t> rows(centres.size());
        auto consume=[&](const radar::SpectralBatch& spectra) {
            writer.write(radar::SarBatch(capture,spectra,clock->segments()));
            rows[spectra.frame.channel]+=spectra.frame.last-spectra.frame.first;
        };
        for(;;){const auto input=bank.next_device();if(!input.count)break;pipeline.process(input,consume);}
        pipeline.finish(consume);writer.finish();
        bool empty=false;
        for(std::size_t c=0;c<rows.size();++c){std::cerr<<"Channel "<<c<<": "<<rows[c]<<" emitted symbols\n";empty|=rows[c]==0;}
        if(empty){std::cerr<<"One or more channels produced no decoded frames\n";return 2;}
    } catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
