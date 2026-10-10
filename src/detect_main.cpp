#include "gpu_frontend.hpp"
#include "matched_detector.hpp"
#include <charconv>
#include <cmath>
#include <iostream>
#include <string_view>
#include <gnuradio/prefs.h>

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
            std::cout<<"Usage: dvbt_detect INPUT.wav --output DIRECTORY --recording-rf-hz HZ\n"
                <<"       --channel RF_HZ,NCO_HZ[,GUARD_DENOMINATOR] (repeat per channel)\n"
                <<"       [--start 0] [--seconds 0] [--response-order 8] [--clock-phase-degrees 3]\n"
                <<"       [--reference-delay-samples 0] (explicit fixed origin; otherwise acquisition estimates it)\n"
                <<"Unpaced Vulkan raw-IQ to complex detection maps, 7 MHz non-hierarchical 8K DVB-T.\n"
                <<"RF_HZ is the declared transmitter centre; NCO_HZ is relative to recording centre.\n"
                <<"Defaults: 3 s rectangular CPI, +/-200 Hz, 0-30 km, stationary suppression.\n"
                <<"       [--integration-seconds 3] [--update-seconds 1] [--doppler-hz 200]\n"
                <<"       [--range-metres 30000] [--keep-dc] [--physical-units] [--archive-maps]\n"
                <<"       [--map-stream FILE|-] [--stream-only] (lossless complex output; logs use stderr)\n"
                <<"RDM1 complex maps include phase epoch and reversible noise calibration.\n";
            return argc<2 ? 1 : 0;
        }
        radar::SarCapture capture;radar::GpuPipelineConfig config;radar::DetectionOptions detection;bool physical=false,archive_maps=false,stream_only=false;
        std::string map_stream;std::ofstream archive;
        std::filesystem::path output;double start=0,seconds=0;
        for(int i=2;i<argc;++i) {
            const std::string_view key=argv[i];
            if(key=="--keep-dc"){detection.focus.remove_dc=false;continue;}
            if(key=="--physical-units"){physical=true;continue;}
            if(key=="--archive-maps"){archive_maps=true;continue;}
            if(key=="--stream-only"){stream_only=true;continue;}
            if(i+1==argc)throw std::invalid_argument("Missing option value");
            const std::string_view value=argv[++i];
            if(key=="--output")output=value;
            else if(key=="--map-stream")map_stream=value;
            else if(key=="--channel")capture.channels.push_back(channel(value));
            else if(key=="--recording-rf-hz")capture.recording_rf_hz=number(value);
            else if(key=="--integration-seconds")detection.integration_seconds=number(value);
            else if(key=="--update-seconds")detection.update_seconds=number(value);
            else if(key=="--doppler-hz")detection.focus.doppler_hz=number(value);
            else if(key=="--range-metres")detection.focus.range_metres=number(value);
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
        if(stream_only && (map_stream.empty() || archive_maps))throw std::invalid_argument("--stream-only requires --map-stream and conflicts with --archive-maps");
        // Preserve the binary sink before directing every library diagnostic to
        // stderr. Constructors may log before the first map is published.
        std::ostream binary_stdout(std::cout.rdbuf());
        struct Redirect {
            std::streambuf* previous;
            explicit Redirect(bool active):previous(active ? std::cout.rdbuf(std::cerr.rdbuf()) : nullptr){}
            ~Redirect(){if(previous)std::cout.rdbuf(previous);}
        } redirect(map_stream=="-");
        if(map_stream=="-")gr::prefs::singleton()->set_string("LOG","log_file","stderr");
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
        std::filesystem::create_directories(output);
        std::ostream* stream=nullptr;
        if(map_stream=="-")stream=&binary_stdout;
        else if(!map_stream.empty()){
            archive.open(map_stream,std::ios::binary|std::ios::out|std::ios::noreplace);
            if(!archive)throw std::runtime_error("Cannot create map stream");stream=&archive;
        }
        detection.standardize_noise=!physical;
        radar::MatchedDetector detector(bank.context(),centres.size(),detection);
        radar::DetectionTimings timing;std::size_t maps=0;
        auto publish=[&](std::size_t channel,radar::DopplerMap&& map){
            auto directory=output;
            if(centres.size()>1)directory/="channel_"+std::string(channel<10 ? "0" : "")+std::to_string(channel);
            std::filesystem::create_directories(directory);
            if(archive_maps){
                const auto file=directory/("map-"+std::to_string(map.sequence)+".rdm");
                if(std::filesystem::exists(file))throw std::runtime_error("Output map already exists: "+file.string());
                map.write(file);
            }
            if(!stream_only)map.write(directory/"latest.rdm");if(stream)map.write(*stream);++maps;
        };
        std::vector<std::size_t> rows(centres.size());
        auto consume=[&](const radar::SpectralBatch& spectra) {
            detector.append(radar::SarBatch(capture,spectra,clock->segments()),publish,&timing);
            rows[spectra.frame.channel]+=spectra.frame.last-spectra.frame.first;
        };
        for(;;){const auto input=bank.next_device();if(!input.count)break;pipeline.process(input,consume);}
        pipeline.finish(consume);
        detector.finish(publish,&timing);
        std::ofstream metadata(output/"metadata.json");metadata<<"{\"start_s\":"<<start<<",\"duration_s\":"<<seconds<<",\"input_samples\":"<<wave.info.samples<<",\"input_sample_rate\":"<<capture.input_rate<<",\"maps\":"<<maps<<",\"detector\":\"finite_waveform_constant_velocity\"}\n";
        std::cerr<<"Maps: "<<maps<<"; GPU noise/profiles/focus ms: "<<timing.noise.gpu*1000<<" / "<<timing.profiles.gpu*1000<<" / "<<timing.focus.gpu*1000<<"; noise calibration CPU ms: "<<timing.calibration_cpu*1000<<'\n';
        bool empty=false;
        for(std::size_t c=0;c<rows.size();++c){std::cerr<<"Channel "<<c<<": "<<rows[c]<<" emitted symbols\n";empty|=rows[c]==0;}
        if(!maps)empty=true;
        if(empty){std::cerr<<"No complete detection map, or a channel produced no decoded frames; allow acquisition, stationary training and a full CPI\n";return 2;}
    } catch(const std::exception& e){std::cerr<<"Error: "<<e.what()<<'\n';return 1;}
}
