#include "pipeline.hpp"
#include <gnuradio/prefs.h>
#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc,char** argv) {
    try {
        radar::PipelineOptions options;
        std::string map_stream;
        std::ostream binary_stdout(std::cout.rdbuf());
        std::ofstream archive;
        auto usage=[] {
            std::cout<<"Usage: dvbt_reconstruct INPUT.wav --output DIRECTORY [--seconds 0 (entire file)] [--start 0]\n"
                <<"       [--offset-hz 4625000] [--guard 16] [--response-order 8] [--reference-delay-samples 0]\n"
                <<"       [--compute vulkan|cpu] [--frontend vulkan|cpu] [--raw-output] [--no-maps]\n"
                <<"       [--fec viterbi] CPU hard-decision Viterbi + errors-only RS.\n"
                <<"       Repeat --offset-hz for multiple coherent channel outputs (channel_00, channel_01, ...).\n"
                <<"       [--clock-phase-degrees 3]  RMS clock budget across the occupied frequency span.\n"
                <<"       [--range-metres 30000] [--doppler-hz 100] [--history-seconds 300]\n"
                <<"       [--integration-seconds 0] [--map-interval 2] [--slow-decimation 4] [--remove-dc]\n"
                <<"       --remove-dc subtracts the Hann-weighted complex mean per range before Doppler FFT.\n"
                <<"       [--map-stream FILE|-]  Lossless RDM frames; '-' uses stdout, logs use stderr.\n"
                <<"7 MHz, non-hierarchical 8K DVB-T; TPS determines constellation and code rate.\n";
        };
        if(argc<2 || std::string_view(argv[1])=="--help") { usage();return argc<2 ? 1 : 0; }
        options.input=argv[1];
        for(int i=2;i<argc;++i) {
            const std::string_view key=argv[i];
            if(key=="--summary-only") { options.summary_only=true;continue; }
            if(key=="--raw-output") { options.summary_only=false;continue; }
            if(key=="--no-maps") { options.maps=false;continue; }
            if(key=="--remove-dc") { options.map.remove_dc=true;continue; }
            if(i+1>=argc)throw std::runtime_error("Missing option value");
            const std::string_view value=argv[++i];
            if(key=="--output") { options.output=value;continue; }
            if(key=="--map-stream") { map_stream=value;continue; }
            if(key=="--fec") {
                if(value!="viterbi")throw std::runtime_error("--fec supports only viterbi (CPU hard decisions + RS)");
                continue;
            }
            if(key=="--compute" || key=="--frontend") {
                if(value!="cpu" && value!="vulkan")throw std::runtime_error("--compute must be cpu or vulkan");
                if(key=="--compute")options.gpu=options.gpu_frontend=value=="vulkan";
                else options.gpu_frontend=value=="vulkan";
                continue;
            }
            double number{};
            const auto [end,error]=std::from_chars(value.data(),value.data()+value.size(),number);
            if(error!=std::errc{} || end!=value.data()+value.size() || !std::isfinite(number))
                throw std::runtime_error("Invalid numeric option");
            if(key=="--seconds")options.seconds=number;
            else if(key=="--start")options.start_seconds=number;
            else if(key=="--offset-hz") {options.offset_hz=number;options.offsets_hz.push_back(number);}
            else if(key=="--range-metres")options.map.range_metres=number;
            else if(key=="--doppler-hz")options.map.doppler_hz=number;
            else if(key=="--history-seconds")options.map.history_seconds=number;
            else if(key=="--integration-seconds")options.map.integration_seconds=number;
            else if(key=="--map-interval")options.map.update_seconds=number;
            else if(key=="--coherence-degrees")options.map.coherence_degrees=number;
            else if(key=="--clock-phase-degrees" && number>0 && number<=30)options.clock_phase_budget_degrees=number;
            else if(key=="--slow-decimation" && number>=1 && number<=16 && std::floor(number)==number)options.map.slow_decimation=static_cast<unsigned>(number);
            else if(key=="--reference-delay-samples") { options.reference_delay_samples=number;options.automatic_origin=false; }
            else if(key=="--guard" && (number==4 || number==8 || number==16 || number==32))options.guard=static_cast<int>(number);
            else if(key=="--response-order" && number>=0 && number<=32 && std::floor(number)==number)options.response_order=static_cast<unsigned>(number);
            else throw std::runtime_error("Unknown option or unsupported value: "+std::string(key));
        }
        if(options.output.empty())throw std::runtime_error("--output is required");
        if(options.offsets_hz.size()>1 && !map_stream.empty())throw std::runtime_error("--map-stream requires a single channel; per-channel maps are still written with multiple channels");
        if(!map_stream.empty()) {
            if(!options.maps)throw std::runtime_error("--map-stream conflicts with --no-maps");
            if(map_stream=="-") {
                options.map.stream=&binary_stdout;
                gr::prefs::singleton()->set_string("LOG","log_file","stderr");
            } else {
                if(std::filesystem::exists(map_stream))throw std::runtime_error("Map stream file already exists");
                archive.open(map_stream,std::ios::binary);
                if(!archive)throw std::runtime_error("Cannot create map stream file");
                options.map.stream=&archive;
            }
        }
        struct Redirect {
            std::streambuf* previous;
            explicit Redirect(bool active) : previous(active ? std::cout.rdbuf(std::cerr.rdbuf()) : nullptr) {}
            ~Redirect(){if(previous)std::cout.rdbuf(previous);}
        } redirect(map_stream=="-");
        return radar::process_recording(options);
    } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n';return 1; }
}
