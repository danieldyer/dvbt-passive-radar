// Exercise the exact production capture path with selected live-map workloads.
// The projector touches a device-resident pixel in place of native-window paint;
// channelization, clocks, decoding, profiles, focusing and archive writes are real.
#include "live_capture.hpp"
#include <atomic>
#include <charconv>
#include <iostream>

int main(int argc,char** argv){try{
    unsigned maps=0;std::vector<char*> arguments{argv[0]};
    for(int i=1;i<argc;++i){
        if(std::string_view(argv[i])=="--map-channels"){
            if(++i==argc)throw std::invalid_argument("Missing --map-channels count");
            const std::string_view s=argv[i];const auto [end,error]=std::from_chars(s.data(),s.data()+s.size(),maps);
            if(error!=std::errc{} || end!=s.data()+s.size())throw std::invalid_argument("Invalid map channel count");
        }else arguments.push_back(argv[i]);
    }
    std::atomic<std::uint64_t> projected=0;
    radar::LiveDisplayFactory factory;
    if(maps)factory=[&](auto,const auto& metadata){
        if(maps>metadata.signals.size())throw std::invalid_argument("More map channels than captured signals");
        radar::LiveDisplay display;
        display.project=[&](const auto&,const auto& values){
            const auto value=static_cast<const radar::C*>(values.mapped)[0];
            if(!std::isfinite(std::norm(value)))throw std::runtime_error("Nonfinite preflight map");
            ++projected;
        };
        display.publish=[](auto,auto&&){};display.active=[&](std::size_t c){return c<maps;};return display;
    };
    const int result=radar::capture_cli(int(arguments.size()),arguments.data(),factory,{},radar::CaptureDisplayMode::headless_projector);
    std::cerr<<"Preflight projected "<<projected<<" complete live maps; native-window painting excluded.\n";
    return result;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
