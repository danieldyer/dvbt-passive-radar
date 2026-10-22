#include "capture_source.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace radar;
namespace {void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}}
int main(){try{
    const auto path=std::filesystem::temp_directory_path()/("dvbt-replay-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".wav");
    struct Remove {std::filesystem::path path;~Remove(){std::error_code error;std::filesystem::remove(path,error);}} remove{path};
    const std::array<Iq16,7> samples{{{0,1},{-32768,32767},{2,-3},{4,-5},{6,-7},{8,-9},{10,-11}}};
    constexpr unsigned rate=1000;
    {std::ofstream file(path,std::ios::binary);auto put=[&](auto value){file.write(reinterpret_cast<const char*>(&value),sizeof(value));};
        file.write("RIFF",4);put(std::uint32_t{36+sizeof(samples)});file.write("WAVEfmt ",8);put(std::uint32_t{16});
        put(std::uint16_t{1});put(std::uint16_t{2});put(std::uint32_t{rate});put(std::uint32_t{rate*4});
        put(std::uint16_t{4});put(std::uint16_t{16});file.write("data",4);put(std::uint32_t{sizeof(samples)});
        file.write(reinterpret_cast<const char*>(samples.data()),sizeof(samples));}
    SdrCaptureSettings settings;settings.centre_hz=202e6;
    auto source=make_replay_source(path,settings,false);source->start();
    CaptureTimeline timeline;CaptureCounters counters;std::array<Iq16,3> out;
    std::uint64_t total=0;unsigned restarts=0;
    while(total<70){auto read=source->receive(out);check(!read.end && read.tick==std::int64_t(total),"Looping replay terminated or restarted sample epoch");
        if(read.restart){timeline.restart();++restarts;}
        const auto span=timeline.receive(read.tick,read.count,counters);
        check(span.first==total && span.segment==restarts,"Replay seam failed to break coherent integration");
        for(std::size_t i=0;i<read.count;++i)check(out[i].i==samples[(total+i)%samples.size()].i && out[i].q==samples[(total+i)%samples.size()].q,"Loop changed ADC bits or I/Q order");
        total+=read.count;
    }
    check(restarts==9 && counters.missing_samples==0 && counters.unknown_gaps==0 && counters.timestamp_gaps==0,"Replay seam falsely reported sample drops");
    source=make_replay_source(path,settings,false,0,0,true,true);source->start();total=0;unsigned loops=0;
    CaptureTimeline continuous;CaptureCounters continuous_counts;
    while(total<70){const auto read=source->receive(out);check(!read.restart,"Verified cyclic source reset the demodulator at a file seam");
        check(read.tick==std::int64_t(total),"Cyclic replay changed the hardware epoch");loops+=read.looped;
        continuous.receive(read.tick,read.count,continuous_counts);total+=read.count;}
    check(loops==9 && continuous_counts.segments==1 && continuous_counts.missing_samples==0,"Cyclic source invented a capture gap");
    source=make_replay_source(path,settings,false,0,0,false);source->start();total=0;
    for(;;){auto read=source->receive(out);if(read.end)break;check(!read.restart,"Single-pass replay looped");total+=read.count;}
    check(total==samples.size(),"Single-pass replay truncated data");
    source=make_replay_source(path,settings,false,.002,2);source->start();unsigned errors=0;
    for(int n=0;n<20;++n){auto read=source->receive(out);check(!read.end,"Injected looping replay ended");errors+=read.error==CaptureRead::Error::overflow;}
    check(errors==1,"Injected transport loss repeated on every loop");
    std::cout<<"Continuous int16 replay, monotonic epochs, seam segmentation without false loss, finite mode and one-shot injection: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
