#include "native_display.hpp"
#include "gpu_buffer.hpp"
#include "gpu_commands.hpp"
#include <cmath>
#include <iostream>
#include <thread>
#include <boost/json.hpp>
#include <fstream>

using namespace radar;
namespace {void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::invalid_argument("Usage: native_display_check METADATA.json NEW_OUTPUT_DIRECTORY");
    const auto description=RecordingMetadata::read(argv[1]);const std::filesystem::path output=argv[2];
    check(std::filesystem::create_directory(output),"Output directory already exists");
    auto context=std::make_shared<gpu::Context>();live::NativeDisplay display;auto consumer=display.factory()(context,description);
    DopplerMap map;map.ranges=801;map.frequencies=321;map.frequency_start=-100;map.frequency_step=.625;
    map.integration=3;map.time_start=4;map.time_end=7;map.amplitude_units="complex_reflectivity";
    const auto bytes=map.ranges*map.frequencies*sizeof(C);
    auto buffer=gpu::make_buffer(context->gpu,bytes);
    auto* values=static_cast<C*>(context->gpu.device->mapMemory(*buffer.memory,0,bytes));
    const gpu::BufferView input{context,*buffer.handle,bytes,values};
    auto pump=[&](int iterations){for(int i=0;i<iterations;++i){display.poll();std::this_thread::sleep_for(std::chrono::milliseconds(8));}};
    boost::json::array results;
    for(std::size_t channel=0;channel<2;++channel){
        display.open(channel);check(consumer.active(channel) && !consumer.active(channel+1),"Wrong focusing subscription");
        for(std::size_t r=0;r<map.ranges;++r)for(std::size_t f=0;f<map.frequencies;++f){
            const double a=std::exp(-.5*std::pow((double(r)-160-120*channel)/3,2)-.5*std::pow((double(f)-200+80*channel)/2,2));
            values[r*map.frequencies+f]={float(std::sqrt(1+200*a+.05*std::sin(double(r*7919+f*71)))),0};
        }
        map.channel=unsigned(channel);map.time_end=7+channel;consumer.project(map,input);pump(20);
        display.save_snapshot(output/(description.signals[channel].id+".png"));pump(10);
        const auto status=display.status();check(status.open && status.channel==channel && status.texture_uploads>=channel+1 && status.rendered_frames>5 && status.map_time==map.time_end,"Native renderer never displayed the projected map");
        boost::json::array rectangle;for(auto pixel:status.plot_pixels)rectangle.push_back(pixel);
        results.emplace_back(boost::json::object{{"id",description.signals[channel].id},{"plot_pixels",rectangle},
            {"frames",status.rendered_frames},{"uploads",status.texture_uploads},
            {"expected_x",(160.5+120*channel)/map.ranges},{"expected_y",(map.frequencies-200+80*channel-.5)/map.frequencies}});
        display.close();check(!display.status().open && !consumer.active(channel),"Closing display retained focusing subscription");
    }
    context->gpu.device->unmapMemory(*buffer.memory);
    std::ofstream(output/"renderer.json")<<boost::json::serialize(results);
    std::cout<<"Native ImPlot/Metal texture rendering, signed Doppler axes, channel selection, close/reopen and owned framebuffer PNG: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
