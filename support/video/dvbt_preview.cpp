// Optional device-native video sink: production DSP has no display framing.
#include "detect_cli.hpp"
#include "gpu_preview.hpp"
#include <boost/json.hpp>
#include <cmath>
#include <iostream>

namespace {
struct Previews {
    std::vector<std::unique_ptr<radar::video::GpuPreview>> channels;
    ~Previews(){
        double gpu=0;std::uint64_t maps=0;
        for(const auto& preview:channels){gpu+=preview->timing.gpu;maps+=preview->timing.batches;}
        std::cerr<<"GPU preview: "<<maps<<" maps, "<<gpu*1000<<" ms\n";
    }
};
}

int main(int argc,char** argv){
    return radar::detection_cli(argc,argv,[](auto context,std::size_t channels,std::ostream& stream){
        auto previews=std::make_shared<Previews>();
        for(std::size_t c=0;c<channels;++c)previews->channels.push_back(std::make_unique<radar::video::GpuPreview>(context));
        radar::DetectionSink sink;
        sink.project=[previews](auto& map,const auto& buffer){
            previews->channels.at(map.channel)->calculate(map,buffer,false);
            map.noise_range.clear();map.noise_doppler.clear();
        };
        sink.publish=[previews,&stream](std::size_t channel,auto&& map){
            const auto& preview=*previews->channels.at(channel);
            auto metadata=boost::json::parse(map.json());auto& m=metadata.as_object();
            m["display_shape"]=boost::json::array{preview.height,preview.width};
            m["preview_format"]="power_f32le";m["preview_order"]="doppler_down,range";
            m["map_peak_db"]=10*std::log10(std::max(*std::max_element(preview.power.begin(),preview.power.end()),1e-45f));
            const auto header=boost::json::serialize(metadata);
            const std::uint32_t length=std::uint32_t(header.size());const std::uint64_t bytes=preview.power.size()*sizeof(float);
            stream.write("RDP1",4);stream.write(reinterpret_cast<const char*>(&length),4);stream.write(reinterpret_cast<const char*>(&bytes),8);
            stream.write(header.data(),length);stream.write(reinterpret_cast<const char*>(preview.power.data()),std::streamsize(bytes));
            if(!stream)throw std::runtime_error("GPU preview output failed");
        };
        return sink;
    });
}
