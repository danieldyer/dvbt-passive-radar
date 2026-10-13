#include "gpu_preview.hpp"
namespace radar::video {
namespace {
constexpr std::uint32_t code[]=
#include "preview.inc"
;
struct Parameters {unsigned ranges,frequencies,range_stride,frequency_stride,width,height,normalize;};
}
GpuPreview::GpuPreview(std::shared_ptr<gpu::Context> context):kernel_(code,sizeof(Parameters),{},std::move(context)){}
void GpuPreview::calculate(const DopplerMap& map,const gpu::BufferView& input,bool normalize){
    if(!map.ranges || !map.frequencies || input.bytes<map.ranges*map.frequencies*sizeof(C) ||
       (normalize && (map.noise_range.size()!=map.ranges || map.noise_doppler.size()!=map.frequencies)))
        throw std::invalid_argument("Invalid GPU preview dimensions or noise calibration");
    const auto rd=unsigned((map.ranges+1623)/1624),fd=unsigned((map.frequencies+739)/740);
    width=unsigned((map.ranges+rd-1)/rd);height=unsigned((map.frequencies+fd-1)/fd);
    kernel_.bind(0,input);
    if(normalize){kernel_.upload<float>(1,map.noise_range);kernel_.upload<float>(2,map.noise_doppler);}
    else {kernel_.storage(1,sizeof(float));kernel_.storage(2,sizeof(float));}
    auto* output=static_cast<float*>(kernel_.storage(3,width*height*sizeof(float)));
    auto* invalid=static_cast<unsigned*>(kernel_.storage(4,sizeof(unsigned)));*invalid=0;
    kernel_.run(Parameters{unsigned(map.ranges),unsigned(map.frequencies),rd,fd,width,height,unsigned(normalize)},width*height,&timing);
    if(*invalid)throw std::runtime_error("Nonfinite scientific map or invalid preview calibration");
    power.assign(output,output+width*height);
}
}
