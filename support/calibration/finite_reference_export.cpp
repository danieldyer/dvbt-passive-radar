// Sparse, reproducible forward-operator probes from saved channel IQ.
// Export the actual reconstruction, fitted weights, and production C0/C1.
#include "gpu_pipeline.hpp"
#include "gpu_commands.hpp"
#include "gpu_noise.hpp"
#include "matched_profiles.hpp"
#include "sar_handoff.hpp"
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
using namespace radar;
constexpr unsigned bins=6817;
template<class T>T read(std::istream& f){T v{};f.read(reinterpret_cast<char*>(&v),sizeof(v));return v;}
template<class T>void write(std::ostream& f,const T& v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
boost::json::object object(std::istream& f,std::size_t n){
    if(!n || n>1024*1024)throw std::runtime_error("Invalid metadata length");
    std::string s(n,'\0');f.read(s.data(),std::streamsize(n));return boost::json::parse(s).as_object();
}
void record(std::ostream& f,const char* tag,const boost::json::object& meta,std::uint64_t bytes){
    const auto text=boost::json::serialize(meta);f.write(tag,4);write(f,std::uint32_t(text.size()));write(f,bytes);f.write(text.data(),std::streamsize(text.size()));
}
template<class T>void plane(std::ostream& f,const T* p,std::size_t n){f.write(reinterpret_cast<const char*>(p),std::streamsize(n*sizeof(T)));}
struct Channel {
    GpuNoise noise;
    std::uint64_t segment=UINT64_MAX;
    std::vector<std::int64_t> last;
    std::vector<unsigned> selected_count;
    Channel(std::shared_ptr<gpu::Context> c,std::size_t count)
        :noise(std::move(c)),last(count,-1),selected_count(count){}
};
}
int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(C)==8);
    if(argc!=4)throw std::invalid_argument("finite_reference_export CAPTURE.dvblive NEW_OUTPUT.fref REQUEST.json");
    std::ifstream request_file(argv[3]);std::string request_text((std::istreambuf_iterator<char>(request_file)),{});
    const auto request=boost::json::parse(request_text).as_object();
    const double step=boost::json::value_to<double>(request.at("step_s"));
    const auto rows_per_sample=request.contains("rows_per_sample")
        ? boost::json::value_to<unsigned>(request.at("rows_per_sample")) : 1u;
    if(!rows_per_sample || rows_per_sample>512)throw std::invalid_argument("Invalid rows per sample");
    std::vector<std::array<double,2>> intervals;
    for(const auto& v:request.at("intervals_s").as_array()){
        const auto& a=v.as_array();if(a.size()!=2)throw std::invalid_argument("Expected interval endpoints");
        intervals.push_back({boost::json::value_to<double>(a[0]),boost::json::value_to<double>(a[1])});
        const auto [first,last]=intervals.back();
        if(!std::isfinite(first+last) || first<0 || last<=first ||
           (intervals.size()>1 && first<intervals[intervals.size()-2][1]))throw std::invalid_argument("Invalid ordered intervals");
    }
    if(intervals.empty() || !std::isfinite(step) || step<=0)throw std::invalid_argument("Invalid probe cadence");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    char magic[8];input.read(magic,8);if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Not a live archive");
    auto header=object(input,read<std::uint32_t>(input));
    if(header.at("archive_mode")!="both" && header.at("archive_mode")!="channels")throw std::runtime_error("Saved channel IQ is required");
    SarCapture capture;
    capture.input_rate=boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
    capture.source_rate=boost::json::value_to<std::uint32_t>(header.at("source_nominal_rate_hz"));
    capture.decimation=boost::json::value_to<unsigned>(header.at("decimation"));
    capture.origin=boost::json::value_to<std::uint64_t>(header.at("origin"));
    capture.recording_rf_hz=boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
    GpuPipelineConfig config;config.max_samples=262144;config.frames_per_batch=6;config.residual_spectra=false;
    for(const auto& value:header.at("applied_channels").as_array()){
        const auto& c=value.as_object();const auto guard=boost::json::value_to<unsigned>(c.at("guard"));
        capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),boost::json::value_to<double>(c.at("nco_hz")),guard});
        config.guards.push_back(guard);
    }
    capture.validate();const auto count=capture.channels.size();
    if(!count || count>64)throw std::runtime_error("Invalid channel count");
    auto context=std::make_shared<gpu::Context>();GpuPipeline pipeline(config,context,{});GpuMatchedProfiles compressor(context);
    std::vector<std::unique_ptr<Channel>> channels;
    for(std::size_t i=0;i<count;++i)channels.push_back(std::make_unique<Channel>(context,intervals.size()));
    std::ofstream output(argv[2],std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    header["source_capture"]=std::filesystem::absolute(argv[1]).string();header["request"]=request;
    header["source_size_bytes"]=std::filesystem::file_size(argv[1]);
    header["payload_layout"]="c64 previous[6817],current[6817],response[6817]; f32 variance[6817]; c64 measured[6817],C0[1024],C1[1024]";
    header["scope"]="Sparse complete reconstructed rows; direct-path cancellation only; original production weights and finite matched profiles. Selection by time only.";
    const auto description=boost::json::serialize(header);output.write("FREF0001",8);write(output,std::uint32_t(description.size()));output.write(description.data(),std::streamsize(description.size()));
    auto storage=gpu::make_buffer(context->gpu,count*config.max_samples*sizeof(C));
    auto* mapped=context->gpu.device->mapMemory(*storage.memory,0,count*config.max_samples*sizeof(C));
    gpu::BufferView samples{context,*storage.handle,count*config.max_samples*sizeof(C),mapped};
    std::vector<ClockSegment> clock;std::uint64_t emitted=0;double progress=-10;
    auto consume=[&](const SpectralBatch& spectra){
        const auto& frame=spectra.frame;auto& c=*channels.at(frame.channel);
        if(c.segment!=frame.segment){c.noise.reset();c.segment=frame.segment;}
        const bool ready=c.noise.ready();const auto variance=c.noise.estimate(spectra);if(!ready)return;
        const SarBatch batch(capture,spectra,clock);
        const auto* valid=static_cast<const std::uint32_t*>(spectra.transmitted.valid.mapped);
        struct Selection { std::size_t row,interval; std::int64_t bin; unsigned offset; };
        std::vector<Selection> selected;
        for(std::size_t row=std::max<std::size_t>(1,frame.first);row<frame.last;++row){
            const double t=batch.seconds(row)+4095.5/8e6;
            for(std::size_t interval=0;interval<intervals.size();++interval){
                if(t<intervals[interval][0] || t>=intervals[interval][1])continue;
                const auto bin=std::int64_t(std::floor((t-intervals[interval][0])/step));
                if(bin<c.last[interval])throw std::runtime_error("Nonmonotone probe time");
                if(bin>c.last[interval]){c.last[interval]=bin;c.selected_count[interval]=0;}
                if(c.selected_count[interval]>=rows_per_sample ||
                   std::ranges::any_of(std::span{valid+(row-1)*bins,2*std::size_t(bins)},[](auto v){return !v;}))continue;
                selected.push_back({row,interval,bin,c.selected_count[interval]++});
            }
        }
        if(selected.empty())return;
        const auto profiles=compressor.calculate(spectra,variance,capture.channels[frame.channel].guard,nullptr,true,{});
        const auto* s=static_cast<const C*>(spectra.transmitted.carriers.mapped);
        const auto* y=static_cast<const C*>(frame.carriers.measured.mapped);
        const auto* shifts=static_cast<const float*>(spectra.estimate.source_offsets.mapped);
        for(const auto [row,interval,bin,offset]:selected){
            const auto i=row-profiles.first;
            boost::json::object meta{{"channel",frame.channel},{"segment",frame.segment},{"time_s",batch.seconds(row)+4095.5/8e6},
                {"rf_hz",capture.channels[frame.channel].rf_hz},{"guard",capture.channels[frame.channel].guard},
                {"reference_origin",spectra.estimate.reference_origin},{"source_offset",shifts ? shifts[row] : 0},
                {"previous_source_offset",shifts ? shifts[row-1] : 0},{"information",profiles.information},
                {"probe_interval",interval},{"probe_bin",bin},{"probe_offset",offset}};
            record(output,"ROW0",meta,36*bins+16*matched_profile_size);
            plane(output,s+(row-1)*bins,bins);plane(output,s+row*bins,bins);
            plane(output,static_cast<const C*>(spectra.estimate.response.mapped),bins);
            plane(output,static_cast<const float*>(variance.mapped),bins);plane(output,y+row*bins,bins);
            plane(output,profiles.values.data()+i*matched_profile_size,matched_profile_size);
            plane(output,profiles.values.data()+(profiles.count+i)*matched_profile_size,matched_profile_size);++emitted;
        }
    };
    for(;;){
        char tag[4];input.read(tag,4);const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);
        const auto meta=object(input,length);const auto payload=input.tellg();const std::string_view type(tag,4);
        if(bytes>64*1024*1024)throw std::runtime_error("Oversized archive payload");
        if(type=="CLCK")for(const auto& value:meta.at("segments").as_array()){
            const auto& s=value.as_array();clock.push_back({boost::json::value_to<std::uint64_t>(s[0]),boost::json::value_to<double>(s[1]),boost::json::value_to<double>(s[2])});
        }
        if(type=="CHIQ"){
            const auto first=boost::json::value_to<std::uint64_t>(meta.at("first"));const auto n=boost::json::value_to<std::size_t>(meta.at("count"));
            if(first/8e6>=intervals.back()[1]+1)break;
            if(!n || n>config.max_samples || bytes!=count*n*sizeof(C) || boost::json::value_to<std::size_t>(meta.at("channels"))!=count || meta.at("order")!="channel,sample")throw std::runtime_error("Invalid channel IQ payload");
            input.read(static_cast<char*>(mapped),std::streamsize(bytes));pipeline.process({samples,first,n,count,n},consume);
            if(first/8e6>=progress+10){progress=first/8e6;std::cerr<<"Processed "<<progress<<" s; exported "<<emitted<<" probes\n";}
        }else input.seekg(payload+std::streamoff(bytes));
        if(type=="DONE")break;
    }
    pipeline.finish(consume);record(output,"DONE",{{"rows",emitted}},0);output.close();context->gpu.device->unmapMemory(*storage.memory);
    std::cerr<<"Completed "<<emitted<<" finite-reference probes\n";
}catch(const std::exception& e){std::cerr<<"Finite reference export: "<<e.what()<<'\n';return 1;}}
