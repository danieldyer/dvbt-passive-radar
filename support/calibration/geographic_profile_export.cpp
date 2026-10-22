// Offline zero-Doppler image input, replayed from saved channel IQ. The
// production waveform reconstruction and finite matched filter are unchanged.
// Only the fitted direct path is cancelled; stationary terrain is retained.
#include "gpu_pipeline.hpp"
#include "gpu_commands.hpp"
#include "gpu_noise.hpp"
#include "matched_profiles.hpp"
#include "sar_handoff.hpp"
#include "dsp.hpp"
#include <boost/json.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
using namespace radar;
constexpr double block_seconds=30*8704.0/8e6;
template<class T>T read(std::istream& f){T value{};f.read(reinterpret_cast<char*>(&value),sizeof(value));return value;}
template<class T>void write(std::ostream& f,const T& value){f.write(reinterpret_cast<const char*>(&value),sizeof(value));}
boost::json::object json(std::istream& f,std::size_t length){
    if(!length || length>1024*1024)throw std::runtime_error("Invalid archive metadata length");
    std::string text(length,'\0');f.read(text.data(),std::streamsize(length));return boost::json::parse(text).as_object();
}
struct Channel {
    GpuNoise noise;
    std::array<std::complex<double>,1024> sum{};
    std::array<double,8192> spectrum{};
    Fft fft{8192};
    bool save_psf{};
    int psf_radius{64};
    std::uint64_t bin=UINT64_MAX,segment=UINT64_MAX;
    double information{},first{},last{},time{};
    explicit Channel(std::shared_ptr<gpu::Context> context,bool psf,int radius):noise(std::move(context)),save_psf(psf),psf_radius(radius){}
    void flush(std::ostream& output,unsigned channel){
        if(bin==UINT64_MAX)return;
        write(output,channel);write(output,std::uint32_t(0));write(output,bin);write(output,segment);
        write(output,first);write(output,last);write(output,information);
        write(output,information>0 ? time/information : (bin+.5)*block_seconds);
        for(auto value:sum){const C z=information>0 ? static_cast<C>(value/information) : C{};write(output,z);}
        if(save_psf){
            std::vector<C> response(8192);
            if(information>0)for(unsigned k=0;k<8192;++k)response[k]={float(spectrum[k]/information),0};
            fft.transform(response,true);
            for(int lag=-psf_radius;lag<=psf_radius;++lag)write(output,response[(lag+8192)%8192]);
            spectrum.fill(0);
        }
        sum.fill({});information=time=0;bin=UINT64_MAX;
    }
};
}

int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(radar::C)==8);
    if(argc<3 || argc>6)throw std::invalid_argument("geographic_profile_export CAPTURE.dvblive NEW_OUTPUT.gprf [end_seconds=0] [save_psf=0] [psf_radius=64]");
    const double end=argc>=4 ? std::stod(argv[3]) : 0;
    const bool save_psf=argc>=5 && std::stoi(argv[4])!=0;
    const int psf_radius=argc>=6 ? std::stoi(argv[5]) : 64;
    if(end<0 || !std::isfinite(end))throw std::invalid_argument("Invalid end time");
    if(psf_radius<64 || psf_radius>4095)throw std::invalid_argument("PSF radius must be between 64 and 4095 samples");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    std::ofstream output(argv[2],std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    char magic[8];input.read(magic,8);if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Not a live archive");
    auto header=json(input,read<std::uint32_t>(input));
    if(header.at("archive_mode")!="both" && header.at("archive_mode")!="channels")throw std::runtime_error("Saved channel IQ is required to retain static terrain");
    SarCapture capture;
    capture.input_rate=boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
    capture.source_rate=boost::json::value_to<std::uint32_t>(header.at("source_nominal_rate_hz"));
    capture.decimation=boost::json::value_to<unsigned>(header.at("decimation"));
    capture.origin=boost::json::value_to<std::uint64_t>(header.at("origin"));
    capture.recording_rf_hz=boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
    GpuPipelineConfig config;config.max_samples=262144;config.frames_per_batch=6;config.residual_spectra=false;
    for(const auto& value:header.at("applied_channels").as_array()){
        const auto& c=value.as_object();const auto guard=boost::json::value_to<unsigned>(c.at("guard"));
        if(guard!=16)throw std::invalid_argument("This static-image export uses a 30-symbol cadence for guard 1/16");
        capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),boost::json::value_to<double>(c.at("nco_hz")),guard});
        config.guards.push_back(guard);
    }
    capture.validate();const auto count=capture.channels.size();
    if(!count || count>64)throw std::runtime_error("Invalid channel count");
    header["source_capture"]=std::filesystem::absolute(argv[1]).string();
    header["product"]="zero_doppler_profile_block_means";
    header["stationary_subtraction"]=false;header["direct_path_subtraction"]=true;
    header["block_seconds"]=block_seconds;header["profile_size"]=1024;
    header["sample_rate_hz"]=8e6;header["range_spacing_m"]=299792458.0/8e6;
    header["record_layout"]="u32 channel,u32 reserved,u64 bin,u64 segment,f64 first,last,information,time,c64 values[1024]";
    header["psf_samples"]=save_psf ? 2*psf_radius+1 : 0;
    header["psf_lag_radius"]=save_psf ? psf_radius : 0;
    header["psf_layout"]="optional c64[psf_samples] after each record; exact block mean circular template autocorrelation, centred lags -psf_lag_radius..psf_lag_radius";
    header["limitations"]="Coherent block means intended for static terrain. Use original IQ/profiles for moving-target processing.";
    const auto description=boost::json::serialize(header);output.write("GEOProf1",8);write(output,std::uint32_t(description.size()));output.write(description.data(),std::streamsize(description.size()));
    auto context=std::make_shared<gpu::Context>();GpuPipeline pipeline(config,context,{});GpuMatchedProfiles compressor(context);
    std::vector<std::unique_ptr<Channel>> channels;
    for(std::size_t i=0;i<count;++i)channels.push_back(std::make_unique<Channel>(context,save_psf,psf_radius));
    auto storage=gpu::make_buffer(context->gpu,count*config.max_samples*sizeof(C));
    auto* mapped=context->gpu.device->mapMemory(*storage.memory,0,count*config.max_samples*sizeof(C));
    gpu::BufferView samples{context,*storage.handle,count*config.max_samples*sizeof(C),mapped};
    std::vector<ClockSegment> clock;
    std::uint64_t valid_rows=0,erased_rows=0;double progress=-10;
    auto consume=[&](const SpectralBatch& spectra){
        const auto& frame=spectra.frame;auto& c=*channels.at(frame.channel);
        if(c.segment!=frame.segment){c.flush(output,unsigned(frame.channel));c.noise.reset();c.segment=frame.segment;}
        const bool ready=c.noise.ready();const auto variance=c.noise.estimate(spectra);
        if(!ready)return;
        const auto profiles=compressor.calculate(spectra,variance,16,nullptr,true,{});
        std::array<double,6817> weighting{};
        if(save_psf){
            const auto* h=static_cast<const C*>(spectra.estimate.response.mapped);
            const auto* noise=static_cast<const float*>(variance.mapped);
            for(unsigned k=0;k<6817;++k)weighting[k]=std::norm(h[k])/std::max(double(noise[k]),1e-30);
        }
        const SarBatch batch(capture,spectra,clock);
        const auto* flags=static_cast<const std::uint32_t*>(spectra.transmitted.valid.mapped);
        for(std::size_t i=0;i<profiles.count;++i){
            const auto row=profiles.first+i;
            const double t=batch.seconds(row)+4095.5/8e6;
            if(end>0 && t>=end)continue;
            const auto bin=std::uint64_t(std::floor(t/block_seconds));
            if(c.bin!=bin){c.flush(output,unsigned(frame.channel));c.bin=bin;c.first=t;}
            c.last=t;
            const auto known=std::span{flags+(row-1)*6817,2*std::size_t(6817)};
            if(std::ranges::any_of(known,[](auto v){return !v;})){++erased_rows;continue;}
            ++valid_rows;c.information+=profiles.information;c.time+=profiles.information*t;
            for(std::size_t k=0;k<1024;++k)c.sum[k]+=profiles.information*std::complex<double>(profiles.values[i*1024+k]);
            if(save_psf){
                const auto* x=static_cast<const C*>(spectra.transmitted.carriers.mapped)+row*6817;
                for(unsigned k=0;k<6817;++k)c.spectrum[(k+8192-3408)%8192]+=weighting[k]*std::norm(x[k]);
            }
        }
    };
    for(;;){
        char tag[4];input.read(tag,4);const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);
        const auto meta=json(input,length);const auto payload=input.tellg();const std::string_view type(tag,4);
        if(bytes>64*1024*1024)throw std::runtime_error("Oversized archive payload");
        if(type=="CLCK")for(const auto& value:meta.at("segments").as_array()){
            const auto& s=value.as_array();clock.push_back({boost::json::value_to<std::uint64_t>(s[0]),boost::json::value_to<double>(s[1]),boost::json::value_to<double>(s[2])});
        }
        if(type=="CHIQ"){
            const auto first=boost::json::value_to<std::uint64_t>(meta.at("first"));const auto n=boost::json::value_to<std::size_t>(meta.at("count"));
            if(end>0 && first/8e6>=end+1)break;
            if(!n || n>config.max_samples || bytes!=count*n*sizeof(C) || boost::json::value_to<std::size_t>(meta.at("channels"))!=count || meta.at("order")!="channel,sample")
                throw std::runtime_error("Invalid channelised-IQ payload");
            input.read(static_cast<char*>(mapped),std::streamsize(bytes));pipeline.process({samples,first,n,count,n},consume);
            if(first/8e6>=progress+10){progress=first/8e6;std::cerr<<"Processed "<<progress<<" s; valid/erased rows "<<valid_rows<<'/'<<erased_rows<<'\n';}
        }else input.seekg(payload+std::streamoff(bytes));
        if(type=="DONE")break;
    }
    pipeline.finish(consume);for(unsigned i=0;i<count;++i)channels[i]->flush(output,i);
    context->gpu.device->unmapMemory(*storage.memory);output.close();
    std::cerr<<"Completed static profile export; valid/erased rows "<<valid_rows<<'/'<<erased_rows<<'\n';
}catch(const std::exception& e){std::cerr<<"Geographic export error: "<<e.what()<<'\n';return 1;}}
