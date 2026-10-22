// Known waveform kernels aligned to previously saved observation blocks.
// Captured filter weights are retained; only the transmitted QAM is replayed.
#include "gpu_pipeline.hpp"
#include "gpu_commands.hpp"
#include "sar_handoff.hpp"
#include "dsp.hpp"
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace radar;
constexpr std::size_t carriers=6817,fft_size=8192,block_rows=16;
template<class T>T read(std::istream& f){T v{};f.read(reinterpret_cast<char*>(&v),sizeof(v));return v;}
template<class T>void write(std::ostream& f,const T& v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
boost::json::object object(std::istream& f,std::size_t n){
    if(!n || n>1024*1024)throw std::runtime_error("Invalid metadata size");
    std::string s(n,'\0');f.read(s.data(),std::streamsize(n));return boost::json::parse(s).as_object();
}
void record(std::ostream& f,const char* tag,const boost::json::object& meta,std::uint64_t bytes){
    const auto text=boost::json::serialize(meta);f.write(tag,4);write(f,std::uint32_t(text.size()));write(f,bytes);
    f.write(text.data(),std::streamsize(text.size()));
}
struct Row { double time,information;std::uint32_t weight; };
struct Channel {
    std::vector<Row> rows;
    std::array<double,fft_size> spectrum{};
    std::array<bool,block_rows> seen{},known{};
    Fft fft{fft_size};
    std::size_t next{};
    void flush(std::ostream& out,unsigned channel){
        const auto first=next*block_rows;
        double information=0;unsigned expected=0,available=0,matched=0;
        for(std::size_t i=0;i<block_rows;++i){
            information+=rows[first+i].information;
            expected+=rows[first+i].information>0;
            available+=known[i];matched+=seen[i];
        }
        std::vector<C> pulse(fft_size);
        if(information>0)for(std::size_t k=0;k<fft_size;++k)pulse[k]={float(spectrum[k]/information),0};
        fft.transform(pulse,true);
        record(out,"KERN",{{"channel",channel},{"block",next},{"first_s",rows[first].time},
            {"last_s",rows[first+block_rows-1].time},{"information",information},
            {"expected_valid_rows",expected},{"known_rows",available},{"matched_rows",matched},
            {"valid",available==expected && information>0}},129*sizeof(C));
        for(int lag=-64;lag<=64;++lag)write(out,pulse[(lag+fft_size)%fft_size]);
        ++next;spectrum.fill(0);seen.fill(false);known.fill(false);
    }
};
}

int main(int argc,char** argv){try{
    static_assert(std::endian::native==std::endian::little && sizeof(C)==8);
    if(argc!=4)throw std::invalid_argument("waveform_kernel_export CAPTURE.dvblive REQUEST.bin NEW_OUTPUT.kernels");
    std::ifstream requested(argv[2],std::ios::binary);requested.exceptions(std::ios::badbit|std::ios::failbit);
    char magic[8];requested.read(magic,8);
    if(std::string_view(magic,8)!="WFKREQ01")throw std::runtime_error("Invalid kernel request");
    const auto request=object(requested,read<std::uint32_t>(requested));
    if(request.at("block_rows")!=block_rows || request.at("pulse_radius")!=64)throw std::runtime_error("Unsupported kernel layout");
    std::vector<std::unique_ptr<Channel>> channels;double last_time=0;
    const auto weight_count=boost::json::value_to<std::size_t>(request.at("weight_records"));
    if(!weight_count || weight_count>100000)throw std::runtime_error("Invalid weight count");
    for(const auto& value:request.at("channels").as_array()){
        const auto& meta=value.as_object();auto c=std::make_unique<Channel>();
        const auto n=boost::json::value_to<std::size_t>(meta.at("rows"));
        if(!n || n%block_rows || n>10000000 || meta.at("channel")!=channels.size())throw std::runtime_error("Invalid requested rows");
        for(std::size_t i=0;i<n;++i){
            Row row{read<double>(requested),read<double>(requested),read<std::uint32_t>(requested)};
            if(!std::isfinite(row.time+row.information) || row.information<0 || row.weight>=weight_count ||
               (i && row.time<=c->rows.back().time))throw std::runtime_error("Invalid row identity/weight");
            c->rows.push_back(row);
        }
        last_time=std::max(last_time,c->rows.back().time);channels.push_back(std::move(c));
    }
    std::vector<double> weights(weight_count*carriers);
    requested.read(reinterpret_cast<char*>(weights.data()),std::streamsize(weights.size()*sizeof(double)));
    if(std::ranges::any_of(weights,[](double v){return !std::isfinite(v) || v<0;}))throw std::runtime_error("Invalid spectral weight");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    input.read(magic,8);if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Invalid live capture");
    auto header=object(input,read<std::uint32_t>(input));
    if(header.at("archive_mode")!="both" && header.at("archive_mode")!="channels")throw std::runtime_error("Saved channel IQ required");
    if(std::filesystem::file_size(argv[1])!=boost::json::value_to<std::uint64_t>(request.at("capture_size_bytes")))throw std::runtime_error("Capture identity changed");
    SarCapture capture;
    capture.input_rate=boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
    capture.source_rate=boost::json::value_to<std::uint32_t>(header.at("source_nominal_rate_hz"));
    capture.decimation=boost::json::value_to<unsigned>(header.at("decimation"));
    capture.origin=boost::json::value_to<std::uint64_t>(header.at("origin"));
    capture.recording_rf_hz=boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
    GpuPipelineConfig config;config.max_samples=262144;config.frames_per_batch=6;config.residual_spectra=false;
    for(const auto& value:header.at("applied_channels").as_array()){
        const auto& c=value.as_object();const auto guard=boost::json::value_to<unsigned>(c.at("guard"));
        capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),boost::json::value_to<double>(c.at("nco_hz")),guard});config.guards.push_back(guard);
    }
    capture.validate();const auto count=capture.channels.size();
    if(!count || count!=channels.size())throw std::runtime_error("Channel identity mismatch");
    std::ofstream output(argv[3],std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    auto description=request;description["kernel_layout"]="Per-block c64 autocorrelation at lags -64..64; saved captured weights, replayed known symbols.";
    const auto text=boost::json::serialize(description);output.write("WFKERN01",8);write(output,std::uint32_t(text.size()));output.write(text.data(),std::streamsize(text.size()));
    auto context=std::make_shared<gpu::Context>();GpuPipeline pipeline(config,context,{});
    auto storage=gpu::make_buffer(context->gpu,count*config.max_samples*sizeof(C));
    auto* mapped=context->gpu.device->mapMemory(*storage.memory,0,count*config.max_samples*sizeof(C));
    gpu::BufferView samples{context,*storage.handle,count*config.max_samples*sizeof(C),mapped};
    std::vector<ClockSegment> clock;std::size_t known=0;double progress=-10;
    auto consume=[&](const SpectralBatch& spectra){
        const auto& frame=spectra.frame;auto& c=*channels.at(frame.channel);const SarBatch batch(capture,spectra,clock);
        const auto* valid=static_cast<const std::uint32_t*>(spectra.transmitted.valid.mapped);
        const auto* symbols=static_cast<const C*>(spectra.transmitted.carriers.mapped);
        for(std::size_t row=std::max<std::size_t>(1,frame.first);row<frame.last;++row){
            const double t=batch.seconds(row)+4095.5/8e6;
            const auto found=std::lower_bound(c.rows.begin(),c.rows.end(),t-2e-8,[](const auto& a,double b){return a.time<b;});
            if(found==c.rows.end() || std::abs(found->time-t)>2e-8)continue;
            const auto index=std::size_t(found-c.rows.begin()),block=index/block_rows,local=index%block_rows;
            if(block<c.next)throw std::runtime_error("Nonmonotone reconstructed rows");
            while(c.next<block)c.flush(output,unsigned(frame.channel));
            if(c.seen[local])throw std::runtime_error("Duplicate reconstructed row");
            c.seen[local]=true;
            if(found->information<=0 || std::ranges::any_of(std::span{valid+(row-1)*carriers,2*carriers},[](auto v){return !v;}))continue;
            const auto* weight=weights.data()+found->weight*carriers;
            for(std::size_t k=0;k<carriers;++k)c.spectrum[(k+fft_size-3408)%fft_size]+=weight[k]*std::norm(std::complex<double>(symbols[row*carriers+k]));
            c.known[local]=true;++known;
        }
    };
    for(;;){
        char tag[4];input.read(tag,4);const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);
        const auto meta=object(input,length);const auto payload=input.tellg();const std::string_view type(tag,4);
        if(bytes>64*1024*1024)throw std::runtime_error("Oversized capture record");
        if(type=="CLCK")for(const auto& value:meta.at("segments").as_array()){
            const auto& s=value.as_array();clock.push_back({boost::json::value_to<std::uint64_t>(s[0]),boost::json::value_to<double>(s[1]),boost::json::value_to<double>(s[2])});
        }
        if(type=="CHIQ"){
            const auto first=boost::json::value_to<std::uint64_t>(meta.at("first"));const auto n=boost::json::value_to<std::size_t>(meta.at("count"));
            if(first/8e6>last_time+1)break;
            if(!n || n>config.max_samples || bytes!=count*n*sizeof(C) || meta.at("channels")!=count || meta.at("order")!="channel,sample")throw std::runtime_error("Invalid channel IQ");
            input.read(static_cast<char*>(mapped),std::streamsize(bytes));pipeline.process({samples,first,n,count,n},consume);
            if(first/8e6>=progress+10){progress=first/8e6;std::cerr<<"Processed "<<progress<<" s; known requested rows "<<known<<'\n';}
        }else input.seekg(payload+std::streamoff(bytes));
        if(type=="DONE")break;
    }
    pipeline.finish(consume);std::size_t blocks=0;
    for(unsigned i=0;i<count;++i){auto& c=*channels[i];while(c.next<c.rows.size()/block_rows)c.flush(output,i);blocks+=c.next;}
    record(output,"DONE",{{"blocks",blocks},{"known_rows",known}},0);output.close();context->gpu.device->unmapMemory(*storage.memory);
    std::cerr<<"Completed "<<blocks<<" waveform kernels from "<<known<<" known rows\n";
}catch(const std::exception& e){std::cerr<<"Waveform kernel export: "<<e.what()<<'\n';return 1;}}
