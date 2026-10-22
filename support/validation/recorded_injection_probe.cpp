// Paired real-background point-target injection, before and after compression.
// This is offline validation tooling; no live-pipeline behavior is changed.
#include "gpu_pipeline.hpp"
#include "gpu_compute.hpp"
#include "gpu_noise.hpp"
#include "matched_profiles.hpp"
#include "sar_handoff.hpp"
#include "fractional_resampler.hpp"
#include "dsp.hpp"
#include <boost/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numbers>
#include <optional>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace radar;
namespace {
constexpr double fs=8e6,light=299792458.,tau=2*std::numbers::pi;
constexpr unsigned bins=6817,nfft=8192;
template<class T>T read(std::istream& f){T v{};f.read(reinterpret_cast<char*>(&v),sizeof(v));return v;}
template<class T>void write(std::ostream& f,const T& v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));}
boost::json::object object(std::istream& f,std::size_t n){
    if(!n || n>1024*1024)throw std::runtime_error("Invalid JSON record size");
    std::string s(n,'\0');f.read(s.data(),std::streamsize(n));return boost::json::parse(s).as_object();
}
void record(std::ostream& f,const char* tag,const boost::json::object& meta,std::uint64_t bytes){
    const auto s=boost::json::serialize(meta);f.write(tag,4);write(f,std::uint32_t(s.size()));write(f,bytes);f.write(s.data(),std::streamsize(s.size()));
}
using V=std::array<double,3>;
V vector(const boost::json::value& value){V v{};for(unsigned j=0;j<3;++j)v[j]=boost::json::value_to<double>(value.as_array()[j]);return v;}
double distance(V a,V b){double s=0;for(unsigned j=0;j<3;++j)s+=(a[j]-b[j])*(a[j]-b[j]);return std::sqrt(s);}
struct Plan {
    double first,last,mid;V receiver;std::array<V,3> rotation,state;
    std::vector<V> transmitters;std::vector<C> gains;
    std::optional<std::array<double,5>> rigid;
    std::optional<V> translation_direction;
    V reference_transmitter{};
    explicit Plan(const boost::json::object& p):first(boost::json::value_to<double>(p.at("interval_s").as_array()[0])),
        last(boost::json::value_to<double>(p.at("interval_s").as_array()[1])),mid(boost::json::value_to<double>(p.at("midpoint_s"))),receiver(vector(p.at("receiver_ecef_m"))){
        for(unsigned i=0;i<3;++i){rotation[i]=vector(p.at("ecef_to_enu").as_array()[i]);state[i]=vector(p.at("state_enu").as_array()[i]);}
        for(const auto& t:p.at("transmitters_ecef_m").as_array())transmitters.push_back(vector(t));
        for(const auto& a:p.at("gains_real_imag").as_array())gains.emplace_back(boost::json::value_to<float>(a.as_array()[0]),boost::json::value_to<float>(a.as_array()[1]));
        if(last-first<3 || gains.size()!=transmitters.size())throw std::invalid_argument("Invalid injection plan");
        if(const auto* value=p.if_contains("rigid_parameters")){
            const auto& values=value->as_array();if(values.size()!=5)throw std::invalid_argument("Five rigid motion parameters required");
            rigid.emplace();for(unsigned i=0;i<5;++i){(*rigid)[i]=boost::json::value_to<double>(values[i]);if(!std::isfinite((*rigid)[i]))throw std::invalid_argument("Nonfinite rigid motion");}
            reference_transmitter=vector(p.at("reference_transmitter_ecef_m"));
            if(const auto* direction=p.if_contains("translation_direction_ecef");direction && !direction->is_null()){
                translation_direction=vector(*direction);double norm=0;
                for(double value:*translation_direction){if(!std::isfinite(value))throw std::invalid_argument("Nonfinite translation direction");norm+=value*value;}
                if(!(norm>0))throw std::invalid_argument("Zero translation direction");
            }
        }
    }
    V centre(double time) const {
        const double dt=time-mid;V p=receiver;
        for(unsigned j=0;j<3;++j)for(unsigned k=0;k<3;++k)p[j]+=(state[0][k]+dt*state[1][k]+.5*dt*dt*state[2][k])*rotation[k][j];
        if(rigid){
            V direction{};
            if(translation_direction)direction=*translation_direction;
            else{
                for(const V antenna:{reference_transmitter,receiver}){
                    const double length=distance(p,antenna);for(unsigned j=0;j<3;++j)direction[j]+=(p[j]-antenna[j])/length;
                }
                double vertical=0;for(unsigned j=0;j<3;++j)vertical+=direction[j]*rotation[2][j];
                double norm=0;for(unsigned j=0;j<3;++j){direction[j]-=vertical*rotation[2][j];norm+=direction[j]*direction[j];}
                if(!(norm>0))throw std::invalid_argument("Singular radial translation direction");
                for(double& value:direction)value/=norm;
            }
            // This injection is at the rigid frame's origin. Angular motion
            // affects the imaged offsets, but not this physical point's path.
            const auto& q=*rigid;const double shift=q[0]*dt+.5*q[1]*dt*dt+q[2]*dt*dt*dt/6;
            for(unsigned j=0;j<3;++j)p[j]+=shift*direction[j];
        }
        return p;
    }
    double path(double time,unsigned channel) const {
        const auto p=centre(time);
        return distance(p,transmitters[channel])+distance(p,receiver)-distance(transmitters[channel],receiver);
    }
};
struct Phase {std::uint64_t sample;double phase,frequency,centre_phase;};
struct Source {
    std::ofstream envelope,phase;
    std::uint64_t first{},last{},written{};
    const C* mapped=nullptr;std::size_t bytes=0;int descriptor=-1;
    std::vector<Phase> phases;std::size_t current=0;
    Source(const std::filesystem::path& directory,unsigned channel,std::uint64_t lo,std::uint64_t hi,bool reading):first(lo),last(hi),written(lo){
        const auto base=directory/("channel-"+std::to_string(channel));
        const auto samples=base.string()+".cf32";
        if(reading){
            bytes=std::filesystem::file_size(samples);if(bytes!=(last-first)*sizeof(C))throw std::runtime_error("Source size mismatch");
            descriptor=open(samples.c_str(),O_RDONLY);if(descriptor<0)throw std::runtime_error("Cannot open decoded source");
            const auto p=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,descriptor,0);if(p==MAP_FAILED)throw std::runtime_error("Cannot map decoded source");mapped=static_cast<const C*>(p);
            std::ifstream f(base.string()+".phase",std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);
            const auto count=std::filesystem::file_size(base.string()+".phase")/sizeof(Phase);phases.resize(count);
            f.read(reinterpret_cast<char*>(phases.data()),std::streamsize(count*sizeof(Phase)));
            if(phases.size()<100)throw std::runtime_error("Insufficient carrier phase coverage");
        }else{
            envelope.open(samples,std::ios::binary|std::ios::noreplace);envelope.exceptions(std::ios::badbit|std::ios::failbit);
            std::filesystem::resize_file(samples,(last-first)*sizeof(C));
            phase.open(base.string()+".phase",std::ios::binary|std::ios::noreplace);phase.exceptions(std::ios::badbit|std::ios::failbit);
        }
    }
    ~Source(){if(mapped)munmap(const_cast<C*>(mapped),bytes);if(descriptor>=0)close(descriptor);}
    double carrier(double n){
        while(current+1<phases.size() && n>=double(phases[current+1].sample))++current;
        const auto& p=phases[current];return p.centre_phase-p.phase+tau*p.frequency*(n-double(p.sample))/fs;
    }
    C sample(double n,const FractionalResampler& filter) const {
        const double local=n-double(first);const auto i=std::int64_t(std::floor(local));
        if(i<47 || i+49>=std::int64_t(last-first))throw std::runtime_error("Injection requests unavailable decoded waveform support");
        return filter.interpolate(mapped+i-47,local-double(i));
    }
};
struct Channel {GpuNoise noise;std::unique_ptr<Source> source;std::uint64_t rows=0,good=0,bad=0,checksum=0;unsigned segment=UINT32_MAX;explicit Channel(std::shared_ptr<gpu::Context> c):noise(std::move(c)){};};
}

int main(int argc,char** argv){try{
    if(argc>=4 && std::string_view(argv[1])=="--paths"){
        std::ifstream input(argv[2]);const std::string text((std::istreambuf_iterator<char>(input)),{});
        const Plan plan(boost::json::parse(text).as_object());boost::json::array rows;
        for(int i=3;i<argc;++i){const double time=std::stod(argv[i]);boost::json::array row{time};for(unsigned c=0;c<plan.gains.size();++c)row.emplace_back(plan.path(time,c));rows.emplace_back(std::move(row));}
        std::cout<<boost::json::serialize(rows)<<'\n';return 0;
    }
    if(argc!=6)throw std::invalid_argument("recorded_injection_probe prepare|baseline|linear|inject CAPTURE PLAN.json SOURCE_DIRECTORY OUTPUT_DIRECTORY");
    const std::string_view mode=argv[1];
    if(mode!="prepare" && mode!="baseline" && mode!="linear" && mode!="inject")throw std::invalid_argument("Invalid injection mode");
    const bool preparing=mode=="prepare",injected=mode=="inject",linear=mode=="linear";
    std::ifstream plan_file(argv[3]);const std::string plan_text((std::istreambuf_iterator<char>(plan_file)),{});
    const auto plan_json=boost::json::parse(plan_text).as_object();const Plan plan(plan_json);
    const bool cancel_direct=!plan_json.if_contains("cancel_direct") || plan_json.at("cancel_direct").as_bool();
    const double processing_first=plan_json.if_contains("processing_first_s") ? boost::json::value_to<double>(plan_json.at("processing_first_s")) : plan.first-3;
    if(!std::isfinite(processing_first) || processing_first<0 || processing_first>plan.first-3)throw std::invalid_argument("Processing prefix must include the acquisition warmup");
    unsigned range_first=320,range_count=161;
    if(const auto* range=plan_json.if_contains("profile_range_samples")){
        const auto& values=range->as_array();
        if(values.size()!=2)throw std::invalid_argument("Profile range needs [first, count]");
        range_first=boost::json::value_to<unsigned>(values[0]);range_count=boost::json::value_to<unsigned>(values[1]);
    }
    if(!range_count || range_first>=1024 || range_count>1024-range_first)throw std::invalid_argument("Profile range outside retained lag support");
    const std::filesystem::path source_directory=argv[4],output_directory=argv[5];
    if(!std::filesystem::create_directories(output_directory))throw std::runtime_error("Use a new output directory");
    if(preparing && !std::filesystem::create_directories(source_directory))throw std::runtime_error("Use a new source directory");
    std::ifstream input(argv[2],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);char magic[8];input.read(magic,8);
    if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Invalid capture archive");
    auto header=object(input,read<std::uint32_t>(input));
    if(header.at("archive_mode")!="both" && header.at("archive_mode")!="channels")throw std::runtime_error("Saved channel IQ required");
    SarCapture capture;capture.input_rate=boost::json::value_to<unsigned>(header.at("corrected_rate_hz"));capture.source_rate=boost::json::value_to<unsigned>(header.at("source_nominal_rate_hz"));
    capture.decimation=boost::json::value_to<unsigned>(header.at("decimation"));capture.origin=boost::json::value_to<std::uint64_t>(header.at("origin"));
    capture.recording_rf_hz=boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
    GpuPipelineConfig config;config.max_samples=262144;config.frames_per_batch=6;config.residual_spectra=false;
    for(const auto& value:header.at("applied_channels").as_array()){
        const auto& c=value.as_object();const auto guard=boost::json::value_to<unsigned>(c.at("guard"));
        capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),boost::json::value_to<double>(c.at("nco_hz")),guard});config.guards.push_back(guard);
    }
    capture.validate();const auto count=capture.channels.size();if(count!=plan.gains.size())throw std::runtime_error("Plan/capture channel mismatch");
    const double epoch=double(capture.origin)/capture.input_rate;
    const auto source_first=std::uint64_t((plan.first-1-epoch)*fs),source_last=std::uint64_t((plan.last+1-epoch)*fs);
    auto context=std::make_shared<gpu::Context>();GpuPipeline pipeline(config,context,{});GpuMatchedProfiles compressor(context);
    std::vector<std::unique_ptr<Channel>> channels;
    for(unsigned c=0;c<count;++c){
        auto ch=std::make_unique<Channel>(context);
        if(mode!="baseline")ch->source=std::make_unique<Source>(source_directory,c,source_first,source_last,!preparing);
        channels.push_back(std::move(ch));
    }
    auto storage=gpu::make_buffer(context->gpu,count*config.max_samples*sizeof(C));
    auto* mapped=static_cast<C*>(context->gpu.device->mapMemory(*storage.memory,0,count*config.max_samples*sizeof(C)));
    const gpu::BufferView samples{context,*storage.handle,count*config.max_samples*sizeof(C),mapped};
    std::ofstream output(output_directory/"profiles.bin",std::ios::binary|std::ios::noreplace);output.exceptions(std::ios::badbit|std::ios::failbit);
    header["plan"]=plan_json;header["injected"]=injected || linear;header["injection_level"]=std::string(mode);header["first_range_sample"]=range_first;header["range_count"]=range_count;
    const auto text=boost::json::serialize(header);output.write("INJPROF1",8);write(output,std::uint32_t(text.size()));output.write(text.data(),std::streamsize(text.size()));
    Fft fft(nfft);std::vector<C> wave(nfft),pulse(nfft);
    FractionalResampler interpolation;double progress=plan.first-10,max_path_interpolation=0;
    auto echo_samples=[&](unsigned c,std::uint64_t first,std::span<C> destination,bool add){
        auto& source=*channels[c]->source;
        for(std::size_t j=0;j<destination.size();j+=64){
            const auto stop=std::min(destination.size(),j+64);const double at=double(first+j),t=epoch+at/fs;
            const double lo=plan.path(t,c),hi=plan.path(t+double(stop-j)/fs,c),step=(hi-lo)/(stop-j);
            max_path_interpolation=std::max(max_path_interpolation,std::abs(plan.path(t+.5*double(stop-j)/fs,c)-.5*(lo+hi)));
            for(std::size_t k=j;k<stop;++k){
                const double sample=double(first+k),path=lo+step*(k-j);
                const auto echo=source.sample(sample-path*fs/light,interpolation)*plan.gains[c]*static_cast<C>(std::polar(1.,source.carrier(sample)-tau*capture.channels[c].rf_hz*path/light));
                if(add)destination[k]+=echo;else destination[k]=echo;
            }
        }
    };
    auto echo_storage=gpu::make_buffer(context->gpu,linear ? 1020*bins*sizeof(C) : sizeof(C));
    auto* echo_mapped=static_cast<C*>(context->gpu.device->mapMemory(*echo_storage.memory,0,linear ? 1020*bins*sizeof(C) : sizeof(C)));
    const gpu::BufferView echo_view{context,*echo_storage.handle,linear ? 1020*bins*sizeof(C) : sizeof(C),echo_mapped};
    auto consume=[&](const SpectralBatch& batch){
        const auto c=batch.frame.channel;auto& channel=*channels[c];auto* source=channel.source.get();
        if(channel.segment!=batch.frame.segment){channel.noise.reset();channel.segment=batch.frame.segment;}
        for(const auto& word:batch.packets.packets){if(word.valid){++channel.good;for(auto b:word.bytes)channel.checksum=channel.checksum*1099511628211ULL^b;}else ++channel.bad;}
        const bool ready=channel.noise.ready();const auto noise=channel.noise.estimate(batch);if(!ready)return;
        auto profiles=compressor.calculate(batch,noise,capture.channels[c].guard,nullptr,cancel_direct);
        if(linear){
            std::fill_n(echo_mapped,batch.frame.symbols.size()*bins,C{});
            for(std::size_t row=profiles.first;row<profiles.first+profiles.count;++row){
                const auto& w=batch.frame.symbols[row].window;
                const double t=epoch+(double(w.sample)+4095.5)/fs;
                if(t<plan.first || t>=plan.last)continue;
                echo_samples(c,w.sample,wave,false);
                for(unsigned j=0;j<nfft;++j)wave[j]*=static_cast<C>(std::polar(1.,w.phase-tau*w.frequency_hz*j/fs));
                fft.transform(wave);
                for(unsigned k=0;k<bins;++k)echo_mapped[row*bins+k]=wave[(k+nfft-3408)%nfft];
            }
            auto frame=batch.frame;frame.carriers.measured=echo_view;
            const SpectralBatch echo{frame,{},batch.transmitted,batch.estimate,batch.residual};
            const auto addition=compressor.calculate(echo,noise,capture.channels[c].guard,nullptr,false);
            if(profiles.first!=addition.first || profiles.count!=addition.count)throw std::runtime_error("Linear profile row identities changed");
            for(std::size_t i=0;i<profiles.values.size();++i)profiles.values[i]+=addition.values[i];
        }
        const auto* x=static_cast<const C*>(batch.transmitted.carriers.mapped);const auto* valid=static_cast<const std::uint32_t*>(batch.transmitted.valid.mapped);
        const auto* h=static_cast<const C*>(batch.estimate.response.mapped);const auto* variance=static_cast<const float*>(noise.mapped);
        const auto* offsets=static_cast<const float*>(batch.estimate.source_offsets.mapped);
        boost::json::array rows;std::vector<C> values,psfs;
        for(std::size_t row=profiles.first;row<profiles.first+profiles.count;++row){
            const auto& symbol=batch.frame.symbols[row];const auto& w=symbol.window;
            const double t=epoch+(double(w.sample)+4095.5)/fs;
            const bool known=!std::ranges::any_of(std::span{valid+(row-1)*bins,2*std::size_t(bins)},[](auto v){return v==0;});
            if(preparing && double(w.sample)+batch.estimate.reference_origin>=double(source->first)-512 && w.sample<source->last){
                if(!known && t>=plan.first-.05 && t<=plan.last+.05)throw std::runtime_error("Decoded source has an erasure in the injection interval");
                if(known){
                    const double origin=batch.estimate.reference_origin-(offsets ? offsets[row] : 0),angle=std::arg(std::complex<double>(h[3408]));
                    const Phase phase{w.sample,w.phase,w.frequency_hz,angle};write(source->phase,phase);
                    std::fill(wave.begin(),wave.end(),C{});
                    for(unsigned k=0;k<bins;++k)wave[(k+nfft-3408)%nfft]=h[k]*x[row*bins+k]*static_cast<C>(std::polar(1.,-tau*(int(k)-3408)*origin/nfft-angle));
                    fft.transform(wave,true);
                    const auto begin=std::max<std::int64_t>(source->first,std::int64_t(std::ceil(double(w.sample)+origin-512)));
                    const auto end=std::min<std::uint64_t>(source->last,std::uint64_t(std::ceil(double(w.sample)+origin+nfft)));
                    if(end>std::uint64_t(begin)){
                        const auto from=source->written==source->first ? std::uint64_t(begin) : source->written;
                        if(from<end){
                            if(from+4<std::uint64_t(begin))throw std::runtime_error("Unexpected gap in reconstructed source symbols");
                            std::vector<C> block(end-from);
                            for(std::uint64_t n=from;n<end;++n)block[n-from]=wave[std::uint64_t(std::int64_t(n)-std::int64_t(w.sample)+2*nfft)%nfft];
                            source->envelope.seekp(std::streamoff((from-source->first)*sizeof(C)));source->envelope.write(reinterpret_cast<const char*>(block.data()),std::streamsize(block.size()*sizeof(C)));source->written=end;
                        }
                    }
                }
            }
            if(t<plan.first || t>=plan.last)continue;
            rows.emplace_back(boost::json::array{w.sample,t,w.phase,w.frequency_hz,known,offsets ? offsets[row] : 0});
            const auto i=row-profiles.first;
            for(unsigned moment=0;moment<2;++moment)values.insert(values.end(),profiles.values.begin()+std::ptrdiff_t((moment*profiles.count+i)*1024+range_first),profiles.values.begin()+std::ptrdiff_t((moment*profiles.count+i)*1024+range_first+range_count));
            std::fill(pulse.begin(),pulse.end(),C{});
            for(unsigned k=0;k<bins;++k)pulse[(k+nfft-3408)%nfft]={float(std::norm(h[k])*std::norm(x[row*bins+k])/std::max(double(variance[k]),1e-30)/profiles.information),0};
            fft.transform(pulse,true);for(int lag=-64;lag<=64;++lag)psfs.push_back(pulse[(lag+nfft)%nfft]);
            ++channel.rows;
        }
        if(!rows.empty()){
            record(output,"PROF",{{"channel",c},{"rows",std::move(rows)},{"information",profiles.information}},(values.size()+psfs.size())*sizeof(C));
            output.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(C)));output.write(reinterpret_cast<const char*>(psfs.data()),std::streamsize(psfs.size()*sizeof(C)));
        }
    };
    for(;;){
        char tag[4];input.read(tag,4);const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);const auto meta=object(input,length);const auto payload=input.tellg();
        const std::string_view type(tag,4);
        if(type=="CHIQ"){
            const auto first=boost::json::value_to<std::uint64_t>(meta.at("first"));const auto n=boost::json::value_to<unsigned>(meta.at("count"));const double time=epoch+first/fs;
            if(time>=plan.last+1)break;
            if(time<processing_first){input.seekg(payload+std::streamoff(bytes));continue;}
            if(!n || n>config.max_samples || bytes!=count*n*sizeof(C))throw std::runtime_error("Invalid IQ payload");
            input.read(reinterpret_cast<char*>(mapped),std::streamsize(bytes));
            if(injected && first>=source_first+2048 && first+n<source_last-2048){
                for(unsigned c=0;c<count;++c)echo_samples(c,first,{mapped+c*n,n},true);
            }
            pipeline.process({samples,first,n,count,n},consume);
            // A deterministic offline feedback schedule makes the paired
            // comparison independent of decoder-worker completion timing.
            if(!preparing)pipeline.drain(consume);
            if(time>=progress+5){progress=time;std::cerr<<mode<<' '<<time<<" s\n";}
        }else input.seekg(payload+std::streamoff(bytes));
        if(type=="DONE")break;
    }
    pipeline.finish(consume);record(output,"DONE",{},0);output.close();
    boost::json::array statistics;
    for(unsigned c=0;c<count;++c){const auto& ch=*channels[c];statistics.emplace_back(boost::json::object{{"channel",c},{"rows",ch.rows},{"rs_good",ch.good},{"rs_bad",ch.bad},{"packet_checksum",ch.checksum}});}
    boost::json::object summary{{"injected",injected || linear},{"mode",std::string(mode)},{"deterministic_feedback",!preparing},{"channels",statistics},{"source_first",source_first},{"source_last",source_last},{"max_path_interpolation_error_m",max_path_interpolation},{"original_capture_modified",false}};
    std::ofstream metadata(output_directory/"summary.json");metadata<<boost::json::serialize(summary)<<'\n';
    if(preparing){std::ofstream description(source_directory/"metadata.json");description<<boost::json::serialize(summary)<<'\n';}
    if(max_path_interpolation>1e-6)throw std::runtime_error("Point-path interpolation exceeds numerical phase budget");
    context->gpu.device->unmapMemory(*echo_storage.memory);context->gpu.device->unmapMemory(*storage.memory);std::cout<<boost::json::serialize(summary)<<'\n';
}catch(const std::exception& e){std::cerr<<"Recorded injection: "<<e.what()<<'\n';return 1;}}
