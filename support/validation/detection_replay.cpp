#include "map_calibration.hpp"
using radar::validation::estimate_map_noise;
using radar::validation::standardize_map;
#include "matched_detector.hpp"
#include "gpu_compute.hpp"
#include "stationary_background.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <time.h>

using namespace radar;
namespace {
template<class T>T get(std::istream& file){T x;file.read(reinterpret_cast<char*>(&x),sizeof(x));if(!file)throw std::runtime_error("Truncated archive");return x;}
template<class T>void read(std::istream& file,T* data,std::size_t n){file.read(reinterpret_cast<char*>(data),n*sizeof(T));if(!file)throw std::runtime_error("Truncated archive plane");}
struct Plane {
    std::shared_ptr<gpu::Context> context;gpu::Buffer buffer;gpu::BufferView view;
    Plane(std::shared_ptr<gpu::Context> c,std::size_t bytes):context(std::move(c)),buffer(gpu::make_buffer(context->gpu,bytes)),
        view{context,*buffer.handle,bytes,context->gpu.device->mapMemory(*buffer.memory,0,bytes)}{}
    ~Plane(){context->gpu.device->unmapMemory(*buffer.memory);}
    template<class T>T* data(){return static_cast<T*>(view.mapped);}
};
double cpu(){timespec t{};clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
}
int main(int argc,char** argv){try {
    if(argc<3)throw std::invalid_argument("detection_replay INPUT.sar OUTPUT_PREFIX [cancel=1] [CPI=3] [capture=10] [stationary=1]");
    const bool cancel=argc<4 || std::stoi(argv[3]);const double cpi=argc>4 ? std::stod(argv[4]) : 3;
    const double capture=argc>5 ? std::stod(argv[5]) : 10;
    const bool stationary=cancel && (argc<7 || std::stoi(argv[6]));
    std::ifstream file(argv[1],std::ios::binary);char magic[8];read(file,magic,8);
    const bool source_clock=std::string_view(magic,8)=="DVBTSAR2";
    if(!source_clock && std::string_view(magic,8)!="DVBTSAR1")throw std::runtime_error("Not a SAR archive");
    const auto rate=get<std::uint32_t>(file),decimation=get<std::uint32_t>(file),channels=get<std::uint32_t>(file);
    get<std::uint32_t>(file);const auto origin=get<std::uint64_t>(file);get<double>(file);get<double>(file);
    if(channels!=1 || rate!=decimation*8'000'000)throw std::runtime_error("Expected one 8 MS/s channel");
    const auto rf=get<double>(file);get<double>(file);const auto guard=get<std::uint32_t>(file);get<std::uint32_t>(file);
    auto context=std::make_shared<gpu::Context>();GpuNoise estimator(context);GpuMatchedProfiles correlator(context);GpuFocus focus(context);StationaryBackground background;
    Plane y(context,1020*6817*sizeof(C)),s(context,1020*6817*sizeof(C)),x(context,1020*6817*sizeof(C)),
        v(context,1020*6817*sizeof(std::uint32_t)),h(context,6817*sizeof(C)),offsets(context,1020*sizeof(float));
    std::vector<std::array<C,2048>> history;std::vector<double> times,information;ComputeTimings noise,profiles,focused;
    const double dt=(8192.0+8192.0/guard)/8e6;const auto count=std::size_t(std::ceil(cpi/dt));focus.prepare(count);
    double begin_time=-1,wall=0,active=0,background_end=-1;std::uint64_t used=0;
    while(file.peek()!=EOF){
        char tag[4];read(file,tag,4);const auto bytes=get<std::uint64_t>(file);const auto next=file.tellg()+std::streamoff(bytes);
        if(std::string_view(tag,4)!="SPEC"){file.seekg(next);continue;}
        const auto channel=get<std::uint32_t>(file),segment=get<std::uint32_t>(file),rows=get<std::uint32_t>(file),
            first=get<std::uint32_t>(file),last=get<std::uint32_t>(file),frames=get<std::uint32_t>(file);
        const auto reference_origin=get<double>(file),error=get<double>(file);
        if(rows>1020 || first>=last || last>rows || bytes!=40+(source_clock ? 48ULL : 32ULL)*rows+92ULL*frames+8ULL*6817+20ULL*rows*6817)
            throw std::runtime_error("Invalid SPEC shape");
        FrameBatch frame;frame.channel=channel;frame.segment=segment;frame.first=first;frame.last=last;frame.symbols.resize(rows);
        for(auto& symbol:frame.symbols){symbol.window.sample=get<std::uint64_t>(file);symbol.window.phase=get<double>(file);
            symbol.window.frequency_hz=get<double>(file);symbol.modulo4=get<std::uint32_t>(file);symbol.discontinuity=get<std::uint32_t>(file);
            if(source_clock){symbol.source_offset_samples=get<double>(file);symbol.source_rate_error=get<double>(file);}}
        for(unsigned r=0;r<rows;++r)offsets.data<float>()[r]=float(frame.symbols[r].source_offset_samples);
        file.seekg(frames*92,std::ios::cur);read(file,h.data<C>(),6817);read(file,y.data<C>(),rows*6817);
        read(file,s.data<C>(),rows*6817);read(file,v.data<std::uint32_t>(),rows*6817);
        for(unsigned r=0;r<rows;++r)for(unsigned k=0;k<6817;++k)x.data<C>()[r*6817+k]=s.data<C>()[r*6817+k]*
            static_cast<C>(std::polar(1.0,-2*std::numbers::pi*(int(k)-3408)*(reference_origin-frame.symbols[r].source_offset_samples)/8192));
        frame.carriers.measured=y.view;frame.carriers.count=rows;
        SpectralBatch batch{frame,{}, {s.view,v.view,rows},{h.view,{},rows,reference_origin,error,offsets.view},
                             {x.view,{},{},h.view,v.view,rows,reference_origin}};
        const auto time_at=[&](std::size_t row){return double(origin+frame.symbols[row].window.sample*decimation)/rate;};
        if(begin_time<0)begin_time=time_at(first);
        const auto mark=std::chrono::steady_clock::now();const double cm=cpu();
        const auto d=estimator.estimate(batch,&noise);
        if(stationary && !background.ready()){
            background.observe(batch,dt,estimator.training());background_end=time_at(rows-1);continue;
        }
        const auto result=correlator.calculate(batch,d,guard,&profiles,cancel,background.response());
        wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();active+=cpu()-cm;
        for(std::size_t n=0;n<result.count;++n){
            const auto row=result.first+n;bool complete=true;
            for(unsigned i=0;i<2*6817;++i)complete&=v.data<std::uint32_t>()[(row-1)*6817+i]!=0;
            if(!complete)continue;
            const double time=time_at(row);if(time<=background_end)continue;
            times.push_back(time);information.push_back(result.information);auto& stored=history.emplace_back();
            for(unsigned moment=0;moment<2;++moment)std::copy_n(result.values.data()+(moment*result.count+n)*1024,1024,stored.data()+moment*1024);
            ++used;
        }
        if(!times.empty() && times.back()-begin_time>=capture)break;
    }
    if(history.size()<count)throw std::runtime_error("No full CPI");
    std::vector<C> input(2*count*1024);std::vector<double> selected(count),selected_information(count);
    for(std::size_t r=0;r<count;++r){selected[r]=times[times.size()-count+r];
        selected_information[r]=information[information.size()-count+r];
        for(unsigned moment=0;moment<2;++moment)std::copy_n(history[history.size()-count+r].data()+moment*1024,1024,input.data()+(moment*count+r)*1024);}
    auto mark=std::chrono::steady_clock::now();double cm=cpu();
    auto map=focus.calculate(input,count,dt,rf,selected.front(),{30000,200,true},&focused,selected_information);
    const double focus_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count(),focus_cpu=cpu()-cm;
    ComputeTimings warm;double warm_wall=0,warm_cpu=0;
    for(unsigned repeat=0;repeat<3;++repeat){
        mark=std::chrono::steady_clock::now();cm=cpu();
        map=focus.calculate(input,count,dt,rf,selected.front(),{30000,200,true},&warm,selected_information);
        warm_wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();warm_cpu+=cpu()-cm;
    }
    mark=std::chrono::steady_clock::now();estimate_map_noise(map);
    const double calibration=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
    if(background.ready()){
        map.stationary_prediction="frozen_source_divided_transfer";map.background_start=begin_time;map.background_end=background_end;
        map.dc_method="frozen_stationary_prediction_and_constant_projection";
    }
    map.write(std::string(argv[2])+".rdm");standardize_map(map);map.write(std::string(argv[2])+"-normalized.rdm");
    std::ofstream moment_file(std::string(argv[2])+".moments.cf32",std::ios::binary),time_file(std::string(argv[2])+".times.f64",std::ios::binary);
    moment_file.write(reinterpret_cast<const char*>(input.data()),input.size()*sizeof(C));time_file.write(reinterpret_cast<const char*>(selected.data()),selected.size()*sizeof(double));
    std::ofstream information_file(std::string(argv[2])+".information.f64",std::ios::binary);
    information_file.write(reinterpret_cast<const char*>(selected_information.data()),selected_information.size()*sizeof(double));
    const double seconds=used*dt,scale=1000/seconds;
    std::cout<<std::setprecision(12)<<"{\"signal_seconds\":"<<seconds<<",\"cancel_direct\":"<<cancel<<",\"frozen_stationary\":"<<stationary<<",\"cpi_s\":"<<count*dt
        <<",\"profile_wall_ms_per_s\":"<<wall*scale<<",\"profile_process_cpu_ms_per_s\":"<<active*scale
        <<",\"gpu_noise_ms_per_s\":"<<noise.gpu*scale<<",\"gpu_profiles_ms_per_s\":"<<profiles.gpu*scale
        <<",\"focus_wall_ms_per_map\":"<<focus_wall*1000<<",\"focus_cpu_ms_per_map\":"<<focus_cpu*1000
        <<",\"gpu_focus_ms_per_map\":"<<focused.gpu*1000<<",\"calibration_cpu_ms_per_map\":"<<calibration*1000<<"}\n";
    std::cerr<<"Warm focus wall/CPU/GPU ms/map: "<<warm_wall*1000/3<<" / "<<warm_cpu*1000/3<<" / "<<warm.gpu*1000/3<<'\n';
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
