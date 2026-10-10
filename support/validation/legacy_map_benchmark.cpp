#include "range_doppler.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <time.h>
using namespace radar;
namespace {double cpu(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}}
int main(int argc,char** argv){try{
    if(argc!=3)throw std::invalid_argument("legacy_map_benchmark MOMENTS.cf32 OUTPUT_DIRECTORY");
    const auto bytes=std::filesystem::file_size(argv[1]),count=bytes/(2*1024*sizeof(C));
    if(count<100 || bytes!=2*1024*count*sizeof(C))throw std::runtime_error("Invalid moments");
    std::vector<C> input(count*1024);std::ifstream file(argv[1],std::ios::binary);file.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(C));
    std::vector<RangeProfile> rows;for(std::size_t n=0;n<count;++n){
        auto profile=std::make_shared<std::vector<C>>(input.begin()+n*1024,input.begin()+n*1024+801);
        rows.push_back({std::move(profile),n*.001088,0,0});
    }
    double wall=0,active=0;
    for(unsigned repeat=0;repeat<4;++repeat){const auto begin=std::chrono::steady_clock::now();const auto cm=cpu();
        const auto map=make_doppler_map(rows,.001088,200,true);
        if(repeat){wall+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();active+=cpu()-cm;}}
    MapOptions options;options.doppler_hz=200;options.slow_decimation=2;options.integration_seconds=3;options.history_seconds=3;options.update_seconds=100;options.remove_dc=true;
    RangeDoppler legacy(options,.001088,argv[2]);RangeTimings timing;
    std::vector<C> cross(6817);for(unsigned k=0;k<6817;++k)cross[k]=C{float(k%13)/13,float(k%17)/17};
    for(std::size_t n=0;n<count;++n)legacy.append(cross,n*.001088,0,0,true,0,&timing);
    legacy.finish();const auto seconds=count*.001088;
    std::cout<<"{\"shape_rows\":"<<count<<",\"legacy_hann_cpu_ms_per_map\":"<<active*1000/3<<",\"legacy_hann_wall_ms_per_map\":"<<wall*1000/3
        <<",\"legacy_range_ifft_ms_per_s\":"<<timing.inverse*1000/seconds<<",\"legacy_slow_filter_ms_per_s\":"<<timing.filter*1000/seconds
        <<",\"note\":\"801 ranges, native Doppler grid, slow decimation 2; inverse/filter timings exclude map output; standalone map tested at full symbol rate\"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
