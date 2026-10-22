// Replay feedback-removed timing through the production clock controller.
// IQ and archived clock segments are not modified. Binary identity records
// whether a result used the incumbent or a later controller implementation.
#include "sample_clock.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("clock_controller_replay INPUT.bin OUTPUT.csv [lead_s=.002]");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    std::array<char,8> magic;input.read(magic.data(),8);if(std::string(magic.data(),8)!="CLKOBS01")throw std::runtime_error("Invalid clock input");
    std::uint32_t channels;double fs;std::uint64_t count;input.read(reinterpret_cast<char*>(&channels),4);input.read(reinterpret_cast<char*>(&fs),8);input.read(reinterpret_cast<char*>(&count),8);
    std::vector<double> centres(channels);for(unsigned c=0;c<channels;++c)centres[c]=-24.5e6+c*49e6/std::max(1u,channels-1);
    radar::SampleClock clock(fs,centres);std::vector<double> last_time(channels,-1),last_offset(channels);
    std::ofstream output(argv[2]);output.exceptions(std::ios::badbit|std::ios::failbit);output<<std::setprecision(17)<<"time_s,offset_samples,rate_ppm\n";
    const double lead=argc>3 ? std::stod(argv[3]) : .002;double next=-1;std::uint64_t updates=0;
    struct Record{double time,value,variance;std::uint64_t channel;};static_assert(sizeof(Record)==32);
    for(std::uint64_t j=0;j<count;++j){Record row;input.read(reinterpret_cast<char*>(&row),32);
        if(next<0)next=row.time+.032;
        while(row.time>next){const auto n=std::uint64_t(std::llround((next+lead)*fs));clock.update(n);const auto s=clock.at(double(n));
            output<<next+lead<<','<<s.offset_at(double(n))<<','<<(s.rate-1)*1e6<<'\n';next+=.032;++updates;}
        const auto c=row.channel;const double n=row.time*fs;const auto now=clock.at(n);
        if(last_time[c]>=0){const auto before=clock.at(last_time[c]*fs);const double interval=(row.time-last_time[c])*fs;
            const double applied=1+(now.offset_at(n)-before.offset_at(last_time[c]*fs))/interval;
            clock.observe_rate(c,n,applied/(1+(row.value-last_offset[c])/interval)-1);}
        clock.observe_offset(c,n,(now.offset_at(n)-row.value)/now.rate,row.variance/(now.rate*now.rate));last_time[c]=row.time;last_offset[c]=row.value;
    }
    std::cout<<"{\"clock_updates\":"<<updates<<",\"lead_s\":"<<lead<<"}\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
