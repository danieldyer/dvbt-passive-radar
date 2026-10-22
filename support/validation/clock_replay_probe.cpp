// Counterfactual clock estimation from feedback-removed archived observations.
// No IQ is altered. Inputs carry explicit measurement variances and model Q.
#include "clock_estimator.hpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc,char** argv){try{
    if(argc<3)throw std::invalid_argument("clock_replay_probe INPUT.bin OUTPUT.csv [ADC_white_FM=1e-11 ADC_RW_FM=1e-11 TX_RW_FM=0 max_time=inf]");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::badbit|std::ios::failbit);
    std::array<char,8> magic;input.read(magic.data(),8);if(std::string(magic.data(),8)!="CLKOBS01")throw std::runtime_error("Invalid timing input");
    std::uint32_t channels;double fs;std::uint64_t count;input.read(reinterpret_cast<char*>(&channels),4);input.read(reinterpret_cast<char*>(&fs),8);input.read(reinterpret_cast<char*>(&count),8);
    radar::ClockNoise noise;noise.white_fm=argc>3 ? std::stod(argv[3]) : 1e-11;noise.random_walk_fm=argc>4 ? std::stod(argv[4]) : 1e-11;
    std::vector<radar::ClockNoise> transmitter(channels,radar::ClockNoise{0,argc>5 ? std::stod(argv[5]) : 0,0});
    const double end=argc>6 ? std::stod(argv[6]) : std::numeric_limits<double>::infinity();
    radar::ClockEstimator estimator(channels,fs,noise,transmitter);
    std::ofstream output(argv[2]);output.exceptions(std::ios::badbit|std::ios::failbit);output<<std::setprecision(17)<<"time_s,offset_samples,velocity_samples_per_second,acceleration_samples_per_second_squared,offset_variance,velocity_variance\n";
    std::vector<double> nis(channels),cross(channels),left(channels),right(channels),previous(channels);std::vector<std::uint64_t> samples(channels);
    double next=-1e100,likelihood=0,first=-1;std::uint64_t used=0;
    struct Record{double time,value,variance;std::uint64_t channel;};static_assert(sizeof(Record)==32);
    for(std::uint64_t j=0;j<count;++j){Record row;input.read(reinterpret_cast<char*>(&row),32);if(row.time>end)break;
        if(first<0)first=row.time;
        auto innovation=estimator.observe(row.channel,row.time,row.value,row.variance);const auto c=row.channel;
        if(row.time>=first+10){likelihood+=innovation.log_likelihood;nis[c]+=innovation.normalized_square;++samples[c];++used;
            const double z=innovation.value/std::sqrt(innovation.variance);
            if(samples[c]>1){cross[c]+=previous[c]*z;left[c]+=previous[c]*previous[c];right[c]+=z*z;}previous[c]=z;}
        if(row.time>=next){const auto state=estimator.estimate(row.time);
            output<<row.time<<','<<state.offset<<','<<state.velocity<<','<<state.acceleration<<','<<state.offset_variance<<','<<state.velocity_variance<<'\n';next=row.time+.032;}
    }
    std::cout<<std::setprecision(17)<<"{\"used_after_ten_second_burnin\":"<<used<<",\"mean_log_likelihood\":"<<likelihood/std::max<std::uint64_t>(used,1)<<",\"channels\":[";
    for(unsigned c=0;c<channels;++c){if(c)std::cout<<',';std::cout<<"{\"channel\":"<<c<<",\"observations\":"<<samples[c]<<",\"mean_normalized_innovation_squared\":"<<nis[c]/std::max<std::uint64_t>(samples[c],1)
        <<",\"innovation_lag_one_correlation\":"<<(left[c]*right[c]>0 ? cross[c]/std::sqrt(left[c]*right[c]) : 0)<<'}';}
    std::cout<<"]}\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
