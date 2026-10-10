#include "range_doppler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

using namespace radar;
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    constexpr double tau=2*std::numbers::pi,dt=.004352;
    // Exact-bin signed tones: verifies phase, normalization and ordering independently.
    std::vector<RangeProfile> rows(1000);
    for(std::size_t t=0;t<rows.size();++t){
        rows[t].time=3+t*dt;auto values=std::make_shared<std::vector<C>>(4);
        (*values)[1]=std::polar(.25F,static_cast<float>(.73+tau*79*t/1000));
        (*values)[3]=std::polar(.125F,static_cast<float>(-.29-tau*121*t/1000));
        rows[t].values=std::move(values);
    }
    auto map=make_doppler_map(rows,dt,100);
    const auto positive=static_cast<std::size_t>(std::llround((79/(1000*dt)-map.frequency_start)/map.frequency_step));
    const auto negative=static_cast<std::size_t>(std::llround((-121/(1000*dt)-map.frequency_start)/map.frequency_step));
    check(std::abs(map.values[map.frequencies+positive]-std::polar(.25F,.73F))<2e-6,"Positive Doppler complex phase or normalization wrong");
    check(std::abs(map.values[3*map.frequencies+negative]-std::polar(.125F,-.29F))<2e-6,"Negative Doppler complex phase or normalization wrong");
    // Strong stationary clutter and much weaker signed moving echoes. Removing
    // DC must remove its whole Hann leakage without rotating the moving tone.
    for(auto& row:rows) {
        auto values=std::make_shared<std::vector<C>>(*row.values);
        for(auto& value:*values)value+=C{8,-4};
        row.values=values;
    }
    const auto clean=make_doppler_map(rows,dt,100,true);
    check(clean.dc_removed && clean.fft_size==1000,"DC option or FFT size missing from map");
    check(clean.json().find("\"dc_removed\":true")!=std::string::npos,"DC status missing from wire metadata");
    for(std::size_t k=0;k<clean.frequencies;++k)
        check(std::abs(clean.values[k])<1e-8,"Stationary Hann leakage was not removed");
    for(std::size_t r=0;r<clean.ranges;++r)
        check(std::abs(clean.values[r*clean.frequencies+clean.frequencies/2])<1e-7,"Weighted DC bin was not removed");
    check(std::abs(clean.values[clean.frequencies+positive]-std::polar(.25F,.73F))<2e-6,"DC removal changed positive echo phase/gain");
    check(std::abs(clean.values[3*clean.frequencies+negative]-std::polar(.125F,-.29F))<2e-6,"DC removal changed negative echo phase/gain");
    // Non-bin-centred echo with FFT padding: an independent direct DFT of the
    // weighted-mean-subtracted samples checks all complex bins, including near DC.
    std::vector<RangeProfile> padded(1013);
    std::vector<double> weights(padded.size());std::complex<double> mean{};double weight_sum=0;
    for(std::size_t t=0;t<padded.size();++t) {
        const auto value=C{2,-3}+std::polar(.03F,static_cast<float>(.41+tau*2.3*t/padded.size()));
        padded[t]={std::make_shared<const std::vector<C>>(1,value),17+t*dt,0,0};
        weights[t]=static_cast<float>(.5-.5*std::cos(tau*t/padded.size()));
        mean+=std::complex<double>(value)*weights[t];weight_sum+=weights[t];
    }
    mean/=weight_sum;
    const auto padded_map=make_doppler_map(padded,dt,100,true);
    check(padded_map.fft_size>padded.size(),"Test must exercise padded Hann leakage");
    for(std::size_t k=0;k<padded_map.frequencies;++k) {
        std::complex<double> expected{};
        const double hz=padded_map.frequency_start+k*padded_map.frequency_step;
        for(std::size_t t=0;t<padded.size();++t)
            expected+=(std::complex<double>((*padded[t].values)[0])-mean)*weights[t]*std::polar(1.0,-tau*hz*t*dt);
        expected/=weight_sum;
        check(std::abs(std::complex<double>(padded_map.values[k])-expected)<1e-8,"DC removal changed padded complex DFT phase or amplitude");
    }
    double score;
    check(coherent_suffix(rows,.996,score)==1000 && score>1-1e-12,"Stable phase should allow the full history");
    for(std::size_t t=0;t<rows.size();++t)rows[t].phase=.004*t;
    const auto shortened=coherent_suffix(rows,.996,score);
    check(shortened>50 && shortened<100 && score>=.996,"Phase drift must shorten coherent integration");
    for(auto& r:rows){r.timing=r.phase*8192/(tau*3408);r.phase=0;}
    check(coherent_suffix(rows,.996,score)==shortened,"Timing drift at band edges must constrain coherence too");
    // End-to-end active carrier IFFT -> zero-phase slow-time filter -> Doppler FFT.
    const auto path=std::filesystem::temp_directory_path()/"radar-map-test";
    std::filesystem::remove_all(path);
    MapOptions options;options.range_metres=2000;options.history_seconds=20;options.update_seconds=100;
    RangeDoppler pipeline(options,dt/4,path);
    std::vector<C> cross(6817);
    constexpr double hz=18.152573529411764; // 79 / (1000 * .004352)
    constexpr std::size_t symbols=4509; // 513 FIR taps -> exactly 1000 output profiles
    for(std::size_t t=0;t<symbols;++t){
        const auto a=std::polar(.25F,static_cast<float>(.73+tau*hz*t*dt/4));
        const auto alias=std::polar(.25F,static_cast<float>(tau*(hz+1/dt)*t*dt/4));
        for(std::size_t k=0;k<cross.size();++k){
            cross[k]=a*std::polar(1.0F,static_cast<float>(-tau*(static_cast<double>(k)-3408)*12/8192));
            cross[k]+=alias*std::polar(1.0F,static_cast<float>(-tau*(static_cast<double>(k)-3408)*38/8192));
        }
        pipeline.append(cross,t*dt/4,0,0,true,0);
    }
    pipeline.finish();
    std::ifstream file(path/"latest.rdm",std::ios::binary);char magic[4];std::uint32_t header;std::uint64_t bytes;
    file.read(magic,4);file.read(reinterpret_cast<char*>(&header),4);file.read(reinterpret_cast<char*>(&bytes),8);
    check(std::string(magic,4)=="RDM1","Map wire magic wrong");
    std::string json(header,' ');file.read(json.data(),header);
    std::vector<C> result(bytes/sizeof(C));file.read(reinterpret_cast<char*>(result.data()),bytes);
    const auto nrange=static_cast<std::size_t>(options.range_metres/(299792458.0/8e6))+1;
    const auto nfrequency=result.size()/nrange;
    const auto peak=std::max_element(result.begin(),result.end(),[](C a,C b){return std::norm(a)<std::norm(b);})-result.begin();
    check(peak/static_cast<std::ptrdiff_t>(nfrequency)==12,"Excess delay must be positive and must not include FIR group delay");
    const double expected_time=256*dt/4;
    const auto expected=std::polar(.25F*6817/8192,static_cast<float>(.73+tau*hz*expected_time));
    check(std::abs(result[peak]-expected)<.0002,"Slow-time filter/decimation phase or gain is incorrect");
    const auto delayed=result[38*nfrequency+static_cast<std::size_t>(peak)%nfrequency];
    // The finite occupied band causes a calculable range sidelobe from the wanted echo.
    C sidelobe{};for(int k=-3408;k<=3408;++k)sidelobe+=std::polar(1.0F,static_cast<float>(tau*k*(38-12)/8192));
    const auto predicted=expected*sidelobe/6817.0F;
    check(std::abs(delayed-predicted)<3e-5,"Slow-time out-of-band tone aliased into the radar map");
    std::filesystem::remove_all(path);
    std::cout<<"PASS: signed Doppler, complex phase, padded DC rejection, delay origin, normalization, anti-alias rejection and coherence gating\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
