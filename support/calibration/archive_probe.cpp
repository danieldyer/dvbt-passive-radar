// One-off calibration candidate discovery using the unchanged production focus.
// The output scores select candidates; they are not calibration/noise weights.
#include "../video/profile_history.hpp"
#include "gpu_focus.hpp"
#include "gpu_commands.hpp"
#include <boost/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace radar;
template<class T>T read(std::istream& f){T x{};f.read(reinterpret_cast<char*>(&x),sizeof(x));return x;}
boost::json::object object(std::istream& f,std::size_t n){
    if(!n || n>1024*1024)throw std::runtime_error("Invalid metadata length");
    std::string s(n,'\0');f.read(s.data(),std::streamsize(n));return boost::json::parse(s).as_object();
}
template<class T>T value(const boost::json::object& o,const char* key){return boost::json::value_to<T>(o.at(key));}
struct Channel {
    video::ProfileHistory history;
    GpuFocus focus;
    double interval,rf;
    Channel(std::shared_ptr<gpu::Context> context,unsigned guard,double frequency,double cpi,double update):
        history(cpi,(8192+8192./guard)/8e6,update,true),focus(std::move(context)),
        interval((8192+8192./guard)/8e6),rf(frequency){}
};
struct Peak {std::size_t r,f;float power;};
struct Selection {double threshold_db=14,minimum_range=600,minimum_doppler=0;unsigned maximum_peaks=64;};
boost::json::array peaks(const DopplerMap& map,const C* z,const Selection& select){
    const auto nr=map.ranges,nf=map.frequencies;
    const auto first=std::size_t(std::ceil(select.minimum_range/map.range_step));
    const double threshold=std::pow(10.,select.threshold_db/10);
    std::vector<float> noise(nf),work;work.reserve(nr-first);
    for(std::size_t f=0;f<nf;++f){
        work.clear();for(std::size_t r=first;r<nr;r+=3)work.push_back(std::norm(z[r*nf+f]));
        std::nth_element(work.begin(),work.begin()+work.size()/2,work.end());
        noise[f]=std::max(work[work.size()/2]/float(std::log(2.)),1e-30F);
    }
    std::vector<Peak> candidates;
    for(std::size_t r=first;r+1<nr;++r)for(std::size_t f=1;f+1<nf;++f){
        const double doppler=map.frequency_start+f*map.frequency_step;
        if(std::abs(doppler)<std::max(map.dc_removed ? 2/map.integration : 0.,select.minimum_doppler))continue;
        const auto power=std::norm(z[r*nf+f]);if(power<threshold*noise[f])continue;
        bool maximum=true;
        for(int dr=-1;dr<=1;++dr)for(int df=-1;df<=1;++df)
            if((dr || df) && power<std::norm(z[(r+dr)*nf+f+df]))maximum=false;
        if(maximum)candidates.push_back({r,f,power});
    }
    std::ranges::sort(candidates,[&](auto a,auto b){return a.power/noise[a.f]>b.power/noise[b.f];});
    std::vector<Peak> accepted;boost::json::array result;
    for(auto p:candidates){
        if(std::ranges::any_of(accepted,[&](auto a){return std::abs(double(a.r)-p.r)*map.range_step<150 &&
            std::abs(double(a.f)-p.f)*map.frequency_step<3/map.integration;}))continue;
        auto delta=[&](std::ptrdiff_t stride){
            const auto at=p.r*nf+p.f;
            const double left=std::log(std::max(double(std::norm(z[at-stride])),1e-30));
            const double middle=std::log(std::max(double(p.power),1e-30));
            const double right=std::log(std::max(double(std::norm(z[at+stride])),1e-30));
            const auto curvature=left-2*middle+right;
            return curvature< -1e-12 ? std::clamp(.5*(left-right)/curvature,-.5,.5) : 0.;
        };
        accepted.push_back(p);
        result.emplace_back(boost::json::object{{"range_m",(p.r+delta(std::ptrdiff_t(nf)))*map.range_step},
            {"doppler_hz",map.frequency_start+(p.f+delta(1))*map.frequency_step},
            {"selection_snr_db",10*std::log10(p.power/noise[p.f])},{"power",p.power},
            {"noise_selection_only",noise[p.f]},{"real",z[p.r*nf+p.f].real()},{"imag",z[p.r*nf+p.f].imag()}});
        if(accepted.size()==select.maximum_peaks)break;
    }
    return result;
}
}
int main(int argc,char** argv){try{
    const bool remove_dc=argc<2 || std::string_view(argv[argc-1])!="--keep-dc";
    if(!remove_dc)--argc;
    if(argc<3 || argc>11)throw std::invalid_argument("calibration_archive_probe CAPTURE OUTPUT.jsonl [CPI=1] [UPDATE=1] [END=0] [DOPPLER=100] [THRESHOLD_DB=14] [MIN_RANGE=600] [MIN_DOPPLER=0] [MAX_PEAKS=64] [--keep-dc]");
    const double cpi=argc>3 ? std::stod(argv[3]) : 1,update=argc>4 ? std::stod(argv[4]) : 1,end=argc>5 ? std::stod(argv[5]) : 0;
    const double doppler=argc>6 ? std::stod(argv[6]) : 100;
    const Selection select{argc>7 ? std::stod(argv[7]) : 14,argc>8 ? std::stod(argv[8]) : 600,
                           argc>9 ? std::stod(argv[9]) : 0,argc>10 ? unsigned(std::stoul(argv[10])) : 64};
    if(!std::isfinite(cpi+update+end+doppler+select.threshold_db+select.minimum_range+select.minimum_doppler) ||
       cpi<=0 || update<=0 || end<0 || doppler<=0 || doppler>400 || select.threshold_db<0 ||
       select.minimum_range<100 || select.minimum_range>29000 || select.minimum_doppler<0 || select.minimum_doppler>=doppler ||
       !select.maximum_peaks || select.maximum_peaks>4096)throw std::invalid_argument("Invalid candidate selection bounds");
    std::ifstream input(argv[1],std::ios::binary);input.exceptions(std::ios::failbit|std::ios::badbit);
    std::ofstream output(argv[2],std::ios::out|std::ios::noreplace);output.exceptions(std::ios::failbit|std::ios::badbit);
    const auto total=std::filesystem::file_size(argv[1]);char magic[8];input.read(magic,8);
    if(std::string_view(magic,8)!="DVBTLIV1")throw std::runtime_error("Invalid live archive");
    const auto header=object(input,read<std::uint32_t>(input));
    const auto* policy=header.if_contains("profile_row_policy");
    if(!policy || *policy!="fixed_grid_zero_information_erasures")throw std::runtime_error("Unsupported profile validity policy");
    auto context=std::make_shared<gpu::Context>();std::vector<std::unique_ptr<Channel>> channels;
    for(const auto& v:header.at("applied_channels").as_array()){
        const auto& c=v.as_object();channels.push_back(std::make_unique<Channel>(context,value<unsigned>(c,"guard"),value<double>(c,"rf_hz"),cpi,update));
    }
    output<<boost::json::serialize(boost::json::object{{"type","header"},{"capture",argv[1]},{"metadata",header},
        {"cpi",cpi},{"update",update},{"doppler_hz",doppler},{"threshold_db",select.threshold_db},
        {"remove_dc",remove_dc},{"stationary_training","Archive contains the frozen initial background subtraction; aperture DC projection is optional here"},
        {"minimum_range_m",select.minimum_range},{"minimum_doppler_hz",select.minimum_doppler},
        {"selection","local peak; range median exponential-noise scale; screening only"}})<<'\n';
    std::vector<C> moments;std::vector<double> information;double progress=-10;
    for(;;){
        char tag[4];input.read(tag,4);const auto length=read<std::uint32_t>(input);const auto bytes=read<std::uint64_t>(input);
        const auto m=object(input,length);const auto payload=std::uint64_t(input.tellg());
        if(bytes>64*1024*1024 || payload>total || bytes>total-payload)throw std::runtime_error("Invalid archive extent");
        if(std::string_view(tag,4)=="PROF"){
            const auto ch=value<std::size_t>(m,"channel"),count=value<std::size_t>(m,"count");
            auto& c=*channels.at(ch);const auto& rows=m.at("rows").as_array();
            if(!count || count>1020 || rows.size()!=count || m.at("size")!=1024 || m.at("moments")!=2 ||
                m.at("order")!="moment,row,range" || bytes<2*count*1024*sizeof(C))throw std::runtime_error("Invalid profiles");
            const auto time=boost::json::value_to<double>(rows.front().as_array().at(1));if(end>0 && time>end+1)break;
            std::vector<C> data(2*count*1024);input.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size()*sizeof(C)));
            for(std::size_t n=0;n<count;++n){
                const auto& row=rows[n].as_array();
                const auto t=boost::json::value_to<double>(row.at(1));
                if(!c.history.append(t,value<double>(m,"information"),row.at(5).as_bool(),value<std::uint64_t>(m,"segment"),
                    {data.data()+n*1024,1024},{data.data()+(count+n)*1024,1024}))continue;
                c.history.snapshot(moments,information);
                auto result=c.focus.calculate_device(moments,c.history.required(),c.interval,c.rf,c.history.first_time(),{30000,doppler,remove_dc},nullptr,information);
                output<<boost::json::serialize(boost::json::object{{"type","map"},{"channel",ch},{"time",result.map.phase_epoch},
                    {"start",result.map.time_start},{"end",result.map.time_end},{"rf_hz",c.rf},
                    {"valid_fraction",double(std::ranges::count_if(information,[](double x){return x>0;}))/information.size()},
                    {"peaks",peaks(result.map,static_cast<const C*>(result.values.mapped),select)}})<<'\n';
            }
            if(time>progress+30){progress=time;output.flush();std::cerr<<"Scanned "<<time<<" s\n";}
        }
        input.seekg(std::streamoff(payload+bytes));
        if(std::string_view(tag,4)=="DONE")break;
    }
    std::cerr<<"Calibration candidate scan complete\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
