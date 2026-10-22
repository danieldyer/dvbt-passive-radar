#include "capture_diagnostics.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

using namespace radar;
namespace {void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}}
int main(){try{
    const auto path=std::filesystem::temp_directory_path()/("dvbt-diagnostics-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Remove {std::filesystem::path p;~Remove(){std::error_code ec;std::filesystem::remove(p,ec);}} remove{path};
    CaptureDiagnostics log(path,{{"test","deterministic"}});
    std::vector<ClockObservation> rows(4096);
    for(std::size_t i=0;i<rows.size();++i){auto& o=rows[i];o.kind=ClockObservation::Kind::offset;o.channel=i%5;o.sample=(1ULL<<35)+i;
        o.value=.125;o.variance=i%2 ? 1e-4 : std::numeric_limits<double>::quiet_NaN();o.innovation={.05,.0001,25,0,i%2==0,i%2 ? .25 : 1.};}
    for(unsigned i=0;i<8;++i)log.observations(rows);
    CaptureStatus s;s.phase=CapturePhase::running;s.rf={100,2,3,25,.99};s.channels.resize(1);s.channels[0].id="seven";
    log.status(s);const DeviceTimeAnchor anchor{35.25,1'700'000'000'123'456'789,55'000,501};log.time_anchor(anchor);
    log.gap({0,1ULL<<40,100,3});log.finish();log.finish();check(log.queued_bytes()==0,"Diagnostics queue did not drain");
    std::ifstream file(path);std::string line;unsigned records=0,observations=0;bool status=false,utc=false,done=false;
    while(std::getline(file,line)){const auto row=boost::json::parse(line);++records;
        if(row.at("type")=="timing_observations"){
            for(const auto& o:row.at("rows").as_array()){
                const auto& a=o.as_array();check(a[2].as_double()==double((1ULL<<35)+(observations%rows.size())),"Diagnostic epoch or FIFO order changed");
                if(observations%2==0)check(a[4].is_null(),"Unknown covariance is not explicit JSON null");++observations;}
        }
        if(row.at("type")=="status"){status=true;check(boost::json::value_to<unsigned>(row.at("rail_samples"))==2,"RF rails missing from log");}
        if(row.at("type")=="utc_anchor"){utc=true;check(row.at("host_utc_ns").as_int64()==anchor.host_utc_ns,"Lost nanosecond timestamp precision");}
        done=row.at("type")=="complete";
    }
    check(records==13 && observations==8*rows.size() && status && utc && done,"Diagnostics lost rows, provenance or completion");
    std::cout<<"Bounded asynchronous diagnostics, exact epochs, uncertainty/nulls, RF health and UTC anchors: PASS\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
