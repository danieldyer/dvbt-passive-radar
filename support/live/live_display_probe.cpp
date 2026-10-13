// Support-only observer of the real capture, FTXUI and native display path.
#include "native_display.hpp"
#include "capture_tui.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <fstream>
#include <mutex>

namespace {
struct ObservedMonitor final:radar::CaptureMonitor {
    radar::CaptureMonitor& target;
    std::function<void(const radar::CaptureStatus&)> observe;
    ObservedMonitor(radar::CaptureMonitor& t,std::function<void(const radar::CaptureStatus&)> o):target(t),observe(std::move(o)){}
    void publish(const radar::CaptureStatus& s) override{observe(s);target.publish(s);}
    bool stop_requested() const override{return target.stop_requested();}
};
}

int main(int argc,char** argv){
    using namespace radar;
    std::filesystem::path output;
    for(int i=1;i+1<argc;++i)if(std::string_view(argv[i])=="--output")output=argv[i+1];
    if(output.empty())return 1;
    live::NativeDisplay display;const auto factory=display.factory();
    const auto start=std::chrono::steady_clock::now();
    auto seconds=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    std::mutex mutex;boost::json::array projected,displayed,telemetry;
    LiveDisplayFactory observed=[&](auto context,const auto& description){
        auto result=factory(std::move(context),description);auto project=result.project;
        result.project=[&,project=std::move(project)](auto& map,const auto& values){
            {std::lock_guard lock(mutex);auto record=boost::json::parse(map.json()).as_object();record["wall_s"]=seconds();projected.emplace_back(std::move(record));}
            project(map,values);
        };
        return result;
    };
    double last_time=-1;std::size_t last_channel=SIZE_MAX;
    CaptureUiRunner ui=[&](const auto& work){
        CaptureWork observed_work=[&](CaptureMonitor& target){
            ObservedMonitor monitor(target,[&](const CaptureStatus& s){
                boost::json::array channels;
                for(const auto& c:s.channels)channels.emplace_back(boost::json::object{{"id",c.id},{"timing_samples",c.receiver.timing_samples},
                    {"residual_rate_ppm",c.receiver.residual_rate_ppm},{"phase_rms_degrees",c.receiver.phase_rms_degrees},{"clock_fit_ns",c.clock_fit_ns},
                    {"cfo_hz",c.receiver.frequency_hz},{"resets",c.receiver.reacquisitions},{"maps",c.maps}});
                std::lock_guard lock(mutex);telemetry.emplace_back(boost::json::object{{"wall_s",seconds()},{"processed_seconds",s.processed_seconds},
                    {"clock_ppm",s.clock_ppm},{"headroom",s.headroom},{"queue_ms",s.queue_ms},{"channels",std::move(channels)}});
            });
            return work(monitor);
        };
        return live::run_capture_tui(observed_work,{
        [&](auto channel){display.open(channel);},[&]{
            display.poll();const auto status=display.status();
            if(status.open && status.map_time>0 && (status.map_time!=last_time || status.channel!=last_channel)){
                last_time=status.map_time;last_channel=status.channel;
                std::lock_guard lock(mutex);displayed.emplace_back(boost::json::object{{"channel",status.channel},{"time_end_s",status.map_time},
                    {"wall_s",seconds()},{"integration_s",status.integration},{"equivalent_aperture_s",status.effective_integration},{"data_age_s",status.data_age}});
            }
        }});};
    const int result=capture_cli(argc,argv,observed,ui);
    if(std::filesystem::is_directory(output)){
        std::lock_guard lock(mutex);std::ofstream file(output/"display-probe.json");
        file<<boost::json::serialize(boost::json::object{{"exit_code",result},{"projected",std::move(projected)},{"displayed",std::move(displayed)},{"telemetry",std::move(telemetry)}})<<'\n';
    }
    return result;
}
