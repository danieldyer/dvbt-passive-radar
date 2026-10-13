#include "capture_tui.hpp"
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <fstream>
#include <iostream>

using namespace radar;
using namespace radar::live;
using namespace ftxui;
namespace {
void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
struct Backend final:CaptureUiBackend {
    CaptureTuiView state;
    std::vector<std::size_t> opened;
    void publish(const CaptureStatus& s) override{state.status=s;}
    bool stop_requested() const override{return state.stop_requested;}
    void request_stop() override{state.stop_requested=true;}
    bool display_available() const override{return true;}
    void open_channel(std::size_t c) override{opened.push_back(c);}
    CaptureTuiView view() const override{return state;}
};
std::string render(Component component,int width,int height,const std::filesystem::path& output={}){
    auto screen=Screen::Create(Dimension::Fixed(width),Dimension::Fixed(height));
    // Reflect terminal layout once before rendering assertions at the new size.
    Render(screen,component->Render());screen.Clear();Render(screen,component->Render());
    if(!output.empty())std::ofstream(output)<<screen.ToString();
    screen.stencil.x_min=0;screen.stencil.x_max=width-1;screen.stencil.y_min=0;screen.stencil.y_max=height-1;
    std::string plain;
    for(int y=0;y<height;++y){for(int x=0;x<width;++x){const auto& c=screen.CellAt(x,y).character;plain+=c.empty() ? " " : c;}plain+='\n';}
    return plain;
}
}
int main(int argc,char** argv){try{
    Backend backend;auto& s=backend.state.status;s.phase=CapturePhase::running;
    s.simulation=true;s.looping=true;s.applied=true;s.elapsed=42;s.processed_seconds=40;
    s.message="SIMULATED VALIDATION DATA";s.headroom=3.25;s.queue_ms=12;s.capacity_ms=524;
    s.phase_budget=3;s.archive="profiles";s.display="Native ImPlot";
    s.settings.rate_hz=61'440'000.048;s.settings.centre_hz=202e6;s.settings.serial="SIM";
    const std::array names{"Seven","SBS","Nine","Ten","ABC"};
    for(std::size_t i=0;i<names.size();++i){CaptureChannelStatus c;c.name=names[i];c.id=c.name;
        c.receiver.locked=c.receiver.ready=true;c.receiver.frequency_hz=12.125;
        c.receiver.residual_rate_ppm=.0042;c.receiver.timing_samples=.015;c.receiver.timing_rms_samples=.001;
        c.receiver.phase_rms_degrees=.25;c.receiver.pilot_snr_db=32;c.rs_good=42'000;c.rows=9000;
        c.clock.anchored=c.clock.active=true;c.clock.observations=25;c.clock.span_seconds=4;
        c.qam=64;c.code_rate="3/4";s.channels.push_back(c);
        auto& h=backend.state.history.emplace_back();for(int n=0;n<100;++n)h.push_back({double(n),.001*n,.002*n,.25,12.125});
    }
    bool quit=false;auto component=capture_screen(backend,[&]{quit=true;});
    std::filesystem::path output;if(argc>1){output=argv[1];std::filesystem::create_directories(output);}
    auto save=[&](const char* file){return output.empty() ? output : output/file;};
    auto text=render(component,120,36,save("overview.ansi"));
    check(text.find("Capture integrity")!=text.npos && text.find("Seven")!=text.npos && text.find("ABC")!=text.npos,"Overview missing capture/channel state");
    component->OnEvent(Event::Character('5'));component->OnEvent(Event::Character('v'));
    check(backend.opened==std::vector<std::size_t>{4} && !backend.stop_requested(),"Map key selected wrong channel or stopped capture");
    component->OnEvent(Event::Character('t'));text=render(component,120,44,save("timing.ansi"));
    check(text.find("Pilot timing and phase")!=text.npos && text.find("Shared-clock observer")!=text.npos && text.find("Reconstruction and FEC")!=text.npos,"Timing/FEC page missing telemetry");
    component->OnEvent(Event::Character('p'));check(render(component,120,44).find("PLOTS HELD")!=std::string::npos,"Timing hold key failed");
    backend.state.events.push_back({42,"Injected loss: 119999 samples",true});component->OnEvent(Event::Character('e'));
    check(render(component,80,24).find("119999")!=std::string::npos,"Loss event absent at small terminal size");
    component->OnEvent(Event::Character('?'));text=render(component,100,38,save("help.ansi"));
    check(text.find("Native ImPlot")!=text.npos && text.find("loops by default")!=text.npos,"Help missing native display or looping controls");
    component->OnEvent(Event::Escape);text=render(component,80,24,save("compact.ansi"));
    check(text.find("Capture integrity")!=text.npos && text.find("ABC")!=text.npos && text.find("Quit")!=text.npos,"Compact layout hid channels or controls");
    component->OnEvent(Event::Character('s'));check(backend.stop_requested() && !quit,"Stop should drain and retain the TUI");
    s.phase=CapturePhase::stopped;s.message="Capture complete";
    check(render(component,80,24).find("STOPPED")!=std::string::npos,"Stopped results not retained");
    component->OnEvent(Event::CtrlC);check(quit,"Ctrl-C did not request exit");
    std::cout<<"Interactive channel selection, native map dispatch, timing/FEC pages, hold, resizing, stop and quit: PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
