#include "capture_tui.hpp"
#include <ftxui/component/event.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/terminal.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace radar::live {
namespace {
using namespace ftxui;
const auto accent=Color::RGB(88,196,211),good=Color::RGB(146,202,153),
    warning=Color::RGB(230,188,94),bad=Color::RGB(242,132,130),muted=Color::RGB(128,140,155);
std::string fixed(double value,int precision=2){
    if(!std::isfinite(value))return "--";
    std::ostringstream text;text<<std::fixed<<std::setprecision(precision)<<value;return text.str();
}
std::string tracked(bool valid,double value,int precision=3){return valid ? fixed(value,precision) : "--";}
std::string count(std::uint64_t value){return std::to_string(value);}
class CaptureScreen final:public ComponentBase {
    CaptureUiBackend& backend_;
    std::function<void()> quit_;
    std::vector<std::string> pages_{"Overview","Timing","Events","Source","Help"},names_{"Waiting"};
    int page_=0,last_page_=-1,channel_=0,event_offset_=0,scroll_=0;
    bool hold_=false,colour_=std::getenv("NO_COLOR")==nullptr;
    Component tabs_,channels_,hold_button_,map_button_,stop_button_,quit_button_;
    Box box_;
    std::vector<std::deque<TimingPoint>> held_history_;
    Element tint(Element element,Color value) const{return colour_ ? element|color(value) : element;}
    Element value(std::string label,std::string reading,Color ink=Color::Default) const{
        return hbox({tint(text(std::move(label)),muted),filler(),tint(text(std::move(reading)),ink)});
    }
    Element panel(std::string title,Elements rows) const{
        return window(tint(text(" "+title+" "),accent)|bold,vbox(std::move(rows)));
    }
    Color health(const CaptureStatus& s) const{
        if(s.phase==CapturePhase::failed || s.missing || s.host_discarded || s.unknown_gaps)return bad;
        return s.phase==CapturePhase::running ? good : warning;
    }
    Element cell(std::string content,int width,Color ink=Color::Default) const{
        return tint(text(std::move(content)),ink)|size(WIDTH,EQUAL,width);
    }
    Element channel_table(const CaptureStatus& s,bool wide) const{
        Elements rows;
        Elements heading{cell("CHANNEL",12),cell("LOCK",8),cell("CFO Hz",11),cell("RAW RATE ppm",12),cell("OFFSET RMS",10),cell("RS %",8)};
        if(wide){heading.push_back(cell("PILOT dB",10));heading.push_back(cell("ROWS",12));heading.push_back(cell("RESETS",8));}
        rows.push_back(hbox(std::move(heading))|bold);rows.push_back(separator());
        for(std::size_t i=0;i<s.channels.size();++i){const auto& c=s.channels[i];const auto& r=c.receiver;
            const auto lock=r.ready ? "TRACK" : r.locked ? "ANCHOR" : "SEARCH";
            Elements cells{cell(c.name,12),cell(lock,8,r.ready ? good : warning),
                cell(tracked(r.locked,r.frequency_hz),11),cell(tracked(r.ready,r.residual_rate_ppm,4),12),
                cell(tracked(r.ready,r.timing_rms_samples*125),10),
                cell(c.rs_good+c.rs_bad ? fixed(100.0*c.rs_good/(c.rs_good+c.rs_bad)) : "--",8,c.rs_bad ? bad : good)};
            if(wide){cells.push_back(cell(tracked(r.ready,r.pilot_snr_db,1),10));cells.push_back(cell(count(c.rows),12));cells.push_back(cell(count(r.reacquisitions),8));}
            auto row=hbox(std::move(cells));
            if(int(i)==channel_){row=row|bold;row=colour_ ? row|bgcolor(Color::RGB(29,55,66)) : row|inverted;}
            rows.push_back(std::move(row));
        }
        if(s.channels.empty())rows.push_back(text("Waiting for capture configuration"));
        return panel("Channel recovery",std::move(rows));
    }
    Element overview(const CaptureTuiView& v,int width,int height) const{
        const auto& s=v.status;const bool compact=height<30;
        Elements capture{value("Received samples",count(s.received)),value("Missing / discarded",count(s.missing)+" / "+count(s.host_discarded),s.missing || s.host_discarded ? bad : good),
            value("Unquantified / overflows",count(s.unknown_gaps)+" / "+count(s.overflows),s.unknown_gaps || s.overflows ? bad : good),
            value("Sequence errors / timeouts",count(s.sequence_errors)+" / "+count(s.timeouts))};
        Elements processing{value("DSP headroom",s.processed_seconds ? fixed(s.headroom)+"x; recent "+fixed(s.recent_headroom)+"x" : "warming"),
            value("Ring backlog / capacity",fixed(s.queue_ms,1)+" / "+fixed(s.capacity_ms,1)+" ms",s.queue_ms>.8*s.capacity_ms && s.capacity_ms ? warning : good),
            value("Shared ADC clock",fixed(s.clock_ppm,4)+" ppm"),
            value("Phase budget / segments",fixed(s.phase_budget,1)+" deg / "+count(s.clock_segments))};
        if(!compact){capture.push_back(value("Acquisition / replay loop",count(s.segments)+" / "+count(s.replay_loops+1)));
            processing.push_back(value("Archive written / queued",fixed(s.archive_bytes/1e6,1)+" / "+fixed(s.archive_queued_bytes/1e6,1)+" MB"));}
        Elements content{hbox({panel("Capture integrity",std::move(capture))|flex,panel("Processing",std::move(processing))|flex}),
            channel_table(s,width>=110)};
        if(!compact)content.push_back(panel("Work and output",{
            value("CPU FEC workers, summed",fixed(s.fec_ms_per_second,1)+" ms/input-s"),
            value("GPU timestamps",s.profile ? fixed(s.gpu_ms_per_second,1)+" ms/input-s" : "off; enable with --profile"),
            value("Scientific archive",s.archive=="none" ? "disabled" : s.archive+"; complex data"),
            value("Live map",s.display.empty() ? "disabled in this build" : "ImPlot; selected channel [v]")}));
        if(!compact)content.push_back(panel("RF health (processed ADC samples)",{
            value("Complex RMS / component peak",s.rf.samples ? fixed(s.rf.rms_dbfs(),1)+" / "+fixed(s.rf.peak_dbfs(),1)+" dBFS" : "--"),
            value("Digital rail / near rail",count(s.rf.rail_samples)+" / "+count(s.rf.near_rail_samples),s.rf.near_rail_samples ? warning : good)}));
        return vbox(std::move(content));
    }
    Element trace(const std::deque<TimingPoint>& history,double TimingPoint::* member,std::string name,std::string unit,double floor,int rows) const{
        std::vector<double> points;points.reserve(history.size());
        double extent=floor;for(const auto& p:history){const auto x=p.*member;points.push_back(x);extent=std::max(extent,std::abs(x));}
        auto line=canvas([points=std::move(points),extent](Canvas& c){
            if(points.empty())return;
            auto y=[&](int x){const auto i=points.size()==1 ? 0 : std::size_t(double(x)*(points.size()-1)/std::max(c.width()-1,1));
                return std::clamp(int(std::lround((.5-.45*points[i]/extent)*(c.height()-1))),0,c.height()-1);};
            for(int x=1;x<c.width();++x)c.DrawPointLine(x-1,y(x-1),x,y(x));
        })|size(HEIGHT,EQUAL,rows);
        return panel(std::move(name),{tint(std::move(line),accent),tint(text("+/- "+fixed(extent,3)+" "+unit+"; "+count(history.size())+" updates"),muted)});
    }
    Element timing(const CaptureTuiView& v,int width,int height) const{
        const auto& s=v.status;
        if(s.channels.empty())return text("Waiting for channel recovery");
        const auto& c=s.channels.at(channel_);const auto& r=c.receiver;const auto& o=c.clock;
        const auto& h=held_history_.at(channel_);const int plot_rows=height>=34 ? 4 : 3;
        auto pair=[&](Element a,Element b){return width>=110 ? hbox({std::move(a)|flex,std::move(b)|flex}) : vbox({std::move(a),std::move(b)});};
        Elements metrics{
            pair(panel("Pilot timing and phase",{
                value("CFO",tracked(r.locked,r.frequency_hz)+" Hz"),
                value("Raw pilot rate",tracked(r.ready,r.residual_rate_ppm,5)+" ppm"),
                value("Timing offset / RMS",tracked(r.ready,r.timing_samples*125)+" / "+tracked(r.ready,r.timing_rms_samples*125)+" ns"),
                value("Pilot timing sigma",tracked(r.ready && std::isfinite(r.timing_sigma_samples),r.timing_sigma_samples*125,3)+" ns (conditional)"),
                value("Phase RMS",tracked(r.ready,r.phase_rms_degrees)+" deg",r.phase_rms_degrees>s.phase_budget ? bad : good),
                value("Pilot quality / SNR",tracked(r.ready,r.pilot_quality)+" / "+tracked(r.ready,r.pilot_snr_db,1)+" dB")}),
                panel("Shared-clock observer",{
                    value("Applied common rate",fixed(s.clock_ppm,5)+" ppm"),
                    value("Acquisition rate estimate",tracked(o.anchored,o.estimated_ppm,5)+" ppm"),
                    value("Innovation RMS / NIS",tracked(o.anchored,c.clock_fit_ns)+" ns / "+tracked(o.anchored,o.normalized_innovation)),
                    value("Predicted clock sigma",fixed(s.clock_sigma_ns,3)+" ns (model conditional)"),
                    value("Bounded / late observations",count(o.downweighted)+" / "+count(o.late),o.downweighted || o.late ? warning : good),
                    value("Observations / span",count(o.observations)+" / "+fixed(o.span_seconds,2)+" s"),
                    value("Recent timing update",o.active ? "YES" : "NO",o.active ? good : warning),
                    value("Lock segment / resets",count(r.segment)+" / "+count(r.reacquisitions))})),
            pair(trace(h,&TimingPoint::drift_ppm,"Raw differential pilot rate","ppm",.001,plot_rows),
                trace(h,&TimingPoint::timing_ns,"Pilot timing offset","ns",.01,plot_rows)),
            pair(trace(h,&TimingPoint::phase_degrees,"Pilot phase RMS","deg",.01,plot_rows),
                trace(h,&TimingPoint::cfo_hz,"Receiver frequency correction","Hz",.01,plot_rows)),
            panel("Reconstruction and FEC",{
                value("RF / NCO",fixed(c.rf_hz/1e6,6)+" / "+fixed(c.nco_hz/1e6,6)+" MHz"),
                value("Modulation / code rate",c.qam ? std::to_string(c.qam)+"-QAM / "+c.code_rate : "TPS pending"),
                value("RS good / bad / corrected bytes",count(c.rs_good)+" / "+count(c.rs_bad)+" / "+count(c.corrected_bytes),c.rs_bad ? bad : good),
                value("Rows / fully valid rows",count(c.rows)+" / "+count(c.valid_rows)),
                value("Response relative fit error",fixed(c.response_error,6)),
                value("Maps / last sample time",count(c.maps)+" / "+fixed(c.map_time,3)+" s")}),
            tint(text((hold_ ? "PLOTS HELD. " : "")+std::string("Timing traces restart on reacquisition; offsets are not absolute path delay.")),muted)};
        return vbox(std::move(metrics));
    }
    Element events(const CaptureTuiView& v,int height){
        Elements lines{tint(text("Session events (newest first)   PgUp/PgDn browse   128 retained"),muted),separator()};
        const auto end=int(v.events.size());event_offset_=std::clamp(event_offset_,0,std::max(0,end-1));
        for(int i=end-1-event_offset_;i>=0 && int(lines.size())<std::max(5,height-12);--i){
            const auto& e=v.events[i];lines.push_back(hbox({cell(fixed(e.seconds,2)+" s",11,muted),tint(text(e.message),e.warning ? bad : Color::Default)}));
        }
        if(v.events.empty())lines.push_back(text("Waiting for capture"));
        return panel("Event log",std::move(lines));
    }
    Element source(const CaptureStatus& s) const{
        const auto& a=s.settings;
        return vbox({panel("Applied SDR settings",{
            value("Rate / centre",s.applied ? fixed(a.rate_hz/1e6,9)+" MS/s / "+fixed(a.centre_hz/1e6,6)+" MHz" : "configuration pending"),
            value("Gain / bandwidth",s.applied ? fixed(a.gain_db)+" dB / "+(a.driver=="usdr" && a.bandwidth_hz==0 ? "LPF bypass" : fixed(a.bandwidth_hz/1e6,3)+" MHz") : "--"),
            value("Antenna / serial",a.antenna+" / "+a.serial),
            value("Clock / time reference",a.clock_source+" / "+a.time_source),
            value("Master clock",s.applied ? fixed(a.master_clock_hz/1e6,9)+" MHz" : "--")}),
            panel("Processing configuration",{
                value("CPI / update",fixed(s.integration_seconds,2)+" / "+fixed(s.update_seconds,2)+" s"),
                value("Map extent","0-"+fixed(s.range_metres/1000,0)+" km / +/- "+fixed(s.doppler_hz,0)+" Hz"),
                value("Stationary suppression",s.remove_dc ? "display only; archive preserved" : "disabled"),
                value("Archive / simulation",s.archive+" / "+(s.looping ? "loops until stopped" : s.simulation ? "single pass" : "SDR continuous"))}),
            panel("Capture health",{
                value("Minimum recent headroom",fixed(s.minimum_recent_headroom)+"x"),
                value("Longest processing block",fixed(s.maximum_block_ms,1)+" ms"),
                value("Peak ring backlog",fixed(s.peak_queue_ms,1)+" ms"),
                value("Map admissions skipped for backlog",count(s.display_pressure_skips)),
                value("RF complex RMS / component peak",s.rf.samples ? fixed(s.rf.rms_dbfs(),1)+" / "+fixed(s.rf.peak_dbfs(),1)+" dBFS" : "--"),
                value("Digital rail / >=0.98 FS",count(s.rf.rail_samples)+" / "+count(s.rf.near_rail_samples)),
                value("Diagnostic history","diagnostics.jsonl; continuous")}),
            panel("Paths",{tint(text("Metadata"),muted),paragraph(s.metadata),tint(text("Source"),muted),paragraph(s.source),
                tint(text("Output"),muted),paragraph(s.output)}),
            tint(text("RF and processing configuration comes from metadata/CLI; stop before starting a new configuration."),muted)});
    }
    Element help() const{
        return panel("Keyboard guide",{
            value("o / t / e / i","Overview / Timing / Events / Source"),
            value("? or F1 / Esc","Help / return to overview"),
            value("1-9 / Up / Down","Select a channel"),
            value("[ / ]","Previous / next page"),
            value("Tab / Shift-Tab","Focus controls; Enter activates"),
            value("p","Hold/resume timing plots; capture continues"),
            value("v / Map button","Native ImPlot for selected channel; capture continues"),
            value("s / Stop button","Stop acquisition and drain; keep results open"),
            value("q / Ctrl-C / Quit button","Stop, drain and exit"),
            value("PgUp / PgDn","Scroll page or browse the event log"),
            separator(),
            paragraph("Mouse selection and terminal resizing are supported. Capture and GPU work run on a separate thread, including startup warm-up. Red indicates losses or faults; no lost samples are inferred from a timeout alone."),
            separator(),
            paragraph("Simulation loops by default. A file seam creates a new acquisition segment and clears coherent history; it is not counted as a sample drop. --simulate-once restores a finite replay. Use --archive none for an indefinite demonstration.")});
    }
public:
    CaptureScreen(CaptureUiBackend& backend,std::function<void()> quit):backend_(backend),quit_(std::move(quit)){
        tabs_=Menu(&pages_,&page_,MenuOption::Horizontal());channels_=Menu(&names_,&channel_,MenuOption::Horizontal());
        hold_button_=Checkbox("Hold plots [p]",&hold_);
        map_button_=Button("Map [v]",[this]{backend_.open_channel(channel_);},ButtonOption::Simple());
        stop_button_=Button("Stop [s]",[this]{backend_.request_stop();},ButtonOption::Simple());
        quit_button_=Button("Quit [q]",[this]{quit_();},ButtonOption::Simple());
        Components controls{hold_button_};if(backend_.display_available())controls.push_back(map_button_);
        controls.push_back(stop_button_);controls.push_back(quit_button_);
        Add(Container::Vertical({tabs_,channels_,Container::Horizontal(std::move(controls))}));
    }
    Element OnRender() override{
        if(page_!=last_page_){scroll_=0;last_page_=page_;}
        const auto v=backend_.view();const auto& s=v.status;
        names_.clear();for(const auto& c:s.channels)names_.push_back(c.name);if(names_.empty())names_.push_back("Waiting");
        channel_=std::clamp(channel_,0,int(names_.size())-1);
        if(!hold_ || held_history_.size()!=v.history.size())held_history_=v.history;
        const auto terminal=Terminal::Size();const int width=box_.x_max>box_.x_min ? box_.x_max-box_.x_min+1 : terminal.dimx;
        const int height=box_.y_max>box_.y_min ? box_.y_max-box_.y_min+1 : terminal.dimy;
        Element content;
        switch(page_){
        case 0:content=overview(v,width,height);break;
        case 1:content=timing(v,width,height);break;
        case 2:content=events(v,height);break;
        case 3:content=source(s);break;
        default:content=help();break;
        }
        const auto mode=s.simulation ? (s.looping ? "SIMULATION / loop "+count(s.replay_loops+1) : "SIMULATION / single pass") : s.settings.driver;
        auto result=vbox({
            hbox({tint(text(" DVB-T PASSIVE RADAR "),accent)|bold,filler(),tint(text(std::string(capture_phase_name(s.phase))+" "),health(s))|bold}),
            hbox({text(" "+mode),filler(),tint(text(fixed(s.elapsed,1)+" s elapsed / "+fixed(s.processed_seconds,1)+" s processed "),muted)}),
            tabs_->Render(),channels_->Render(),separator(),
            std::move(content)|focusPosition(0,scroll_)|yframe|vscroll_indicator|flex,
            separator(),
            hbox({hold_button_->Render(),filler(),backend_.display_available() ? map_button_->Render() : text(""),text(" "),stop_button_->Render(),text(" "),quit_button_->Render()}),
            tint(text(v.stop_requested && s.phase!=CapturePhase::stopped && s.phase!=CapturePhase::failed ? " Stop requested; completing outstanding work..." : " "+s.message),health(s))
        })|reflect(box_);
        return colour_ ? result|bgcolor(Color::RGB(15,19,25))|color(Color::RGB(220,228,239)) : result;
    }
    bool OnEvent(Event event) override{
        if(event==Event::Character('q') || event==Event::CtrlC){quit_();return true;}
        if(event==Event::Character('s')){backend_.request_stop();return true;}
        if(event==Event::Character('v')){backend_.open_channel(channel_);return true;}
        if(event==Event::Character('p')){hold_=!hold_;return true;}
        if(event==Event::Character('?') || event==Event::F1){page_=4;scroll_=0;return true;}
        if(event==Event::Character('o') || event==Event::Escape){page_=0;scroll_=0;return true;}
        if(event==Event::Character('t')){page_=1;scroll_=0;return true;}
        if(event==Event::Character('e')){page_=2;scroll_=0;return true;}
        if(event==Event::Character('i')){page_=3;scroll_=0;return true;}
        if(event==Event::Character('[')){page_=(page_+4)%5;scroll_=0;return true;}
        if(event==Event::Character(']')){page_=(page_+1)%5;scroll_=0;return true;}
        if(event==Event::ArrowUp || event==Event::ArrowDown){channel_=std::clamp(channel_+(event==Event::ArrowUp ? -1 : 1),0,int(names_.size())-1);return true;}
        if(event==Event::PageUp){if(page_==2)event_offset_+=10;else scroll_=std::max(0,scroll_-10);return true;}
        if(event==Event::PageDown){if(page_==2)event_offset_=std::max(0,event_offset_-10);else scroll_+=10;return true;}
        if(event.is_character() && event.character().size()==1){const char c=event.character()[0];
            if(c>='1' && c<='9' && std::size_t(c-'1')<names_.size()){channel_=c-'1';return true;}}
        return ComponentBase::OnEvent(std::move(event));
    }
};
}
ftxui::Component capture_screen(CaptureUiBackend& backend,std::function<void()> quit){return std::make_shared<CaptureScreen>(backend,std::move(quit));}
}
