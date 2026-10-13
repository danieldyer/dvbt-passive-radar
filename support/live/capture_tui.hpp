#pragma once
#include "capture_monitor.hpp"
#include <ftxui/component/component.hpp>
#include <deque>

namespace radar::live {
struct TimingPoint {double seconds{},drift_ppm{},timing_ns{},phase_degrees{},cfo_hz{};};
struct CaptureEvent {double seconds{};std::string message;bool warning=false;};
struct CaptureTuiView {
    CaptureStatus status;
    std::vector<std::deque<TimingPoint>> history;
    std::deque<CaptureEvent> events;
    bool stop_requested=false;
};
class CaptureUiBackend:public CaptureMonitor {
public:
    virtual CaptureTuiView view() const=0;
    virtual void request_stop()=0;
    virtual bool display_available() const=0;
    virtual void open_channel(std::size_t)=0;
};
struct CaptureUiHooks {
    std::function<void(std::size_t)> open_channel;
    std::function<void()> poll_display;
};
ftxui::Component capture_screen(CaptureUiBackend&,std::function<void()> quit);
int run_capture_tui(const CaptureWork&,CaptureUiHooks={});
}
