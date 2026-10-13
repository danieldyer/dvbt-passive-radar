#include "live_capture.hpp"
#ifdef RADAR_TUI
#include "capture_tui.hpp"
#endif
#ifdef RADAR_NATIVE_DISPLAY
#include "native_display.hpp"
#endif
int main(int argc,char** argv){
    radar::CaptureUiRunner ui;
#ifdef RADAR_TUI
    ui=[](const auto& work){return radar::live::run_capture_tui(work);};
#endif
#ifdef RADAR_NATIVE_DISPLAY
    radar::live::NativeDisplay display;
    ui=[&](const auto& work){return radar::live::run_capture_tui(work,{[&](auto channel){display.open(channel);},[&]{display.poll();}});};
    return radar::capture_cli(argc,argv,display.factory(),ui);
#else
    return radar::capture_cli(argc,argv,{},ui);
#endif
}
