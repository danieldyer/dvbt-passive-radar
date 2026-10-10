#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <SDL.h>
#include <GLES3/gl3.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
SDL_Window* window;
SDL_GLContext context;
GLuint texture=0;
std::vector<float> powers;
std::vector<unsigned char> pixels;
int columns=0,rows=0;
double range0=0,range_step=1,freq0=0,freq_step=1,cpi=0,resolution=0,coherence=0,record_time=0,sequence=0;
double rmin=0,rmax=30000,fmin=-100,fmax=100;
float floor_db=-90,ceil_db=-10,integration=10;
bool playing=true,automatic=true,dirty=false,first=true,remove_dc=false,auto_scale=true;
const ImVec4 accent{.27f,.85f,.76f,1};
void limits(){
    rmin=std::max(0.0,rmin);rmax=std::max(rmin+1,rmax);fmax=std::max(fmin+.001,fmax);
    EM_ASM({Object.assign(window.radar,{rmin:$0,rmax:$1,fmin:$2,fmax:$3});},rmin,rmax,fmin,fmax);
}
void colour_texture(){
    if(!dirty || powers.empty())return;
    pixels.resize(powers.size()*4);
    // Continuous dark blue -> teal -> amber -> ivory. Values remain in dB;
    // changes to the display scale do not alter the scientific spectrum.
    constexpr float stops[][3]={{.018f,.032f,.07f},{.08f,.22f,.40f},{.08f,.65f,.66f},{.94f,.65f,.20f},{1,.98f,.82f}};
    for(std::size_t i=0;i<powers.size();++i){
        const float z=std::clamp((powers[i]-floor_db)/std::max(1.f,ceil_db-floor_db),0.f,1.f)*4;
        const int k=std::min(3,static_cast<int>(z));const float fraction=z-k;
        for(int c=0;c<3;++c)pixels[i*4+c]=static_cast<unsigned char>(255*(stops[k][c]*(1-fraction)+stops[k+1][c]*fraction));
        pixels[i*4+3]=255;
    }
    if(!texture)glGenTextures(1,&texture);
    glBindTexture(GL_TEXTURE_2D,texture);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,columns,rows,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    dirty=false;
}
void loop(){
    SDL_Event event;
    while(SDL_PollEvent(&event))ImGui_ImplSDL2_ProcessEvent(&event);
    int width,height;emscripten_get_canvas_element_size("#canvas",&width,&height);
    const int target_width=EM_ASM_INT({return innerWidth;}),target_height=EM_ASM_INT({return innerHeight;});
    if(width!=target_width || height!=target_height){SDL_SetWindowSize(window,target_width,target_height);emscripten_set_canvas_element_size("#canvas",target_width,target_height);}
    colour_texture();ImGui_ImplOpenGL3_NewFrame();ImGui_ImplSDL2_NewFrame();ImGui::NewFrame();
    const auto display=ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize(display);
    ImGui::Begin("Radar",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);
    ImGui::TextColored(accent,"DVB-T  /  PASSIVE RADAR");ImGui::SameLine();ImGui::TextDisabled("   RESIDUAL CROSS-SPECTRUM");
    ImGui::Separator();ImGui::Spacing();
    ImGui::BeginChild("Controls",{300,0},ImGuiChildFlags_Borders);
    const bool finished=EM_ASM_INT({return !!window.radar.meta?.finished;});
    ImGui::TextColored(accent,playing ? (finished ? "RECORDING COMPLETE" : "LIVE VIEW") : "PAUSED");
    if(ImGui::Button(playing ? "Pause display" : "Resume display",{-1,30})){playing=!playing;EM_ASM({window.radar.playing=!!$0;},playing);}
    ImGui::Spacing();ImGui::Text("Recording  %.2f s",record_time);ImGui::Text("Map  %.0f",sequence);
    ImGui::SeparatorText("Coherent integration");
    ImGui::BeginDisabled(finished);
    if(ImGui::Checkbox("Auto: longest coherent interval",&automatic))EM_ASM({window.requestIntegration($0);},automatic ? 0.0 : integration);
    if(!automatic){ImGui::SetNextItemWidth(-1);if(ImGui::InputFloat("##seconds",&integration,1,10,"%.2f s",ImGuiInputTextFlags_EnterReturnsTrue)){integration=std::clamp(integration,.1f,3600.f);EM_ASM({window.requestIntegration($0);},integration);}}
    ImGui::EndDisabled();
    ImGui::Text("Actual CPI     %.3f s",cpi);ImGui::Text("Doppler bins   %.5f Hz",resolution);
    ImGui::Text("Coherence      %.4f",coherence);
    ImGui::TextDisabled("Hann window; ENBW %.5f Hz",cpi>0 ? 1.5/cpi : 0);
    ImGui::SeparatorText("View limits");
    double lo=rmin/1000,hi=rmax/1000;
    ImGui::SetNextItemWidth(118);bool changed=ImGui::InputDouble("##rmin",&lo,0,0,"%.3f");ImGui::SameLine();ImGui::SetNextItemWidth(118);changed|=ImGui::InputDouble("##rmax",&hi,0,0,"%.3f");
    ImGui::TextDisabled("Excess path length [km]");if(changed){rmin=lo*1000;rmax=hi*1000;limits();}
    ImGui::SetNextItemWidth(118);changed=ImGui::InputDouble("##fmin",&fmin,0,0,"%.3f");ImGui::SameLine();ImGui::SetNextItemWidth(118);changed|=ImGui::InputDouble("##fmax",&fmax,0,0,"%.3f");
    ImGui::TextDisabled("Doppler [Hz]");if(changed)limits();
    if(ImGui::Button("Reset view",{-1,0})){rmin=0;rmax=EM_ASM_DOUBLE({return window.radar.meta?.full_range_max_m||30000;});fmax=EM_ASM_DOUBLE({return window.radar.meta?.full_doppler_max_hz||100;});fmin=-fmax;limits();}
    ImGui::SeparatorText("Power scale");
    const bool native_dc=EM_ASM_INT({return !!window.radar.meta?.native_dc_removed;});
    if(native_dc)remove_dc=true;
    ImGui::BeginDisabled(native_dc);
    if(ImGui::Checkbox("Remove DC / stationary returns",&remove_dc)) {
        first=true;
        EM_ASM({window.radar.remove_dc=!!$0;window.radar.refresh=true;},remove_dc);
    }
    ImGui::EndDisabled();
    if(native_dc)ImGui::TextDisabled("DC already removed in source map");
    if(ImGui::Checkbox("Auto colour scale",&auto_scale)&&auto_scale)first=true;
    ImGui::BeginDisabled(auto_scale);
    ImGui::SetNextItemWidth(-1);dirty|=ImGui::SliderFloat("##floor",&floor_db,-180,30,"Floor %.1f dB");
    ImGui::SetNextItemWidth(-1);dirty|=ImGui::SliderFloat("##ceiling",&ceil_db,-160,60,"Top %.1f dB");
    ImGui::EndDisabled();
    if(ImGui::Button("Fit colour scale",{-1,0})&&!powers.empty()){ceil_db=*std::max_element(powers.begin(),powers.end());floor_db=ceil_db-70;dirty=true;}
    ImGui::Spacing();if(ImGui::Button("Download source complex map",{-1,30}))EM_ASM({window.open('/api/native','_blank');});
    ImGui::Spacing();ImGui::TextWrapped("Wheel to zoom, drag to pan. Shift-wheel: range only. Ctrl-wheel: Doppler only. Double-click to reset.");
    ImGui::TextWrapped("Display preserves narrow peaks when pooling pixels.");
    ImGui::EndChild();ImGui::SameLine();
    ImGui::BeginChild("Map",{0,0});
    ImGui::Text("EXCESS RANGE / DOPPLER");ImGui::SameLine();ImGui::TextDisabled("   coherent residual power [relative dB]");
    const auto top=ImGui::GetCursorScreenPos();const auto available=ImGui::GetContentRegionAvail();
    const ImVec2 origin{top.x+62,top.y+14};const ImVec2 size{std::max(32.f,available.x-83),std::max(32.f,available.y-77)};
    ImGui::SetCursorScreenPos(origin);
    if(texture)ImGui::Image(static_cast<ImTextureID>(texture),size,{0,1},{1,0});
    else{ImGui::InvisibleButton("Empty map",size);ImGui::GetWindowDrawList()->AddRectFilled(origin,{origin.x+size.x,origin.y+size.y},IM_COL32(7,12,22,255));}
    const bool hovered=ImGui::IsItemHovered();
    auto* draw=ImGui::GetWindowDrawList();char label[80];
    for(int i=0;i<=5;++i){
        const float x=origin.x+size.x*i/5,y=origin.y+size.y-size.y*i/5;
        std::snprintf(label,sizeof(label),"%.2f",(range0+range_step*(columns-1)*i/5)/1000);draw->AddText({x-15,origin.y+size.y+10},IM_COL32(180,199,217,255),label);
        std::snprintf(label,sizeof(label),"%.2f",freq0+freq_step*(rows-1)*i/5);draw->AddText({origin.x-58,y-7},IM_COL32(180,199,217,255),label);
    }
    draw->AddText({origin.x+size.x/2-80,origin.y+size.y+34},IM_COL32(190,210,228,255),"Excess path length [km]");
    draw->AddText({top.x,top.y-3},IM_COL32(190,210,228,255),"Hz");
    if(hovered && columns>0){
        const auto mouse=ImGui::GetIO().MousePos;const double x=(mouse.x-origin.x)/size.x,y=1-(mouse.y-origin.y)/size.y;
        const double rr=range0+x*range_step*(columns-1),ff=freq0+y*freq_step*(rows-1);
        const int col=std::clamp(static_cast<int>(x*columns),0,columns-1),row=std::clamp(static_cast<int>(y*rows),0,rows-1);
        ImGui::SetTooltip("%.4f km   /   %+.5f Hz\n%.2f dB",rr/1000,ff,powers[row*columns+col]);
        draw->AddLine({mouse.x,origin.y},{mouse.x,origin.y+size.y},IM_COL32(225,240,240,130));
        draw->AddLine({origin.x,mouse.y},{origin.x+size.x,mouse.y},IM_COL32(225,240,240,130));
        const auto& io=ImGui::GetIO();
        if(io.MouseWheel){const double scale=std::pow(.8,io.MouseWheel);
            if(!io.KeyCtrl){rmin=rr+(rmin-rr)*scale;rmax=rr+(rmax-rr)*scale;}
            if(!io.KeyShift){fmin=ff+(fmin-ff)*scale;fmax=ff+(fmax-ff)*scale;}limits();}
        if(ImGui::IsMouseDragging(0)){const double dr=-io.MouseDelta.x/size.x*(rmax-rmin),df=io.MouseDelta.y/size.y*(fmax-fmin);rmin+=dr;rmax+=dr;fmin+=df;fmax+=df;limits();}
        if(ImGui::IsMouseDoubleClicked(0)){rmin=0;rmax=EM_ASM_DOUBLE({return window.radar.meta.full_range_max_m;});fmax=EM_ASM_DOUBLE({return window.radar.meta.full_doppler_max_hz;});fmin=-fmax;limits();}
    }
    EM_ASM({window.radar.width=Math.min(1600,Math.max(16,$0));window.radar.height=Math.min(1200,Math.max(16,$1));},size.x,size.y);
    ImGui::EndChild();ImGui::End();ImGui::Render();
    glViewport(0,0,target_width,target_height);glClearColor(.03f,.05f,.08f,1);glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());SDL_GL_SwapWindow(window);
}
}
extern "C" EMSCRIPTEN_KEEPALIVE void receive_frame(const float* data,int width,int height,double rs,double dr,double fs,double df,double duration,double bins,double quality,double time,double seq){
    columns=width;rows=height;range0=rs;range_step=dr;freq0=fs;freq_step=df;cpi=duration;resolution=bins;coherence=quality;record_time=time;sequence=seq;
    powers.assign(data,data+width*height);dirty=true;
    if((first||auto_scale)&&!powers.empty()){ceil_db=*std::max_element(powers.begin(),powers.end());floor_db=ceil_db-70;first=false;}
}
int main(){
    if(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER)!=0)return 1;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS,0);SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,0);
    window=SDL_CreateWindow("Passive radar",0,0,1280,800,SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE);
    context=SDL_GL_CreateContext(window);SDL_GL_MakeCurrent(window,context);
    IMGUI_CHECKVERSION();ImGui::CreateContext();auto& io=ImGui::GetIO();io.IniFilename=nullptr;
    io.Fonts->AddFontFromFileTTF("/font.ttf",16);
    ImGui::StyleColorsDark();auto& style=ImGui::GetStyle();style.WindowPadding={18,16};style.FramePadding={8,7};style.ItemSpacing={8,10};style.WindowRounding=0;style.ChildRounding=8;style.FrameRounding=5;
    style.Colors[ImGuiCol_WindowBg]={.035f,.055f,.085f,1};style.Colors[ImGuiCol_ChildBg]={.05f,.075f,.11f,1};
    style.Colors[ImGuiCol_Button]={.09f,.23f,.28f,1};style.Colors[ImGuiCol_ButtonHovered]={.12f,.35f,.40f,1};style.Colors[ImGuiCol_CheckMark]=accent;style.Colors[ImGuiCol_SliderGrab]=accent;
    ImGui_ImplSDL2_InitForOpenGL(window,context);ImGui_ImplOpenGL3_Init("#version 300 es");
    emscripten_set_main_loop(loop,0,true);
}
