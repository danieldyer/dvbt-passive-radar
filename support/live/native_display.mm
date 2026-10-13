#include "native_display.hpp"
#include "gpu_preview.hpp"
#include <imgui.h>
#include <implot.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_metal.h>
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <AppKit/AppKit.h>
#include <mutex>
#include <optional>
#include <cmath>

namespace radar::live {
namespace {
struct Preview {
    std::uint64_t segment{};
    unsigned width{},height{};
    double time{},integration{},effective_integration{},range_end{},frequency_start{},frequency_end{},floor_db{},peak_db{};
    std::vector<float> power;
};
struct Exchange {
    std::mutex mutex;
    std::vector<std::string> names;
    std::vector<std::shared_ptr<const Preview>> frames;
    std::optional<std::size_t> selected;
    std::atomic<double> capture_time=0;
    bool active(std::size_t channel){std::lock_guard lock(mutex);return selected==channel;}
};
struct Projector {
    std::shared_ptr<Exchange> exchange;
    std::vector<std::unique_ptr<video::GpuPreview>> previews;
    Projector(std::shared_ptr<gpu::Context> context,const RecordingMetadata& description,std::shared_ptr<Exchange> state):exchange(std::move(state)){
        std::lock_guard lock(exchange->mutex);exchange->names.clear();exchange->frames.resize(description.signals.size());
        for(const auto& channel:description.signals){
            exchange->names.push_back(channel.name);previews.push_back(std::make_unique<video::GpuPreview>(context));
            previews.back()->timing.timestamp_queries=false;
        }
    }
    void project(DopplerMap& map,const gpu::BufferView& input){
        if(!exchange->active(map.channel))return;
        auto& gpu=*previews.at(map.channel);gpu.calculate(map,input,false);
        auto frame=std::make_shared<Preview>();frame->width=gpu.width;frame->height=gpu.height;
        frame->time=map.time_end;frame->segment=map.segment;frame->integration=map.integration;frame->range_end=map.ranges*map.range_step/1000;
        frame->effective_integration=map.enbw>0 ? 1/map.enbw : map.integration;
        frame->frequency_start=map.frequency_start;frame->frequency_end=map.frequency_start+map.frequencies*map.frequency_step;
        frame->power=gpu.power;
        auto values=frame->power;auto median=values.begin()+values.size()/2;std::nth_element(values.begin(),median,values.end());
        frame->floor_db=10*std::log10(std::max(*median,1e-30f));
        frame->peak_db=10*std::log10(std::max(*std::max_element(values.begin(),values.end()),1e-30f));
        std::lock_guard lock(exchange->mutex);exchange->frames[map.channel]=std::move(frame);
    }
};
}
struct NativeDisplay::Impl {
    std::shared_ptr<Exchange> exchange=std::make_shared<Exchange>();
    GLFWwindow* window=nullptr;
    bool initialized=false,backends=false,hold=false,hide_dc=false;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    CAMetalLayer* layer;
    id<MTLTexture> texture;
    std::shared_ptr<const Preview> shown;
    struct Limits {float low{},high{};bool fitted=false;};
    std::vector<Limits> limits;
    std::vector<std::uint32_t> rgba;
    std::size_t channel=0;
    bool dirty=true;
    std::uint64_t rendered_frames=0,texture_uploads=0;
    std::array<double,4> plot_pixels{};
    std::filesystem::path snapshot;
    std::chrono::steady_clock::time_point last_draw{};
    ~Impl(){close();if(initialized)glfwTerminate();}
    void close(){
        {std::lock_guard lock(exchange->mutex);exchange->selected.reset();}
        if(!window)return;
        if(backends){ImGui_ImplMetal_Shutdown();ImGui_ImplGlfw_Shutdown();backends=false;}
        if(ImPlot::GetCurrentContext())ImPlot::DestroyContext();
        if(ImGui::GetCurrentContext())ImGui::DestroyContext();
        glfwDestroyWindow(window);window=nullptr;texture=nil;layer=nil;queue=nil;device=nil;shown.reset();
    }
    void open(std::size_t selected){
        {std::lock_guard lock(exchange->mutex);
            if(selected>=exchange->names.size())return;
        }
        if(!initialized){if(!glfwInit())throw std::runtime_error("Cannot initialize native window system");initialized=true;}
        if(!window){
            glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);window=glfwCreateWindow(1120,760,"DVB-T range / Doppler",nullptr,nullptr);
            if(!window)throw std::runtime_error("Cannot create native map window");
            device=MTLCreateSystemDefaultDevice();if(!device)throw std::runtime_error("No Metal display device available");
            queue=[device newCommandQueue];layer=[CAMetalLayer layer];layer.device=device;
            layer.pixelFormat=MTLPixelFormatBGRA8Unorm;layer.framebufferOnly=YES;layer.displaySyncEnabled=YES;
            NSWindow* native=glfwGetCocoaWindow(window);native.contentView.layer=layer;native.contentView.wantsLayer=YES;
            IMGUI_CHECKVERSION();ImGui::CreateContext();ImPlot::CreateContext();
            ImGui::GetIO().IniFilename=nullptr;ImGui::GetIO().LogFilename=nullptr;
            ImGui::StyleColorsDark();ImGui_ImplGlfw_InitForOther(window,true);ImGui_ImplMetal_Init(device);backends=true;
        }
        select(selected);glfwShowWindow(window);glfwFocusWindow(window);
    }
    void select(std::size_t selected){
        std::lock_guard lock(exchange->mutex);channel=selected;exchange->selected=selected;
        limits.resize(exchange->names.size());shown.reset();hold=false;dirty=true;
        glfwSetWindowTitle(window,("DVB-T range / Doppler — "+exchange->names[channel]).c_str());
    }
    void upload(){
        if(!shown)return;
        if(!texture || texture.width!=shown->width || texture.height!=shown->height){
            auto description=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:shown->width height:shown->height mipmapped:NO];description.usage=MTLTextureUsageShaderRead;
            texture=[device newTextureWithDescriptor:description];
        }
        const auto& scale=limits[channel];rgba.resize(shown->power.size());
        for(std::size_t i=0;i<rgba.size();++i){
            const double frequency=shown->frequency_end-(double(i/shown->width)+.5)*(shown->frequency_end-shown->frequency_start)/shown->height;
            if(hide_dc && std::abs(frequency)<3/shown->integration){rgba[i]=0xff000000U;continue;}
            const float db=10*std::log10(std::max(shown->power[i],1e-30f));
            const auto colour=ImPlot::SampleColormap(std::clamp((db-scale.low)/(scale.high-scale.low),0.f,1.f),ImPlotColormap_Viridis);
            rgba[i]=ImGui::ColorConvertFloat4ToU32(colour);
        }
        [texture replaceRegion:MTLRegionMake2D(0,0,shown->width,shown->height) mipmapLevel:0
            withBytes:rgba.data() bytesPerRow:shown->width*sizeof(std::uint32_t)];dirty=false;++texture_uploads;
    }
    void poll(){@autoreleasepool {
        if(!window)return;
        glfwPollEvents();
        if(glfwWindowShouldClose(window)){close();return;}
        const auto now=std::chrono::steady_clock::now();if(now-last_draw<std::chrono::milliseconds(16))return;last_draw=now;
        std::vector<std::string> names;std::shared_ptr<const Preview> latest;
        {std::lock_guard lock(exchange->mutex);names=exchange->names;latest=exchange->frames[channel];}
        if(!hold && latest && latest!=shown){shown=std::move(latest);auto& scale=limits[channel];
            if(!scale.fitted){scale.low=float(shown->floor_db-1.5);scale.high=std::max(scale.low+8,float(shown->peak_db+.25));scale.fitted=true;}
            // Keep the broad background fixed; extend the upper limit only if
            // a new maximum would saturate. User changes remain channel-local.
            scale.high=std::max(scale.high,float(shown->peak_db+.25));dirty=true;
        }
        int width=0,height=0;glfwGetFramebufferSize(window,&width,&height);if(!width || !height)return;
        layer.framebufferOnly=snapshot.empty();
        layer.drawableSize=CGSizeMake(width,height);id<CAMetalDrawable> drawable=[layer nextDrawable];if(!drawable)return;
        auto pass=[MTLRenderPassDescriptor renderPassDescriptor];pass.colorAttachments[0].texture=drawable.texture;
        pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
        pass.colorAttachments[0].clearColor=MTLClearColorMake(.03,.04,.06,1);
        ImGui_ImplMetal_NewFrame(pass);ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0,0));ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Range / Doppler",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetNextItemWidth(180);
        if(ImGui::BeginCombo("Channel",names[channel].c_str())){
            for(std::size_t c=0;c<names.size();++c){if(ImGui::Selectable(names[c].c_str(),channel==c)){select(c);break;}}
            ImGui::EndCombo();
        }
        ImGui::SameLine();ImGui::Checkbox("Hold display",&hold);ImGui::SameLine();
        dirty|=ImGui::Checkbox("Hide DC band",&hide_dc);ImGui::SameLine();
        if(ImGui::Button("Fit colour scale") && shown){auto& s=limits[channel];s.low=float(shown->floor_db-1.5);s.high=std::max(s.low+8,float(shown->peak_db+.25));dirty=true;}
        if(shown){auto& s=limits[channel];ImGui::SetNextItemWidth(130);dirty|=ImGui::DragFloat("Floor dB",&s.low,.1f);
            ImGui::SameLine();ImGui::SetNextItemWidth(130);dirty|=ImGui::DragFloat("Peak dB",&s.high,.1f);
            s.high=std::max(s.high,s.low+.1f);if(dirty)upload();
            ImGui::Text("t %.3f s   CPI %.3f s   Equivalent aperture %.3f s   Data age %.2f s",
                shown->time,shown->integration,shown->effective_integration,std::max(0.0,exchange->capture_time.load()-shown->time));
            ImGui::TextUnformatted("Physical complex-map power; colour scale only. No range/Doppler normalization.");
            if(ImPlot::BeginPlot("##map",ImVec2(-85,-28),ImPlotFlags_NoLegend)){
                ImPlot::SetupAxes("Excess path (km)","Doppler (Hz)",ImPlotAxisFlags_NoGridLines,ImPlotAxisFlags_NoGridLines);
                ImPlot::SetupAxesLimits(0,shown->range_end,shown->frequency_start,shown->frequency_end,ImGuiCond_Once);
                const auto origin=ImPlot::GetPlotPos(),size=ImPlot::GetPlotSize(),scale=ImGui::GetIO().DisplayFramebufferScale;
                plot_pixels={origin.x*scale.x,origin.y*scale.y,size.x*scale.x,size.y*scale.y};
                ImPlot::PlotImage("Power",(ImTextureID)(__bridge void*)texture,ImPlotPoint(0,shown->frequency_start),
                    ImPlotPoint(shown->range_end,shown->frequency_end));
                ImPlot::EndPlot();
            }
            ImGui::SameLine();ImPlot::ColormapScale("dB",s.low,s.high,ImVec2(60,-28),"%.1f",0,ImPlotColormap_Viridis);
        }else ImGui::TextWrapped("Waiting for an uninterrupted coherent interval. Capture continues in the terminal; close this window to hide it.");
        ImGui::TextUnformatted("Drag to pan; wheel to zoom; double-click to fit. Closing this window keeps acquisition running.");
        ImGui::End();ImGui::Render();
        auto buffer=[queue commandBuffer];auto encoder=[buffer renderCommandEncoderWithDescriptor:pass];
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(),buffer,encoder);[encoder endEncoding];
        id<MTLBuffer> readback;
        if(!snapshot.empty()){
            readback=[device newBufferWithLength:std::size_t(width)*height*4 options:MTLResourceStorageModeShared];
            auto copy=[buffer blitCommandEncoder];
            [copy copyFromTexture:drawable.texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                sourceSize:MTLSizeMake(width,height,1) toBuffer:readback destinationOffset:0
                destinationBytesPerRow:width*4 destinationBytesPerImage:std::size_t(width)*height*4];[copy endEncoding];
        }
        [buffer presentDrawable:drawable];[buffer commit];++rendered_frames;
        if(readback){
            [buffer waitUntilCompleted];if(buffer.status==MTLCommandBufferStatusError)throw std::runtime_error("Native display snapshot GPU copy failed");
            auto* bytes=static_cast<unsigned char*>(readback.contents);for(std::size_t i=0;i<std::size_t(width)*height*4;i+=4)std::swap(bytes[i],bytes[i+2]);
            auto image=[[NSBitmapImageRep alloc] initWithBitmapDataPlanes:&bytes pixelsWide:width pixelsHigh:height bitsPerSample:8
                samplesPerPixel:4 hasAlpha:YES isPlanar:NO colorSpaceName:NSDeviceRGBColorSpace bytesPerRow:width*4 bitsPerPixel:32];
            auto png=[image representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            if(![png writeToFile:[NSString stringWithUTF8String:snapshot.c_str()] atomically:YES])throw std::runtime_error("Cannot save native display snapshot");
            snapshot.clear();
        }
    }}
};
NativeDisplay::NativeDisplay():impl_(std::make_shared<Impl>()){}
NativeDisplay::~NativeDisplay()=default;
LiveDisplayFactory NativeDisplay::factory() const{
    return [exchange=impl_->exchange](auto context,const auto& description){
        auto project=std::make_shared<Projector>(std::move(context),description,exchange);
        return LiveDisplay{[project](auto& map,const auto& input){project->project(map,input);},
            [](std::size_t,DopplerMap&&){},[exchange](double time){exchange->capture_time=time;},
            [exchange](std::size_t channel){return exchange->active(channel);},project};
    };
}
void NativeDisplay::open(std::size_t channel){@autoreleasepool {
    try{impl_->open(channel);}catch(...){impl_->close();throw;}
}}
void NativeDisplay::poll(){impl_->poll();}
void NativeDisplay::close(){@autoreleasepool {impl_->close();}}
NativeDisplay::Status NativeDisplay::status() const{
    Status result{bool(impl_->window),impl_->channel,impl_->rendered_frames,impl_->texture_uploads,impl_->shown ? impl_->shown->time : 0,impl_->plot_pixels};
    if(impl_->shown){result.integration=impl_->shown->integration;result.effective_integration=impl_->shown->effective_integration;
        result.data_age=std::max(0.0,impl_->exchange->capture_time.load()-impl_->shown->time);}
    return result;
}
void NativeDisplay::save_snapshot(const std::filesystem::path& path){
    if(!impl_->window || path.empty())throw std::invalid_argument("Snapshot requires an open native map and an output path");
    impl_->snapshot=path;
}
}
