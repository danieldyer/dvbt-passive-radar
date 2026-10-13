#include "gpu_focus.hpp"
#include "gpu_fft_plan.hpp"
#include <bit>
#include <cmath>
#include <numbers>

namespace radar {
namespace {
constexpr unsigned frequency_tile=1024;
constexpr std::uint32_t mean_code[] =
#include "focus_mean.inc"
;
constexpr std::uint32_t pack_code[] =
#include "focus_pack.inc"
;
constexpr std::uint32_t shear_code[] =
#include "focus_shear.inc"
;
constexpr std::uint32_t extract_code[] =
#include "focus_extract.inc"
;
struct Parameters {unsigned rows,size,frequencies,first,count,ranges;int half;unsigned remove_dc;float interval,rf;};
std::size_t smooth(std::size_t n){
    auto best=std::bit_ceil(n);
    for(std::size_t a=1;a<=best;a*=2)for(std::size_t b=a;b<=best;b*=3)for(std::size_t c=b;c<=best;c*=5)
        if(c>=n)best=std::min(best,c);
    return best;
}
}
struct GpuFocus::Impl {
    gpu::Kernel kernel;
    vk::UniquePipeline pack,shear,extract;
    std::unique_ptr<gpu::FftPlan> range,doppler0,doppler1,inverse;
    std::size_t rows{},size{};
    double scale_interval{};
    int scale_half=-1;
    explicit Impl(std::shared_ptr<gpu::Context> context)
        :kernel(mean_code,sizeof(Parameters),{},std::move(context)),pack(kernel.pipeline(pack_code)),
         shear(kernel.pipeline(shear_code)),extract(kernel.pipeline(extract_code)) {
        kernel.storage(5,frequency_tile*4096*sizeof(C));kernel.storage(7,2048*sizeof(C));
        inverse=std::make_unique<gpu::FftPlan>(kernel,5,4096,frequency_tile);
        std::vector<float> weights(4096*12);
        for(unsigned i=0;i<4096;++i)for(int j=0;j<12;++j){
            double w=1,x=double(i)/4095;
            for(int k=0;k<12;++k)if(k!=j)w*=(x-(k-5))/double(j-k);
            weights[i*12+j]=float(w);
        }
        kernel.upload<float>(4,weights);
    }
    ~Impl(){kernel.wait();}
    void shape(std::size_t n){
        if(rows==n)return;
        kernel.wait();range.reset();doppler0.reset();doppler1.reset();
        // Twofold internal padding is sufficient for centred DFT interpolation;
        // the evaluated Doppler grid remains fourfold. Avoid twice the FFT work
        // and scratch traffic merely to oversample the final map.
        rows=n;size=2*smooth(n);scale_half=-1;
        kernel.storage(0,2*n*1024*sizeof(C));kernel.storage(1,1024*size*sizeof(C));kernel.storage(2,1024*size*sizeof(C));
        std::vector<C> phases(size);
        for(std::size_t j=0;j<size;++j){
            const auto signed_bin=j<=size/2 ? double(j) : double(j)-size;
            phases[j]=static_cast<C>(std::polar(1.0,2*std::numbers::pi*signed_bin/size*((n-1)*.5)));
        }
        kernel.upload<C>(3,phases);
        range=std::make_unique<gpu::FftPlan>(kernel,0,1024,2*n);
        doppler0=std::make_unique<gpu::FftPlan>(kernel,1,size,1024);
        doppler1=std::make_unique<gpu::FftPlan>(kernel,2,size,1024);
    }
};
GpuFocus::GpuFocus(std::shared_ptr<gpu::Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
GpuFocus::~GpuFocus()=default;
void GpuFocus::prepare(std::size_t rows){
    if(rows<8 || rows>10000)throw std::invalid_argument("Invalid CPI FFT plan shape");
    impl_->shape(rows);
}
GpuFocus::Result GpuFocus::calculate_device(std::span<const C> values,std::size_t rows,double interval,double rf,double first_time,
                              FocusOptions options,ComputeTimings* timing,std::span<const double> information) {
    if(rows<8 || rows>10000 || values.size()!=2*rows*1024 || !std::isfinite(interval) || interval<=0 ||
       !std::isfinite(rf) || rf<100e6 || !std::isfinite(first_time) || !std::isfinite(options.doppler_hz) ||
       options.doppler_hz<=0 || options.doppler_hz>200 || options.doppler_hz>=.45/interval ||
       !std::isfinite(options.range_metres) || options.range_metres<600 || options.range_metres>30000)
        throw std::invalid_argument("Invalid constant-velocity focus coordinates (limits: 30 km, 200 Hz)");
    const double excursion=options.doppler_hz*8e6/rf*(rows-1)*interval*.5;
    if(excursion>matched_negative_context-8 || options.range_metres/(299792458.0/8e6)+excursion>matched_positive_context-8)
        throw std::invalid_argument("CPI motion exceeds the finite reference context");
    if(!information.empty() && information.size()!=rows)throw std::invalid_argument("Incomplete focusing information weights");
    std::vector<float> weights(rows,1);
    if(!information.empty()){
        double total=0;
        for(double value:information){if(!std::isfinite(value) || value<0)throw std::invalid_argument("Invalid focusing information weight");total+=value;}
        if(!std::isfinite(total) || total<=0)throw std::invalid_argument("No finite focusing information");
        for(std::size_t n=0;n<rows;++n)weights[n]=float(information[n]/total*rows);
    }
    auto& p=*impl_;p.shape(rows);p.kernel.upload<C>(0,values);p.kernel.upload<float>(9,weights);
    DopplerMap map;map.range_step=299792458.0/(4*8e6);map.ranges=std::size_t(options.range_metres/map.range_step)+1;
    map.samples=rows;map.fft_size=2*p.size;map.integration=rows*interval;map.frequency_step=1/(map.fft_size*interval);
    const int half=int(std::ceil(options.doppler_hz/map.frequency_step));
    map.frequency_start=-half*map.frequency_step;map.frequencies=2*half+1;map.dc_removed=options.remove_dc;
    map.time_start=first_time;map.time_end=first_time+(rows-1)*interval;
    map.phase_epoch=first_time+(rows-1)*interval*.5+4095.5/8e6;
    map.window=information.empty() ? "rectangular" : "inverse_noise";
    double squared_weight=0;
    for(float weight:weights)squared_weight+=double(weight)*weight;
    map.enbw=squared_weight/(double(rows)*rows*interval);map.dc_method="constant_projection";
    map.motion_model="constant_bistatic_path_rate";
    map.amplitude_units="complex_reflectivity";
    map.rf_hz=rf;
    p.kernel.storage(6,map.ranges*map.frequencies*sizeof(C));
    if(p.scale_interval!=interval || p.scale_half!=half){
        std::vector<float> scales(map.frequencies);
        for(std::size_t f=0;f<map.frequencies;++f){
            const double x=std::numbers::pi*.001024*(double(f)-half)/(2*p.size*interval);
            const double gain=std::abs(x)<.001 ? 1+x*x/6 : (2*std::sin(x)-x*std::cos(x))/x;
            scales[f]=float(4/(rows*gain));
        }
        p.kernel.upload<float>(8,scales);p.scale_interval=interval;p.scale_half=half;
    }
    Parameters push{unsigned(rows),unsigned(p.size),unsigned(map.frequencies),0,0,unsigned(map.ranges),half,
                    unsigned(options.remove_dc),float(interval),float(rf)};
    auto cmd=p.kernel.begin(timing);p.range->append(cmd,false);
    p.kernel.groups(cmd,p.kernel.primary(),push,2048);p.kernel.barrier(cmd);
    p.kernel.dispatch(cmd,*p.pack,push,128*((p.size+7)/8)*64);p.kernel.barrier(cmd);
    p.doppler0->append(cmd,false);p.doppler1->append(cmd,false);
    for(std::size_t first=0;first<map.frequencies;first+=frequency_tile){
        push.first=unsigned(first);push.count=unsigned(std::min<std::size_t>(frequency_tile,map.frequencies-first));
        p.kernel.dispatch(cmd,*p.shear,push,frequency_tile*1024);p.kernel.barrier(cmd);
        p.inverse->append(cmd,true);p.kernel.dispatch(cmd,*p.extract,push,((map.ranges+7)/8)*((push.count+7)/8)*64);p.kernel.barrier(cmd);
    }
    p.kernel.finish(cmd);
    return {std::move(map),p.kernel.view(6)};
}
DopplerMap GpuFocus::calculate(std::span<const C> values,std::size_t rows,double interval,double rf,double first_time,
                              FocusOptions options,ComputeTimings* timing,std::span<const double> information) {
    auto result=calculate_device(values,rows,interval,rf,first_time,options,timing,information);
    const auto* data=static_cast<const C*>(result.values.mapped);
    result.map.values.assign(data,data+result.map.ranges*result.map.frequencies);
    return std::move(result.map);
}
}
