#include "gpu_response.hpp"
#include "gpu_compute.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string_view>

using namespace radar;
int main(int argc,char** argv) {
    try {
        constexpr unsigned bins=6817;
        unsigned rows=1020;
        if(argc!=1){
            if(argc!=3 || std::string_view(argv[1])!="--rows")throw std::invalid_argument("Usage: sensitivity_response [--rows 1..1020]");
            const std::string_view text=argv[2];const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),rows);
            if(error!=std::errc{} || end!=text.data()+text.size() || !rows || rows>1020)throw std::invalid_argument("Invalid response-fit row count");
        }
        constexpr double tau=2*std::numbers::pi,dt=8704.0/8e6,echo_amplitude=.01;
        auto context=std::make_shared<gpu::Context>();
        auto ys=gpu::make_buffer(context->gpu,rows*bins*sizeof(C));
        auto xs=gpu::make_buffer(context->gpu,rows*bins*sizeof(C));
        auto vs=gpu::make_buffer(context->gpu,rows*bins*sizeof(std::uint32_t));
        auto* y=static_cast<C*>(context->gpu.device->mapMemory(*ys.memory,0,rows*bins*sizeof(C)));
        auto* x=static_cast<C*>(context->gpu.device->mapMemory(*xs.memory,0,rows*bins*sizeof(C)));
        auto* v=static_cast<std::uint32_t*>(context->gpu.device->mapMemory(*vs.memory,0,rows*bins*sizeof(std::uint32_t)));
        const gpu::BufferView measured{context,*ys.handle,rows*bins*sizeof(C),y};
        const ReferenceBatch reference{{context,*xs.handle,rows*bins*sizeof(C),x},{context,*vs.handle,rows*bins*sizeof(std::uint32_t),v},rows};
        std::mt19937 rng(7491);const auto points=qam_constellation(2);
        for(unsigned i=0;i<rows*bins;++i){x[i]=points[rng()%points.size()];v[i]=1;}
        GpuResponse fitter(context);
        // The baseline is known; paired subtraction isolates an echo's response
        // from the estimator's own small approximation/rounding error.
        std::array<C,bins> truth,baseline;
        for(unsigned k=0;k<bins;++k) {
            const double omega=tau*(int(k)-3408)/8192;
            truth[k]=C{.7F,.4F}*std::exp(C{.18F,.07F}*static_cast<C>(std::polar(1.0,-omega)));
        }
        for(unsigned i=0;i<rows*bins;++i)y[i]=truth[i%bins]*x[i];
        // Separate solver convergence from echo retention. Even this noiseless
        // in-model channel should not be declared optimally cancelled merely
        // because the GPU agrees with the same fixed-iteration CPU algorithm.
        for(unsigned order:{1U,8U})for(unsigned iterations:{16U,32U,64U,128U}){
            const auto fitted=fitter.fit(measured,reference,0,order,iterations);
            const auto* h=static_cast<const C*>(fitted.response.mapped);
            double error=0,amplitude_error=0,phase_error=0,energy=0;
            for(unsigned k=0;k<bins;++k){
                error+=std::norm(h[k]-truth[k]);energy+=std::norm(truth[k]);
                amplitude_error+=std::pow(double(std::abs(h[k]))-std::abs(truth[k]),2);
                phase_error+=std::pow(double(std::arg(h[k]/truth[k])),2);
            }
            std::cerr<<std::setprecision(12)<<"fit_order="<<order<<" iterations="<<iterations
                <<" relative_response_power_error="<<error/energy
                <<" relative_amplitude_square_error="<<amplitude_error/energy
                <<" phase_rms_degrees="<<std::sqrt(phase_error/bins)*180/std::numbers::pi<<'\n';
        }
        const auto base=fitter.fit(measured,reference,0,8,16);
        std::copy_n(static_cast<const C*>(base.response.mapped),bins,baseline.begin());
        double baseline_error=0,baseline_energy=0;
        for(unsigned k=0;k<bins;++k){baseline_error+=std::norm(baseline[k]-truth[k]);baseline_energy+=std::norm(truth[k]);}
        std::cerr<<"Response-fit rows: "<<rows<<"; baseline relative response power error: "<<baseline_error/baseline_energy<<'\n';
        if(!std::isfinite(baseline_error) || baseline_error/baseline_energy>.01)
            throw std::runtime_error("Baseline response error exceeds paired-probe tolerance");
        std::cout<<std::setprecision(12)<<"delay_samples,doppler_hz,refit_echo_voltage_gain,refit_echo_phase_deg,frozen_echo_voltage_gain\n";
        for(double delay:{0.,1.,4.,8.,16.,64.,128.,512.,800.})for(double frequency:{0.,.1,1.,10.,229.77941176470588}) {
            for(unsigned i=0;i<rows*bins;++i) {
                const double phase=-tau*(int(i%bins)-3408)*delay/8192+tau*frequency*(i/bins)*dt;
                y[i]=truth[i%bins]*x[i]*(C{1,0}+static_cast<C>(std::polar(echo_amplitude,phase)));
            }
            const auto fitted=fitter.fit(measured,reference,0,8,16);
            const auto* h=static_cast<const C*>(fitted.response.mapped);
            std::complex<double> recovered{};double energy=0;
            for(unsigned i=0;i<rows*bins;++i) {
                const double phase=-tau*(int(i%bins)-3408)*delay/8192+tau*frequency*(i/bins)*dt;
                const auto echo=std::complex<double>(truth[i%bins]*x[i])*std::polar(echo_amplitude,phase);
                const auto difference=echo-std::complex<double>((h[i%bins]-baseline[i%bins])*x[i]);
                recovered+=std::conj(echo)*difference;energy+=std::norm(echo);
            }
            recovered/=energy;
            if(!std::isfinite(std::abs(recovered)))throw std::runtime_error("Non-finite echo response");
            std::cout<<delay<<','<<frequency<<','<<std::abs(recovered)<<','<<std::arg(recovered)*180/std::numbers::pi<<",1\n";
        }
        context->gpu.device->unmapMemory(*ys.memory);context->gpu.device->unmapMemory(*xs.memory);context->gpu.device->unmapMemory(*vs.memory);
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
