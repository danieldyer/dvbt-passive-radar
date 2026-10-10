#include "gpu_response.hpp"
#include "dsp.hpp" // Independent CPU minimum-phase reference.
#include "gpu_compute.hpp"
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(int argc,char** argv) {
    try {
        constexpr unsigned bins=6817;constexpr double tau=2*std::numbers::pi;
        const bool benchmark=argc>1 && std::string_view(argv[1])=="--benchmark";
        const unsigned rows=benchmark ? 1020 : 136;const auto count=rows*bins;
        auto context=std::make_shared<gpu::Context>();
        auto y_storage=gpu::make_buffer(context->gpu,count*sizeof(C));
        auto x_storage=gpu::make_buffer(context->gpu,count*sizeof(C));
        auto v_storage=gpu::make_buffer(context->gpu,count*sizeof(std::uint32_t));
        auto* y=static_cast<C*>(context->gpu.device->mapMemory(*y_storage.memory,0,count*sizeof(C)));
        auto* x=static_cast<C*>(context->gpu.device->mapMemory(*x_storage.memory,0,count*sizeof(C)));
        auto* valid=static_cast<std::uint32_t*>(context->gpu.device->mapMemory(*v_storage.memory,0,count*sizeof(std::uint32_t)));
        const gpu::BufferView measured{context,*y_storage.handle,count*sizeof(C),y};
        const ReferenceBatch reference{{context,*x_storage.handle,count*sizeof(C),x},{context,*v_storage.handle,count*sizeof(std::uint32_t),valid},rows};
        std::mt19937 rng(93513);const auto points=qam_constellation(6);
        for(unsigned i=0;i<count;++i){x[i]=points[rng()%64];valid[i]=(rng()%31)!=0;}
        GpuResponse gpu(context);double largest_error=0;
        for(unsigned order:{0U,1U,8U,32U}) {
            if(benchmark && order!=8)continue;
            const double origin=128.375;
            std::vector<C> rotated(count),masked_y(count);
            for(unsigned i=0;i<count;++i) {
                const double omega=tau*(int(i%bins)-3408)/8192;
                const auto h=C{.7F,.4F}*std::exp(order ? C{.18F,.07F}*static_cast<C>(std::polar(1.0,-omega)) : C{});
                const auto r=x[i]*static_cast<C>(std::polar(1.0,-omega*origin));
                y[i]=h*r;
                if(valid[i]){rotated[i]=r;masked_y[i]=y[i];}else y[i]=C{1e6F,-2e6F};
            }
            auto estimate=gpu.fit(measured,reference,origin,order);
            auto output=gpu.apply(measured,reference,estimate);
            if(benchmark) {
                ComputeTimings fit_time,apply_time;
                for(unsigned repeat=0;repeat<5;++repeat) {
                    estimate=gpu.fit(measured,reference,origin,order,16,&fit_time);
                    output=gpu.apply(measured,reference,estimate,&apply_time);
                }
                const double scale=1000/5/(rows*8704.0/8e6);
                std::cout<<"fit_ms_per_input_second,apply_ms_per_input_second\n"<<fit_time.gpu*scale<<','<<apply_time.gpu*scale<<'\n';
                continue;
            }
            const auto cpu=fit_minimum_phase(masked_y,rotated,bins,order,16);
            const auto* h=static_cast<const C*>(estimate.response.mapped);
            const auto* r=static_cast<const C*>(output.reference.mapped);
            const auto* cross=static_cast<const C*>(output.cross.mapped);
            const auto* normal=static_cast<const C*>(output.normalized.mapped);
            double error=0;
            for(unsigned k=0;k<bins;++k)error=std::max(error,double(std::abs(h[k]-cpu.response[k])/std::abs(cpu.response[k])));
            largest_error=std::max(largest_error,error);
            check(error<3e-5,"GPU minimum-phase response differs from double-precision reference");
            check(std::abs(estimate.relative_error-cpu.relative_error)<2e-6,"GPU fit error statistic disagrees with CPU");
            for(unsigned i=0;i<count;++i) {
                if(!valid[i]){check(r[i]==C{} && cross[i]==C{} && normal[i]==C{},"Invalid carrier leaked into residual");continue;}
                check(std::abs(r[i]-rotated[i])<3e-7,"Explicit reference-origin rotation changed phase");
                const C predicted=h[i%bins]*r[i],residual=y[i]-predicted;
                check(std::abs(cross[i]-residual*std::conj(r[i]))<5e-7,"Resident residual cross product changed phase");
                check(std::abs(normal[i]-residual/predicted)<2e-6,"Target/direct normalization changed phase");
            }
            std::cout<<"Order "<<order<<": CPU/GPU max relative response error="<<error<<'\n';
        }
        if(!benchmark) {
            // A pure all-pass delay must remain outside the estimated response.
            std::fill_n(valid,count,1U);
            for(unsigned i=0;i<count;++i)y[i]=x[i]*static_cast<C>(std::polar(.8,-tau*(int(i%bins)-3408)*8/8192+.9));
            auto estimate=gpu.fit(measured,reference,0,8);
            const auto* h=static_cast<const C*>(estimate.response.mapped);
            for(unsigned k=1;k<bins;++k)check(std::abs(h[k]-h[0])<2e-5,"Minimum-phase estimator absorbed an all-pass delay");
            check(std::abs(GpuResponse::direct_origin_adjustment(estimate)-8)<2e-5,"Explicit origin calibration has wrong sign/units");
            estimate=gpu.fit(measured,reference,8,8);
            const auto aligned=gpu.apply(measured,reference,estimate);
            const auto* residual=static_cast<const C*>(aligned.normalized.mapped);
            for(unsigned i=0;i<count;++i)check(std::abs(residual[i])<3e-6,"Explicit origin did not align pure delayed reference");
            // Estimate on a direct path, then inject an echo. Independent TX/gain
            // phases must cancel in the target/direct ratio; echo phase must not.
            double echo_error=0;
            for(double phase:{-1.7,.93}) {
                for(unsigned i=0;i<count;++i) {
                    const double omega=tau*(int(i%bins)-3408)/8192;
                    const auto h=std::polar(.8,phase)*std::exp(std::complex<double>{.18,.07}*std::polar(1.0,-omega));
                    y[i]=static_cast<C>(h)*x[i];
                }
                const auto fitted=gpu.fit(measured,reference,0,1,32);
                for(unsigned i=0;i<count;++i) {
                    const double angle=-tau*(int(i%bins)-3408)*37.25/8192+tau*29*(i/bins)*8704/8e6-.63;
                    y[i]*=C{1,0}+static_cast<C>(std::polar(.08,angle));
                }
                const auto echo=gpu.apply(measured,reference,fitted);
                const auto* actual=static_cast<const C*>(echo.normalized.mapped);
                for(unsigned i=0;i<count;++i) {
                    const double angle=-tau*(int(i%bins)-3408)*37.25/8192+tau*29*(i/bins)*8704/8e6-.63;
                    echo_error=std::max(echo_error,std::abs(std::complex<double>(actual[i])-std::polar(.08,angle)));
                }
            }
            check(echo_error<5e-6,"Minimum-phase residual normalization changed delayed moving-echo phase");
            std::fill_n(valid,count,0U);bool rejected=false;
            try{gpu.fit(measured,reference,0,8);}catch(const std::runtime_error&){rejected=true;}
            check(rejected,"Minimum-phase fit accepted an empty validity mask");
            std::cout<<"Masks, all-pass rejection, explicit origin, independent gain phase and moving echo PASS; echo max error="<<echo_error<<'\n';
        }
        context->gpu.device->unmapMemory(*y_storage.memory);context->gpu.device->unmapMemory(*x_storage.memory);context->gpu.device->unmapMemory(*v_storage.memory);
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
