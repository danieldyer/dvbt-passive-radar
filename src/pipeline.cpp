#include "pipeline.hpp"
#include "cpu_fec.hpp"
#include "broadcast_metadata.hpp"
#include "gpu_residual.hpp"
#include "gpu_frontend.hpp"
#include "gpu_receiver.hpp"
#include "gpu_frames.hpp"
#include "gpu_reconstruction.hpp"
#include "gpu_pipeline.hpp"
#include "dvbt_ofdm_synchronization_impl.h"
#include <gnuradio/dl5eu/dvbt_ofdm_synchronization.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/top_block.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace radar {
namespace {
constexpr std::size_t bins=6817;
class Input : public gr::sync_block {
    std::function<std::vector<std::vector<C>>()> next_;
    std::vector<std::vector<C>> buffer_;
    std::size_t position_=0;
public:
    std::exception_ptr failure;
    double elapsed=0;
    Input(std::size_t channels,std::function<std::vector<std::vector<C>>()> next) : gr::sync_block("recording_channelizer",
        gr::io_signature::make(0,0,0),gr::io_signature::make(channels,channels,sizeof(C))),next_(std::move(next)) {}
    int work(int n,gr_vector_const_void_star&,gr_vector_void_star& output) override {
        try {
        if(buffer_.empty() || position_==buffer_[0].size()) {
            const auto start=std::chrono::steady_clock::now();buffer_=next_();position_=0;
            elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        if(buffer_.empty() || buffer_[0].empty()) return WORK_DONE;
        const auto count=std::min<std::size_t>(n,buffer_[0].size()-position_);
        for(std::size_t c=0;c<buffer_.size();++c)
            std::copy_n(buffer_[c].data()+position_,count,static_cast<C*>(output[c]));
        position_+=count;
        return static_cast<int>(count);
        } catch(...) {failure=std::current_exception();return WORK_DONE;}
    }
};
template<class T> void write(std::ofstream& file,std::span<const T> data) {
    file.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size_bytes()));
    if(!file) throw std::runtime_error("Output write failed");
}
class Processor {
    const PipelineOptions& options_;
    struct Coordinates {std::uint64_t origin;std::uint32_t decimation;} channel_;
    std::shared_ptr<SampleClock> clock_;
    Modem modem_;
    std::vector<Symbol> pending_, batch_;
    Tps mode_{};
    bool aligned_=false, history_=false;
    int last_frame_=-1;
    bool origin_calibrated_=false;
    double timing_origin_=0;
    std::ofstream measured_,reference_,cross_,mask_,timing_,fits_,iq_,residual_iq_,tps_,coherence_;
    std::vector<C> diagnostic_channel_;
    std::unique_ptr<GpuResidual> gpu_;
    std::unique_ptr<CpuFec> fec_;
    std::unique_ptr<RangeDoppler> maps_;
    Fft fft_{8192};
    BroadcastMetadata broadcast_;
    std::size_t row_=0, batch_number_=0, segment_=0;
    double decoder_time_=0,fit_time_=0,output_time_=0;
    void process(bool final,Reconstruction* prepared=nullptr,const ResponseEstimate* estimate=nullptr,const ResidualBatch* residual=nullptr) {
        if(batch_.size()<68) return;
        const auto count=batch_.size();
        auto mark=std::chrono::steady_clock::now();
        auto reconstruction=prepared ? std::move(*prepared) : modem_.reconstruct(batch_,mode_,nullptr,fec_.get());
        decoder_time_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
        mark=std::chrono::steady_clock::now();
        broadcast_.batch(reconstruction.corrected_packets,reconstruction.packet_valid);
        packets+=reconstruction.packets; bad_packets+=reconstruction.bad_packets;
        std::vector<C> observed(count*bins), model=std::move(reconstruction.carriers);
        ResponseFit fit;
        if(estimate) {
            const auto* response=static_cast<const C*>(estimate->response.mapped);
            fit.response.assign(response,response+bins);fit.relative_error=estimate->relative_error;
            for(std::size_t s=0;s<count;++s)for(std::size_t k=0;k<bins;++k)
                if(reconstruction.valid[s*bins+k])observed[s*bins+k]=batch_[s].measured[k];
        } else {
        std::vector<C> rotation(bins);
        for(std::size_t k=0;k<bins;++k)rotation[k]=std::polar(1.0F,static_cast<float>(
            -2*std::numbers::pi*(static_cast<double>(k)-3408)*(fft_window_margin+options_.reference_delay_samples)/8192));
        for(std::size_t s=0;s<count;++s) for(std::size_t k=0;k<bins;++k) {
            const auto i=s*bins+k;
            // Start early enough that the fractional-resampler support stays in the cyclic extension.
            // Additional fixed timing calibration is supplied explicitly, never fitted here.
            model[i]*=rotation[k];
            if(reconstruction.valid[i]) observed[i]=batch_[s].measured[k];
            else model[i]=C{};
        }
        if(!options_.automatic_origin)origin_calibrated_=true;
        if(!origin_calibrated_) {
            fit=fit_minimum_phase(observed,model,bins,options_.response_order,16);
            // Establish the fractional direct-symbol origin once per acquisition segment.
            // CP acquisition is integer/coarse: it does not define a fractionally exact origin.
            // This fixed coordinate choice shifts every path equally; the FRF has no delay parameter.
            std::vector<std::complex<double>> transfer(bins);
            std::vector<double> power(bins);
            for(std::size_t i=0;i<model.size();++i) {
                transfer[i%bins]+=std::complex<double>(observed[i])*std::conj(std::complex<double>(model[i]));
                power[i%bins]+=std::norm(model[i]);
            }
            double phase=0,previous=0,sw=0,sx=0,sy=0,sxx=0,sxy=0;
            for(std::size_t k=0;k<bins;++k) {
                const auto ratio=transfer[k]/std::complex<double>(fit.response[k]);
                const double angle=std::arg(ratio);
                phase+=k ? std::remainder(angle-previous,2*std::numbers::pi) : angle;
                previous=angle;
                const double weight=std::abs(transfer[k]),x=static_cast<double>(k)-3408;
                sw+=weight;sx+=weight*x;sy+=weight*phase;sxx+=weight*x*x;sxy+=weight*x*phase;
            }
            const double slope=(sw*sxy-sx*sy)/(sw*sxx-sx*sx);
            timing_origin_=-slope*8192/(2*std::numbers::pi);
            origin_calibrated_=true;
            std::cout<<"Fixed fractional DVB-T origin adjustment: "<<timing_origin_<<" samples\n";
        }
        for(std::size_t k=0;k<bins;++k)rotation[k]=std::polar(1.0F,static_cast<float>(-2*std::numbers::pi*(static_cast<double>(k)-3408)*timing_origin_/8192));
        for(std::size_t s=0;s<count;++s)for(std::size_t k=0;k<bins;++k)model[s*bins+k]*=rotation[k];
        fit=fit_minimum_phase(observed,model,bins,options_.response_order,16);
        }
        fit_time_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
        mark=std::chrono::steady_clock::now();
        const auto copied_cross=gpu_ && !residual ? gpu_->calculate(observed,model,fit.response) : std::vector<C>{};
        const std::span<const C> gpu_cross=residual ? std::span<const C>{static_cast<const C*>(residual->cross.mapped),count*bins} : std::span<const C>{copied_cross};
        if(diagnostic_channel_.empty()) {
            diagnostic_channel_.resize(bins);
            std::vector<double> power(bins);
            for(std::size_t i=0;i<model.size();++i) {
                diagnostic_channel_[i%bins]+=observed[i]*std::conj(model[i]);power[i%bins]+=std::norm(model[i]);
            }
            for(std::size_t k=0;k<bins;++k)diagnostic_channel_[k]/=static_cast<float>(std::max(power[k],1e-30));
        }
        if(!options_.summary_only || batch_number_==0)
            for(std::size_t k=0;k<bins;++k) fits_<<batch_number_<<','<<k<<','<<fit.response[k].real()<<','<<fit.response[k].imag()<<'\n';
        const std::size_t first=history_ ? 68 : 0;
        const std::size_t last=final ? count : count-68;
        std::vector<C> spectrum(bins),ref_fft(8192),error_fft(8192),time_reference;
        for(std::size_t s=first;s<last;++s) {
            const auto off=s*bins;
            bool full=true;
            std::complex<double> phase_error{};
            double sw=0,sx=0,sy=0,sxx=0,sxy=0;
            std::fill(ref_fft.begin(),ref_fft.end(),C{});
            std::fill(error_fft.begin(),error_fft.end(),C{});
            for(std::size_t k=0;k<bins;++k) {
                const auto i=off+k;
                const auto ref=model[i];
                const auto error=batch_[s].measured[k]-fit.response[k]*ref;
                spectrum[k]=reconstruction.valid[i] ? (!gpu_cross.empty() ? gpu_cross[i] : error*std::conj(ref)) : C{};
                const auto bin=(k+8192-3408)%8192;
                ref_fft[bin]=ref;
                error_fft[bin]=reconstruction.valid[i] ? error : C{};
                full=full && reconstruction.valid[i];
                if(reconstruction.valid[i]) {
                    const auto z=std::complex<double>(batch_[s].measured[k])*std::conj(std::complex<double>(ref*diagnostic_channel_[k]));
                    phase_error+=z;
                    const double w=std::abs(z),x=static_cast<double>(k)-3408,y=std::arg(z);
                    sw+=w;sx+=w*x;sy+=w*y;sxx+=w*x*x;sxy+=w*x*y;
                }
            }
            coherence_<<row_<<','<<segment_<<','<<std::arg(phase_error)*180/std::numbers::pi<<','
                <<-(sw*sxy-sx*sy)/std::max(sw*sxx-sx*sx,1e-30)*8192/(2*std::numbers::pi)<<'\n';
            if(maps_)maps_->append(spectrum,(channel_.origin+batch_[s].sample*channel_.decimation)/(8e6*channel_.decimation),
                std::arg(phase_error),-(sw*sxy-sx*sy)/std::max(sw*sxx-sx*sx,1e-30)*8192/(2*std::numbers::pi),full,segment_);
            if(!options_.summary_only) {
            write<C>(measured_,batch_[s].measured);
            write<C>(reference_,std::span<const C>{model}.subspan(off,bins));
            write<C>(cross_,spectrum);
            write<std::uint8_t>(mask_,std::span<const std::uint8_t>{reconstruction.valid}.subspan(off,bins));
            fft_.transform(ref_fft,true); fft_.transform(error_fft,true);
            const auto cp=8192/static_cast<std::size_t>(options_.guard);
            write<C>(iq_,std::span<const C>{ref_fft}.last(cp));
            write<C>(iq_,ref_fft);
            // Useful OFDM window only: no invented measurements in the cyclic-prefix interval.
            write<C>(residual_iq_,error_fft);
            }
            const auto& symbol=batch_[s];
            const double corrected=channel_.origin+symbol.sample*channel_.decimation;
            const auto clock=clock_ ? clock_->at(corrected) : ClockSegment{};
            timing_<<row_<<','<<segment_<<','<<batch_number_<<','
                <<corrected+clock.offset_at(corrected)<<','
                <<symbol.sample_step*channel_.decimation*clock.rate<<','<<symbol.phase<<','
                <<symbol.frequency_hz<<','<<(symbol.sample_step*clock.rate-1)*1e6<<','<<full<<','
                <<fft_window_margin+options_.reference_delay_samples+timing_origin_<<','<<corrected<<'\n';
            ++row_; valid_symbols+=full;
        }
        std::cout<<"Batch "<<batch_number_++<<": "<<count<<" symbols, RS "
                 <<reconstruction.packets-reconstruction.bad_packets<<'/'<<reconstruction.packets
                 <<" passed, "<<reconstruction.corrected_bytes<<" corrected bytes; residual/input "
                 <<10*std::log10(std::max(fit.relative_error,1e-30))<<" dB\n"<<std::flush;
        output_time_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
        if(!final) {
            batch_.erase(batch_.begin(),batch_.end()-136);
            history_=true;
        }
    }
public:
    std::size_t packets=0,bad_packets=0,valid_symbols=0,tps_frames=0;
    Processor(const PipelineOptions& options,std::uint64_t origin,std::uint32_t decimation,std::shared_ptr<SampleClock> clock={}) : options_(options),channel_{origin,decimation},clock_(std::move(clock)),modem_(options.guard),broadcast_(options.output/"broadcast_metadata.txt") {
        auto open=[&](const char* name,bool binary=false) {
            std::ofstream f(options.output/name,binary ? std::ios::binary : std::ios::out);
            if(!f) throw std::runtime_error(std::string("Cannot create output: ")+name);
            f<<std::setprecision(17);
            return f;
        };
        if(!options.summary_only) {
            measured_=open("measured.cf32",true); reference_=open("reference.cf32",true);
            cross_=open("residual_cross.cf32",true); mask_=open("valid.u8",true);
            iq_=open("reference_iq.cf32",true); residual_iq_=open("residual_useful_iq.cf32",true);
        }
        timing_=open("timing.csv"); fits_=open("response.csv");tps_=open("tps.csv");
        coherence_=open("coherence.csv");
        coherence_<<"row,segment,phase_error_deg,timing_error_samples\n";
        if(options.gpu && !options.gpu_frontend)gpu_=std::make_unique<GpuResidual>();
        if(!options.gpu_frontend)fec_=std::make_unique<CpuFec>();
        if(options.maps)maps_=std::make_unique<RangeDoppler>(options.map,(8192+8192/options.guard)/8e6,options.output);
        timing_<<"row,segment,response_batch,input_sample,input_samples_per_output_sample,applied_carrier_phase_rad,carrier_error_hz,sample_rate_error_ppm,all_carriers_valid,reference_origin_samples,corrected_adc_sample\n";
        fits_<<"batch,carrier,real,imag\n";
        tps_<<"frame,constellation,rate,guard,mode,hierarchy,bits,input_sample,segment\n";
    }
    void append(Symbol symbol) {
        if(symbol.discontinuity) {
            if(!batch_.empty()) process(true);
            pending_.clear();batch_.clear();aligned_=false;history_=false;origin_calibrated_=false;diagnostic_channel_.clear();++segment_;
        }
        pending_.push_back(std::move(symbol));
        if(pending_.size()<68) return;
        Tps t;
        if(!modem_.tps(pending_,t)) {
            if(aligned_) {
                if(!batch_.empty())process(true);
                batch_.clear();history_=false;aligned_=false;origin_calibrated_=false;diagnostic_channel_.clear();++segment_;
            }
            pending_.erase(pending_.begin());
            return;
        }
        if(t.guard != (options_.guard==32 ? 0 : options_.guard==16 ? 1 : options_.guard==8 ? 2 : 3))
            throw std::runtime_error("BCH-valid TPS disagrees with selected guard interval");
        if(tps_frames==0) std::cout<<"TPS BCH pass: frame="<<t.frame<<" constellation="<<t.constellation
            <<" rate="<<t.rate<<" guard="<<t.guard<<" mode="<<t.mode<<" hierarchy="<<t.hierarchy<<'\n'<<std::flush;
        if(aligned_ && (t.constellation!=mode_.constellation || t.rate!=mode_.rate || t.mode!=mode_.mode || t.hierarchy!=mode_.hierarchy))
            throw std::runtime_error("DVB-T mode changed within recording segment");
        if(aligned_ && t.frame!=(last_frame_+1)%4) {
            if(!batch_.empty())process(true);
            batch_.clear();history_=false;origin_calibrated_=false;diagnostic_channel_.clear();++segment_;
        }
        for(int i=0;i<68;++i) if(pending_[i].modulo4!=i%4) throw std::runtime_error("TPS/pilot symbol alignment disagrees");
        tps_<<t.frame<<','<<t.constellation<<','<<t.rate<<','<<t.guard<<','<<t.mode<<','<<t.hierarchy<<',';
        for(auto bit:t.bits)tps_<<static_cast<int>(bit);
        tps_<<','<<channel_.origin+pending_.front().sample*channel_.decimation<<','<<segment_<<'\n'; ++tps_frames;
        aligned_=true;mode_=t;last_frame_=t.frame;
        for(auto& s:pending_)batch_.push_back(std::move(s));
        pending_.clear();
        if(batch_.size()>=1020)process(false);
    }
    void spectral(const SpectralBatch& output) {
        // Host materialization belongs to this legacy output consumer. The
        // numerical pipeline has already produced all complex spectra on device.
        const auto& frame=output.frame;
        Reconstruction reconstruction;
        const auto* reference=static_cast<const C*>(output.residual.reference.mapped);
        const auto* mask=static_cast<const std::uint32_t*>(output.residual.valid.mapped);
        reconstruction.carriers.assign(reference,reference+frame.symbols.size()*bins);
        reconstruction.valid.assign(mask,mask+frame.symbols.size()*bins);
        const auto packets=output.packets.packets;
        for(std::size_t p=0;p<output.packets.count;++p) {
            const auto& packet=packets[p];
            ++reconstruction.packets;reconstruction.bad_packets+=!packet.valid;reconstruction.corrected_bytes+=packet.corrected;
            std::array<std::uint8_t,188> data;std::copy_n(packet.bytes.begin(),188,data.begin());
            reconstruction.corrected_packets.push_back(data);reconstruction.packet_valid.push_back(packet.valid);
        }
        if(segment_!=frame.segment) {
            origin_calibrated_=false;diagnostic_channel_.clear();segment_=frame.segment;
        }
        const auto* measured=static_cast<const C*>(frame.carriers.measured.mapped);
        batch_.clear();batch_.resize(frame.symbols.size());
        for(std::size_t j=0;j<batch_.size();++j) {
            auto& symbol=batch_[j];const auto& source=frame.symbols[j];
            symbol.measured.assign(measured+j*bins,measured+(j+1)*bins);
            symbol.sample=static_cast<double>(source.window.sample);symbol.sample_step=1;
            symbol.phase=source.window.phase;symbol.frequency_hz=source.window.frequency_hz;symbol.modulo4=source.modulo4;
        }
        for(std::size_t j=frame.first ? 2 : 0;j<frame.frames.size();++j) {
            const auto& t=frame.frames[j];
            if(!tps_frames)std::cout<<"TPS BCH pass: frame="<<t.frame<<" constellation="<<t.constellation
                <<" rate="<<t.rate<<" guard="<<t.guard<<" mode="<<t.mode<<" hierarchy="<<t.hierarchy<<'\n'<<std::flush;
            tps_<<t.frame<<','<<t.constellation<<','<<t.rate<<','<<t.guard<<','<<t.mode<<','<<t.hierarchy<<',';
            for(auto bit:t.bits)tps_<<static_cast<int>(bit);
            tps_<<','<<channel_.origin+batch_[j*68].sample*channel_.decimation<<','<<segment_<<'\n';++tps_frames;
        }
        timing_origin_=output.residual.reference_origin-fft_window_margin-options_.reference_delay_samples;
        origin_calibrated_=true;mode_=frame.frames.front();history_=frame.first!=0;
        process(frame.last==batch_.size(),&reconstruction,&output.estimate,&output.residual);batch_.clear();
    }
    void finish() {
        if(!batch_.empty())process(true);if(maps_)maps_->finish();
        if(options_.gpu && options_.gpu_frontend)std::cout<<"Legacy output CPU seconds: "<<output_time_<<'\n';
        else std::cout<<"Stage seconds: decoder="<<decoder_time_<<" response="<<fit_time_<<" output="<<output_time_<<'\n';
    }
};
class Output : public gr::sync_block {
    std::function<void(Symbol)> receive_;
public:
    std::exception_ptr failure;
    explicit Output(std::function<void(Symbol)> receive) : gr::sync_block("reconstruct_residual",
        gr::io_signature::make(2,2,bins*sizeof(C)),gr::io_signature::make(0,0,0)),receive_(std::move(receive)) {}
    int work(int n,gr_vector_const_void_star& input,gr_vector_void_star&) override {
        try {
            const auto* eq=static_cast<const C*>(input[0]);
            const auto* raw=static_cast<const C*>(input[1]);
            for(int i=0;i<n;++i) {
                Symbol s;
                s.equalized.assign(eq+i*bins,eq+(i+1)*bins);
                s.measured.assign(raw+i*bins,raw+(i+1)*bins);
                std::vector<gr::tag_t> tags;
                get_tags_in_range(tags,0,nitems_read(0)+i,nitems_read(0)+i+1);
                bool have_time=false,have_symbol=false;
                for(const auto& tag:tags) {
                    const auto key=pmt::symbol_to_string(tag.key);
                    if(key=="radar_timing") {
                        s.sample=pmt::to_double(pmt::tuple_ref(tag.value,0));
                        s.sample_step=pmt::to_double(pmt::tuple_ref(tag.value,1));
                        s.phase=pmt::to_double(pmt::tuple_ref(tag.value,2));
                        s.frequency_hz=pmt::to_double(pmt::tuple_ref(tag.value,3));
                        have_time=true;
                    } else if(key=="relative_symbol_index") { s.modulo4=static_cast<int>(pmt::to_long(tag.value));have_symbol=true; }
                    else if(key=="resync" || key=="sync_start") s.discontinuity=true;
                }
                if(!have_time || !have_symbol)throw std::runtime_error("Synchronizer omitted sample coordinates");
                receive_(std::move(s));
            }
            return n;
        } catch(...) { failure=std::current_exception();return WORK_DONE; }
    }
};
class MemoryInput : public gr::sync_block {
    std::span<const C> input_;
public:
    explicit MemoryInput(std::span<const C> input) : gr::sync_block("test_samples",
        gr::io_signature::make(0,0,0),gr::io_signature::make(1,1,sizeof(C))),input_(input) {}
    int work(int n,gr_vector_const_void_star&,gr_vector_void_star& out) override {
        if(input_.empty())return WORK_DONE;
        const auto count=std::min<std::size_t>(n,input_.size());
        std::copy_n(input_.begin(),count,static_cast<C*>(out[0]));input_=input_.subspan(count);
        return static_cast<int>(count);
    }
};
}
void synchronize_samples(std::span<const C> input,int guard,const std::function<void(Symbol)>& receive,
                         std::shared_ptr<SampleClock> clock,std::uint64_t origin,unsigned decimation) {
    auto source=gnuradio::make_block_sptr<MemoryInput>(input);
    auto sink=gnuradio::make_block_sptr<Output>(receive);
    const auto gi=guard==4 ? gr::dtv::GI_1_4 : guard==8 ? gr::dtv::GI_1_8 : guard==16 ? gr::dtv::GI_1_16 : gr::dtv::GI_1_32;
    auto sync=gr::dl5eu::dvbt_ofdm_synchronization::make(gr::dtv::T8k,gi,30,true);
    auto instrumented=std::static_pointer_cast<gr::dl5eu::dvbt_ofdm_synchronization_impl>(sync);
    if(clock)instrumented->radar_use_common_clock(clock,0,origin,decimation);
    auto graph=gr::make_top_block("phase_coordinate_test");
    graph->connect(source,0,sync,0);graph->connect(sync,0,sink,0);graph->connect(sync,1,sink,1);
    sync->set_max_noutput_items(8);sink->set_max_noutput_items(8);graph->run(65536);
    if(sink->failure)std::rethrow_exception(sink->failure);
}
int process_recording(const PipelineOptions& options) {
    WaveReader wave(options.input);
    if(std::filesystem::exists(options.output) && !std::filesystem::is_empty(options.output))
        throw std::runtime_error("Output directory must be empty (existing results are not overwritten)");
    const auto offsets=options.offsets_hz.empty() ? std::vector{options.offset_hz} : options.offsets_hz;
    if(offsets.size()>64)throw std::runtime_error("At most 64 output channels are supported");
    const double seconds=options.seconds==0 ? static_cast<double>(wave.info.samples)/wave.info.sample_rate-options.start_seconds : options.seconds;
    std::unique_ptr<GpuChannelBank> bank;
    std::vector<std::unique_ptr<Channelizer>> cpu_channels;
    auto clock=std::make_shared<SampleClock>(wave.info.sample_rate,offsets,options.clock_phase_budget_degrees);
    if(!options.gpu_frontend && offsets.size()>1)
        throw std::runtime_error("The CPU comparison pipeline is single-channel; coherent multi-channel processing uses --frontend vulkan");
    std::uint64_t origin{};std::uint32_t decimation{};
    std::function<std::vector<std::vector<C>>()> next;
    if(options.gpu_frontend) {
        bank=std::make_unique<GpuChannelBank>(wave,offsets,options.start_seconds,seconds);
        bank->set_clock(clock);
        origin=bank->origin;decimation=bank->decimation;
        next=[&]{return bank->next();};
        std::cout<<"Vulkan channel bank: "<<offsets.size()<<" channels, "<<bank->taps()<<" taps, "<<bank->capacity()<<" samples/batch; common ADC interpolator "<<bank->interpolation_taps()<<" taps\n";
    } else {
        for(double offset:offsets)cpu_channels.push_back(std::make_unique<Channelizer>(wave,offset,options.start_seconds,seconds));
        origin=cpu_channels[0]->origin;decimation=cpu_channels[0]->decimation;
        next=[&]{std::vector<std::vector<C>> values;for(auto& c:cpu_channels)values.push_back(c->next());return values;};
    }
    std::filesystem::create_directories(options.output);
    std::cout<<"Input: "<<wave.info.sample_rate<<" samples/s, "<<wave.info.samples<<" complex samples"
        <<(wave.info.wrapped_size ? " (wrapped RIFF sizes recovered)" : "")
        <<"; decimation "<<decimation<<"; common input centre "<<origin<<'\n'<<std::flush;
    std::vector<PipelineOptions> channel_options;
    for(std::size_t c=0;c<offsets.size();++c) {
        auto current=options;current.offset_hz=offsets[c];
        if(offsets.size()>1) current.output/=std::string("channel_")+(c<10 ? "0" : "")+std::to_string(c);
        std::filesystem::create_directories(current.output);
        channel_options.push_back(std::move(current));
    }
    std::vector<std::unique_ptr<Processor>> processors;
    for(std::size_t c=0;c<offsets.size();++c)
        processors.push_back(std::make_unique<Processor>(channel_options[c],origin,decimation,options.gpu_frontend ? clock : nullptr));
    double frontend_seconds=0;
    const bool gpu_receiver=options.gpu && options.gpu_frontend;
    if(gpu_receiver) {
        const std::vector<unsigned> guards(offsets.size(),static_cast<unsigned>(options.guard));
        GpuPipelineTimings timing;
        const GpuPipelineConfig config{guards,bank->capacity(),origin,decimation,options.response_order,
            options.reference_delay_samples,options.automatic_origin};
        GpuPipeline pipeline(config,bank->context(),clock);
        auto consume=[&](const SpectralBatch& output){processors[output.frame.channel]->spectral(output);};
        for(;;) {
            const auto mark=std::chrono::steady_clock::now();
            const auto input=bank->next_device();
            frontend_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-mark).count();
            if(!input.count)break;
            pipeline.process(input,consume,&timing);
        }
        pipeline.finish(consume,&timing);
        std::cout<<"GPU seconds: OFDM="<<timing.receiver.gpu<<" frame_copy="<<timing.frames.gpu
            <<" reconstruction="<<timing.reconstruction.gpu<<" response_residual="<<timing.response.gpu<<'\n';
        std::cout<<"CPU FEC seconds: demap="<<timing.fec.demap<<" Viterbi="<<timing.fec.viterbi
            <<" RS="<<timing.fec.reed_solomon<<'\n';
    } else {
        auto source=gnuradio::make_block_sptr<Input>(offsets.size(),next);
        source->set_min_output_buffer(262144);
        const auto guard=options.guard==4 ? gr::dtv::GI_1_4 : options.guard==8 ? gr::dtv::GI_1_8 : options.guard==16 ? gr::dtv::GI_1_16 : gr::dtv::GI_1_32;
        auto graph=gr::make_top_block("dvbt_passive_radar");
        std::vector<std::shared_ptr<Output>> sinks;
        for(std::size_t c=0;c<offsets.size();++c) {
            auto sink=gnuradio::make_block_sptr<Output>([&,c](Symbol s){processors[c]->append(std::move(s));});
            auto sync=std::static_pointer_cast<gr::dl5eu::dvbt_ofdm_synchronization_impl>(
                gr::dl5eu::dvbt_ofdm_synchronization::make(gr::dtv::T8k,guard,20,true));
            if(options.gpu_frontend)sync->radar_use_common_clock(clock,c,origin,decimation);
            graph->connect(source,c,sync,0);graph->connect(sync,0,sink,0);graph->connect(sync,1,sink,1);
            sync->set_max_noutput_items(8);sink->set_max_noutput_items(8);
            sinks.push_back(sink);
        }
        graph->run(65536);
        if(source->failure)std::rethrow_exception(source->failure);
        for(auto& sink:sinks)if(sink->failure)std::rethrow_exception(sink->failure);
        frontend_seconds=source->elapsed;
    }
    if(options.gpu_frontend) {
        std::ofstream trajectory(options.output/"sample_clock.csv");
        trajectory<<std::setprecision(17)<<"corrected_adc_sample,original_adc_offset_samples,input_samples_per_corrected_sample\n";
        for(auto s:clock->segments())trajectory<<s.start<<','<<s.offset<<','<<s.rate<<'\n';
        if(!trajectory)throw std::runtime_error("Common sample-clock trajectory write failed");
    }
    std::cout<<"Channelizer seconds: "<<frontend_seconds<<'\n';
    bool valid=true;
    for(std::size_t c=0;c<offsets.size();++c) {
    auto& processor=*processors[c];processor.finish();
    const auto& current=channel_options[c];
    std::ofstream metadata(current.output/"metadata.json");
    metadata<<std::setprecision(17)<<"{\n  \"sample_rate\": 8000000,\n  \"fft_size\": 8192,\n  \"active_carriers\": 6817,\n"
        <<"  \"carrier_order\": \"-3408..3408\",\n  \"complex_format\": \"little-endian float32 I,Q\",\n"
        <<"  \"input_sample_rate\": "<<wave.info.sample_rate<<",\n  \"input_samples\": "<<wave.info.samples<<",\n"
        <<"  \"requested_seconds\": "<<options.seconds<<",\n"
        <<"  \"automatic_reference_origin\": "<<(options.automatic_origin ? "true" : "false")<<",\n"
        <<"  \"nco_offset_hz\": "<<current.offset_hz<<",\n  \"input_origin\": "<<origin
        <<",\n  \"frontend_compute\": \""<<(options.gpu_frontend ? "vulkan" : "cpu")<<"\""
        <<",\n  \"ofdm_compute\": \""<<(gpu_receiver ? "vulkan" : "cpu")<<"\""
        <<",\n  \"reconstruction_compute\": \""<<(gpu_receiver ? "vulkan" : "cpu")<<"\""
        <<",\n  \"response_compute\": \""<<(gpu_receiver ? "vulkan" : "cpu")<<"\""
        <<",\n  \"fec_decoder\": \""<<"cpu_hard_viterbi_rs"<<"\""
        <<",\n  \"shared_adc_clock\": "<<(options.gpu_frontend ? "true" : "false")
        <<",\n  \"channel_index\": "<<c<<",\n  \"channel_count\": "<<offsets.size()
        <<",\n  \"nco_phase_origin_input_sample\": 0"
        <<",\n  \"timing_coordinates\": \"original ADC samples and corrected_adc_sample; exact warp in root sample_clock.csv for Vulkan frontend\""
        <<",\n  \"clock_phase_budget_rms_degrees\": "<<options.clock_phase_budget_degrees
        <<",\n  \"clock_timing_budget_adc_samples\": "<<clock->timing_budget_samples()
        <<",\n  \"guard_denominator\": "<<options.guard<<",\n  \"response_cepstral_order\": "<<options.response_order
        <<",\n  \"reference_delay_samples\": "<<options.reference_delay_samples
        <<",\n  \"summary_only\": "<<(options.summary_only ? "true" : "false")
        <<",\n  \"residual_compute\": \""<<(options.gpu ? "vulkan" : "cpu")<<"\""
        <<",\n  \"tps_frames\": "<<processor.tps_frames<<",\n  \"rs_packets_examined_including_overlap\": "<<processor.packets
        <<",\n  \"rs_failures_including_overlap\": "<<processor.bad_packets
        <<",\n  \"fully_valid_symbols\": "<<processor.valid_symbols
        <<",\n  \"absolute_time_calibrated\": false,\n  \"coherence_qualified\": false\n}\n";
    if(!metadata)throw std::runtime_error("Metadata write failed");
    std::cout<<"Output: "<<current.output<<"; fully valid symbols: "<<processor.valid_symbols
        <<'\n';
    valid=valid && processor.tps_frames && processor.valid_symbols;
    }
    if(offsets.size()>1) {
        std::ofstream manifest(options.output/"channels.json");
        manifest<<std::setprecision(17)<<"{\"input_sample_rate\":"<<wave.info.sample_rate<<",\"input_origin\":"<<origin
            <<",\"decimation\":"<<decimation<<",\"offsets_hz\":[";
        for(std::size_t c=0;c<offsets.size();++c)manifest<<(c ? "," : "")<<offsets[c];
        manifest<<"]}\n";
        if(!manifest)throw std::runtime_error("Channel manifest write failed");
    }
    if(!valid)throw std::runtime_error("A channel has no fully FEC-validated reconstruction; inspect its TPS/guard/frequency and timing outputs");
    return 0;
}
} // namespace radar
