#include "recording_metadata.hpp"
#include "gpu_frontend.hpp"
#include "gpu_noise.hpp"
#include "stationary_background.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <set>
#include <cmath>

// Read-only acquisition/FEC check for a metadata-defined capture. Counters
// include overlapping decoder context and are not unique transport packet counts.
int main(int argc,char** argv) {
    try {
        if(argc<3)throw std::invalid_argument("recording_probe INPUT.wav METADATA.json [seconds=20] [start=0] [frames_per_batch=15] [analyse=0]");
        std::ostream result(std::cout.rdbuf());std::cout.rdbuf(std::cerr.rdbuf());
        const double seconds=argc>3 ? std::stod(argv[3]) : 20,start=argc>4 ? std::stod(argv[4]) : 0;
        const bool analyse=argc>6 && std::stoi(argv[6]);
        const auto metadata=radar::RecordingMetadata::read(argv[2]);radar::WaveReader wave(argv[1]);metadata.validate_wave(wave.info);
        radar::SarCapture capture;metadata.configure(capture);capture.source_rate=wave.info.sample_rate;
        std::vector<double> centres;radar::GpuPipelineConfig config;
        if(argc>5)config.frames_per_batch=std::stoul(argv[5]);
        for(const auto& c:capture.channels){centres.push_back(c.nco_hz);config.guards.push_back(c.guard);}
        radar::GpuChannelBank bank(wave,centres,start,seconds);capture.input_rate=bank.coordinate_rate();capture.decimation=bank.decimation;
        auto clock=std::make_shared<radar::SampleClock>(capture.input_rate,centres);bank.set_clock(clock);
        capture.origin=bank.origin;capture.validate();config.origin=bank.origin;config.decimation=bank.decimation;config.max_samples=bank.capacity();
        radar::GpuPipeline pipeline(config,bank.context(),clock);
        struct Count {
            std::uint64_t rows{},packets{},bad{},corrected{},edge_packets{},edge_bad{};
            std::set<std::array<int,5>> modes;
            boost::json::array failed_batches;
            double measured{},residual{},left{},right{},maximum_source_rate{};
            std::complex<double> adjacent{};
            std::unique_ptr<radar::GpuNoise> noise;
            radar::StationaryBackground background;
            std::uint64_t background_end{},segment=UINT64_MAX;
            double heldout_power{},heldout_residual{};
        };
        std::vector<Count> counts(centres.size());
        if(analyse)for(auto& c:counts)c.noise=std::make_unique<radar::GpuNoise>(bank.context());
        auto consume=[&](const radar::SpectralBatch& spectra){
            auto& c=counts.at(spectra.frame.channel);c.rows+=spectra.frame.last-spectra.frame.first;
            std::uint64_t bad=0;
            boost::json::array failed_positions;
            for(std::size_t p=0;p<spectra.packets.count;++p){const auto& packet=spectra.packets.packets[p];
                ++c.packets;bad+=!packet.valid;c.corrected+=packet.corrected;
                if(p<16 || p+16>=spectra.packets.count){++c.edge_packets;c.edge_bad+=!packet.valid;}
                if(!packet.valid)failed_positions.emplace_back(p);
            }
            c.bad+=bad;
            if(analyse){
                if(c.segment!=spectra.frame.segment){c.segment=spectra.frame.segment;c.background.reset();c.noise->reset();}
                if(!c.background.ready()){
                    c.noise->estimate(spectra);
                    c.background.observe(spectra,(8192.+8192./capture.channels[spectra.frame.channel].guard)/8e6,c.noise->training());
                    c.background_end=spectra.frame.symbols.back().window.sample;
                }
                const auto* y=static_cast<const radar::C*>(spectra.frame.carriers.measured.mapped);
                const auto* x=static_cast<const radar::C*>(spectra.residual.reference.mapped);
                const auto* h=static_cast<const radar::C*>(spectra.estimate.response.mapped);
                const auto* valid=static_cast<const std::uint32_t*>(spectra.transmitted.valid.mapped);
                for(std::size_t row=spectra.frame.first;row<spectra.frame.last;++row){
                    c.maximum_source_rate=std::max(c.maximum_source_rate,std::abs(spectra.frame.symbols[row].source_rate_error));
                    for(unsigned k=0;k<6817;++k){const auto at=row*6817+k;if(!valid[at])continue;
                        const auto error=std::complex<double>(y[at])-std::complex<double>(h[k])*std::complex<double>(x[at]);
                        c.measured+=std::norm(std::complex<double>(y[at]));c.residual+=std::norm(error);
                        if(c.background.ready() && spectra.frame.symbols[row].window.sample>c.background_end){
                            const auto predicted=std::complex<double>(h[k])*std::complex<double>(x[at])*std::complex<double>(c.background.response()[k]);
                            c.heldout_power+=std::norm(std::complex<double>(y[at]));c.heldout_residual+=std::norm(error-predicted);
                        }
                        if(row && valid[at-6817]){
                            const auto previous=std::complex<double>(y[at-6817])-std::complex<double>(h[k])*std::complex<double>(x[at-6817]);
                            c.adjacent+=error*std::conj(previous);c.left+=std::norm(previous);c.right+=std::norm(error);
                        }
                    }
                }
            }
            if(bad)c.failed_batches.emplace_back(boost::json::object{
                {"emitted_start_s",(capture.origin+spectra.frame.symbols.at(spectra.frame.first).window.sample*capture.decimation)/double(capture.input_rate)},
                {"emitted_end_s",(capture.origin+spectra.frame.symbols.at(spectra.frame.last-1).window.sample*capture.decimation)/double(capture.input_rate)},
                {"failed_rs_words_including_context",bad},{"packet_positions",failed_positions}});
            for(const auto& t:spectra.frame.frames)c.modes.insert({t.constellation,t.rate,t.guard,t.mode,t.hierarchy});
        };
        for(;;){const auto input=bank.next_device();if(!input.count)break;pipeline.process(input,consume);}pipeline.finish(consume);
        boost::json::array signals;bool empty=false;
        for(unsigned i=0;i<counts.size();++i){
            const auto& c=counts[i];boost::json::array modes;
            for(const auto& m:c.modes)modes.emplace_back(boost::json::array{m[0],m[1],m[2],m[3],m[4]});
            signals.emplace_back(boost::json::object{{"id",metadata.signals[i].id},{"rf_hz",capture.channels[i].rf_hz},
                {"nco_hz",capture.channels[i].nco_hz},{"symbols",c.rows},{"packets_examined_including_context",c.packets},
                {"failed_rs_words",c.bad},{"corrected_bytes",c.corrected},{"bch_validated_tps_modes",modes},
                {"edge_packets",c.edge_packets},{"edge_rejected",c.edge_bad},
                {"failed_batches",c.failed_batches},
                {"analysis_enabled",analyse},{"valid_carrier_power_reduction_db",analyse ? 10*std::log10(c.measured/std::max(c.residual,1e-300)) : 0},
                {"pooled_residual_lag1_correlation",analyse ? std::abs(c.adjacent)/std::sqrt(c.left*c.right) : 0},
                {"frozen_background_power_reduction_db",c.heldout_power>0 ? 10*std::log10(c.heldout_power/std::max(c.heldout_residual,1e-300)) : 0},
                {"maximum_source_clock_ppm",c.maximum_source_rate*1e6}});
            empty|=!c.rows || !c.packets;
        }
        const auto segments=clock->segments();double minimum=1,maximum=1;
        for(const auto& segment:segments){minimum=std::min(minimum,segment.rate);maximum=std::max(maximum,segment.rate);}
        result<<boost::json::serialize(boost::json::object{{"start_s",start},{"seconds",seconds},{"source_rate",capture.source_rate},
            {"coordinate_rate",capture.input_rate},{"frames_per_batch",config.frames_per_batch},{"input_samples",wave.info.samples},{"wrapped_riff",wave.info.wrapped_size},
            {"phase_budget_degrees",clock->phase_budget_degrees()},{"clock_segments",segments.size()},
            {"clock_rate_min",minimum},{"clock_rate_max",maximum},{"signals",signals}})<<'\n';
        if(empty)throw std::runtime_error("A channel did not decode");
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
