#include "capture_source.hpp"
#include <chrono>
#include <thread>

namespace radar {
namespace {
class ReplaySource final:public CaptureSource {
    AppliedCapture applied_;
    WaveInfo wave_;
    std::ifstream file_;
    bool paced_,loop_,injected_=false;
    std::uint64_t position_=0,epoch_=0,gap_at_=0;
    std::size_t gap_;
    std::chrono::steady_clock::time_point start_;
public:
    ReplaySource(const std::filesystem::path& path,const SdrCaptureSettings& settings,bool paced,double gap_at,std::size_t gap,bool loop):
        paced_(paced),loop_(loop),gap_(gap) {
        WaveReader wave(path);wave_=wave.info;
        if(wave_.sample_bytes!=sizeof(Iq16))throw std::invalid_argument("Capture simulation requires a stereo int16 I/Q WAV");
        if(!wave_.samples)throw std::invalid_argument("Capture simulation cannot replay an empty WAV");
        applied_={"simulated capture", "simulation",settings.antenna,settings.clock_source,settings.time_source,
            double(wave_.sample_rate),settings.centre_hz,settings.gain_db,settings.bandwidth_hz,double(wave_.sample_rate)};
        file_.exceptions(std::ios::badbit|std::ios::failbit);file_.open(path,std::ios::binary);file_.seekg(wave_.data_offset);
        gap_at_=std::uint64_t(gap_at*wave_.sample_rate);
    }
    AppliedCapture applied() const override{return applied_;}
    void start() override{start_=std::chrono::steady_clock::now();}
    CaptureRead receive(std::span<Iq16> out) override {
        bool restart=false;
        if(position_==wave_.samples){
            if(!loop_)return {.end=true};
            epoch_+=wave_.samples;position_=0;file_.seekg(wave_.data_offset);restart=true;
        }
        if(gap_ && !injected_ && position_>=gap_at_) {
            position_+=std::min<std::uint64_t>(gap_,wave_.samples-position_);file_.seekg(wave_.data_offset+position_*sizeof(Iq16));injected_=true;
            return {.error=CaptureRead::Error::overflow};
        }
        auto count=std::size_t(std::min<std::uint64_t>(out.size(),wave_.samples-position_));
        if(gap_ && !injected_ && position_<gap_at_)count=std::min(count,std::size_t(gap_at_-position_));
        if(!count)return {.end=true};
        const auto first=epoch_+position_;file_.read(reinterpret_cast<char*>(out.data()),std::streamsize(count*sizeof(Iq16)));position_+=count;
        if(paced_)std::this_thread::sleep_until(start_+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(double(epoch_+position_)/applied_.rate_hz)));
        return {.count=count,.tick=std::int64_t(first),.restart=restart};
    }
    void stop() override{}
};
}
std::unique_ptr<CaptureSource> make_replay_source(const std::filesystem::path& path,const SdrCaptureSettings& settings,bool paced,double gap_seconds,std::size_t gap,bool loop){
    return std::make_unique<ReplaySource>(path,settings,paced,gap_seconds,gap,loop);
}
}
