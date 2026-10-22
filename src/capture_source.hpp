#pragma once
#include "live_input.hpp"
#include "recording_metadata.hpp"
#include "capture_time.hpp"
#include <memory>

namespace radar {
struct AppliedCapture {
    std::string device,serial,antenna,clock_source,time_source;
    double rate_hz{},centre_hz{},gain_db{},bandwidth_hz{},master_clock_hz{};
    std::string driver="uhd";
};
struct CaptureRead {
    enum class Error {none,timeout,overflow,fatal};
    std::size_t count{};
    std::optional<std::int64_t> tick;
    Error error=Error::none;
    bool sequence=false,end=false,restart=false,looped=false;
    std::string message;
};
class CaptureSource {
public:
    virtual ~CaptureSource()=default;
    virtual AppliedCapture applied() const=0;
    virtual void start()=0;
    virtual CaptureRead receive(std::span<Iq16>)=0;
    virtual void stop()=0;
    virtual std::optional<DeviceTimeAnchor> time_anchor(){return {};}
};
std::unique_ptr<CaptureSource> make_capture_source(const SdrCaptureSettings&);
std::unique_ptr<CaptureSource> make_usdr_source(const SdrCaptureSettings&);
std::unique_ptr<CaptureSource> make_uhd_source(const SdrCaptureSettings&);
// File replay is only a capture-transport test source. It shares the complete
// live ring/channelizer/decoder path and can inject timestamped sample losses.
std::unique_ptr<CaptureSource> make_replay_source(const std::filesystem::path&,const SdrCaptureSettings&,
                                               bool paced,double gap_seconds=0,std::size_t gap_samples=0,bool loop=true,bool cyclic=false);
}
