#include "capture_source.hpp"
#include <dm_dev.h>
#include <dm_stream.h>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <system_error>

namespace radar {
namespace {
void checked(int result,const char* operation) {
    if(result)throw std::system_error(-result,std::generic_category(),operation);
}
class UsdrSource final:public CaptureSource {
    pdm_dev_t device_=nullptr;
    pusdr_dms_t stream_=nullptr;
    AppliedCapture applied_;
    usdr_dms_rx_block_t block_{};
    std::size_t offset_=0;
    bool loan_=false,started_=false;
    void release() {
        if(!loan_)return;
        checked(usdr_dms_release(stream_,&block_),"uSDR release DMA block");
        loan_=false;block_={};offset_=0;
    }
    void close() noexcept {
        // Retry a failed release/stop during unwinding before closing the fd.
        try{stop();}catch(...){}
        if(stream_){usdr_dms_destroy(stream_);stream_=nullptr;}
        if(device_){usdr_dmd_close(device_);device_=nullptr;}
    }
    void set(const char* path,std::uint64_t value){checked(usdr_dme_set_uint(device_,path,value),path);}
    double get(const char* path) {
        std::uint64_t value=0;checked(usdr_dme_get_uint(device_,path,&value),path);return double(value);
    }
public:
    explicit UsdrSource(const SdrCaptureSettings& s) {
        static_assert(std::endian::native==std::endian::little);
        if(s.channel || s.clock_source!="internal" || s.time_source!="internal")
            throw std::invalid_argument("uSDR capture supports channel 0 and the internal clock/time source");
        if(s.device_args!="bus=pcie" && !s.device_args.starts_with("bus=pcie,") && !s.device_args.starts_with("/dev/usdr"))
            throw std::invalid_argument("uSDR capture requires bus=pcie[,dev=usdrN] or /dev/usdrN");
        for(double v:{s.sample_rate_hz,s.centre_hz,s.bandwidth_hz,s.gain_db})
            if(!std::isfinite(v) || v<0 || v>std::numeric_limits<unsigned>::max())
                throw std::invalid_argument("Invalid uSDR setting");
        if(s.sample_rate_hz<8e6 || s.sample_rate_hz>65e6 || !s.centre_hz || !s.bandwidth_hz || s.gain_db>85)
            throw std::invalid_argument("uSDR capture rate must be 8–65 MS/s and relative gain 0–85 dB");
        try {
            checked(usdr_dmd_create_string(s.device_args.c_str(),&device_),"open PCIe uSDR");
            set("/dm/power/en",1);
            set("/dm/rate/master",std::llround(s.sample_rate_hz));
            checked(usdr_dme_set_string(device_,"/dm/sdr/0/rx/path",s.antenna.c_str()),"uSDR input path");
            set("/dm/sdr/0/rx/frequency",std::llround(s.centre_hz));
            set("/dm/sdr/0/rx/bandwidth",std::llround(s.bandwidth_hz));
            set("/dm/sdr/0/rx/gain/auto",std::llround(s.gain_db));
            // Stream creation performs the quiet-input analog DC calibration.
            checked(usdr_dms_create(device_,"/ll/srx/0","ci16",1,16384,&stream_),"create uSDR RX stream");
            const char* antenna=nullptr;
            checked(usdr_dme_get_string(device_,"/dm/sdr/0/rx/path",&antenna),"read uSDR input path");
            applied_={"uSDR PCIe ("+s.device_args+")","",antenna,"internal","internal",
                get("/dm/rate/master"),get("/dm/sdr/0/rx/frequency"),get("/dm/sdr/0/rx/gain/auto"),
                get("/dm/sdr/0/rx/bandwidth"),get("/dm/rate/adc")};
            applied_.driver="usdr";
            if(applied_.rate_hz<8e6 || applied_.rate_hz>65e6)
                throw std::runtime_error("Applied uSDR rate is outside 8–65 MS/s");
        }catch(...){close();throw;}
    }
    ~UsdrSource() override{close();}
    AppliedCapture applied() const override{return applied_;}
    void start() override {
        if(started_)return;
        checked(usdr_dms_sync(device_,"off",1,&stream_),"hold uSDR RX synchronization");
        started_=true; // Ensure cleanup if either start step fails.
        checked(usdr_dms_op(stream_,USDR_DMS_START,0),"start uSDR RX DMA");
        checked(usdr_dms_sync(device_,"none",1,&stream_),"release uSDR RX synchronization");
    }
    CaptureRead receive(std::span<Iq16> out) override {
        if(!started_)return {.error=CaptureRead::Error::fatal,.message="uSDR stream has not started"};
        if(out.empty())return {};
        if(!loan_){
            const int result=usdr_dms_acquire(stream_,250,&block_);
            if(result==-ETIMEDOUT || result==-EAGAIN)return {.error=CaptureRead::Error::timeout};
            if(result)return {.error=CaptureRead::Error::fatal,.message="uSDR DMA acquire: "+std::error_code(-result,std::generic_category()).message()};
            loan_=true;offset_=0;
            if(!block_.data || !block_.samples || block_.format!=USDR_DMS_NATIVE_CI16_LE ||
               block_.bytes!=std::size_t(block_.samples)*sizeof(Iq16) ||
               block_.timestamp>std::uint64_t(INT64_MAX)-block_.samples)
                return {.error=CaptureRead::Error::fatal,.message="Invalid uSDR DMA block layout or timestamp"};
            // The timeline consumes error events before the following data. Keep
            // this loan until the next receive so the gap is counted only once.
            if(block_.lost_samples || (block_.flags & USDR_DMS_BLOCK_DISCONTINUITY))
                return {.error=CaptureRead::Error::overflow};
        }
        CaptureRead result{.count=std::min(out.size(),std::size_t(block_.samples)-offset_),
                           .tick=std::int64_t(block_.timestamp+offset_)};
        // The destination is the Vulkan mapped CI16 ring. No staging buffer,
        // CPU float conversion, or second sample copy is involved.
        std::memcpy(out.data(),static_cast<const std::byte*>(block_.data)+offset_*sizeof(Iq16),result.count*sizeof(Iq16));
        offset_+=result.count;
        if(offset_==block_.samples){
            try{release();}catch(const std::exception& e){return {.error=CaptureRead::Error::fatal,.message=e.what()};}
        }
        return result;
    }
    void stop() override {
        release();
        if(started_){checked(usdr_dms_op(stream_,USDR_DMS_STOP,0),"stop uSDR RX DMA");started_=false;}
    }
};
}
std::unique_ptr<CaptureSource> make_usdr_source(const SdrCaptureSettings& s){return std::make_unique<UsdrSource>(s);}
}
