#include "capture_source.hpp"
#include <cmath>
#include <uhd/types/metadata.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>

namespace radar {
namespace {
class UhdSource final : public CaptureSource {
    uhd::usrp::multi_usrp::sptr device_;
    uhd::rx_streamer::sptr stream_;
    AppliedCapture applied_;
    bool started_ = false;

  public:
    explicit UhdSource(const SdrCaptureSettings &s) : device_(uhd::usrp::multi_usrp::make(s.device_args)) {
        if (s.channel >= device_->get_rx_num_channels())
            throw std::invalid_argument("SDR RX channel does not exist");
        device_->set_clock_source(s.clock_source);
        device_->set_time_source(s.time_source);
        device_->set_rx_rate(s.sample_rate_hz, s.channel);
        device_->set_rx_antenna(s.antenna, s.channel);
        device_->set_rx_gain(s.gain_db, s.channel);
        device_->set_rx_freq(uhd::tune_request_t(s.centre_hz), s.channel);
        device_->set_rx_bandwidth(s.bandwidth_hz, s.channel);
        const auto info = device_->get_usrp_rx_info(s.channel);
        applied_ = {device_->get_pp_string(),
                    info.has_key("mboard_serial") ? info["mboard_serial"] : "",
                    device_->get_rx_antenna(s.channel),
                    device_->get_clock_source(0),
                    device_->get_time_source(0),
                    device_->get_rx_rate(s.channel),
                    device_->get_rx_freq(s.channel),
                    device_->get_rx_gain(s.channel),
                    device_->get_rx_bandwidth(s.channel),
                    device_->get_master_clock_rate()};
        if (!std::isfinite(applied_.rate_hz) || applied_.rate_hz < 8e6 || applied_.rate_hz > 128e6)
            throw std::runtime_error("Applied SDR sample rate is outside the pipeline's supported range");
        uhd::stream_args_t args("sc16", "sc16");
        args.channels = {s.channel};
        stream_ = device_->get_rx_stream(args);
    }
    ~UhdSource() override {
        try {
            stop();
        } catch (...) {
        }
    }
    AppliedCapture applied() const override {
        return applied_;
    }
    std::optional<DeviceTimeAnchor> time_anchor() override {
        auto best = measure_device_time([&] { return device_->get_time_now().get_real_secs(); });
        for (unsigned i = 1; i < 8; ++i) {
            const auto next = measure_device_time([&] { return device_->get_time_now().get_real_secs(); });
            if (next.query_uncertainty_ns < best.query_uncertainty_ns)
                best = next;
        }
        return best;
    }
    void start() override {
        uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
        cmd.stream_now = false;
        cmd.time_spec = device_->get_time_now() + uhd::time_spec_t(.1);
        stream_->issue_stream_cmd(cmd);
        started_ = true;
    }
    CaptureRead receive(std::span<Iq16> output) override {
        uhd::rx_metadata_t md;
        CaptureRead result;
        result.count = stream_->recv(output.data(), output.size(), md, .25);
        if (md.has_time_spec)
            result.tick = md.time_spec.to_ticks(applied_.rate_hz);
        result.sequence = md.out_of_sequence;
        switch (md.error_code) {
        case uhd::rx_metadata_t::ERROR_CODE_NONE:
            break;
        case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT:
            result.error = CaptureRead::Error::timeout;
            break;
        case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW:
            result.error = CaptureRead::Error::overflow;
            break;
        default:
            result.error = CaptureRead::Error::fatal;
            result.message = md.strerror();
            break;
        }
        return result;
    }
    void stop() override {
        if (started_) {
            stream_->issue_stream_cmd(uhd::stream_cmd_t(uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS));
            started_ = false;
        }
    }
};
} // namespace
std::unique_ptr<CaptureSource> make_uhd_source(const SdrCaptureSettings &settings) {
    return std::make_unique<UhdSource>(settings);
}
} // namespace radar
