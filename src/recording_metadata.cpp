#include "recording_metadata.hpp"
#include <boost/json/src.hpp>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>

namespace radar {
namespace {
namespace json=boost::json;
double number(const json::value& v) {
    const double x=json::value_to<double>(v);
    if(!v.is_number() || !std::isfinite(x))throw std::invalid_argument("Metadata number must be finite");
    return x;
}
unsigned integer(const json::value& v,unsigned maximum) {
    const auto x=number(v);
    if(x<0 || x>maximum || x!=std::floor(x))throw std::invalid_argument("Invalid metadata integer");
    return unsigned(x);
}
std::string string(const json::value& v) {return std::string(v.as_string());}
}
RecordingMetadata RecordingMetadata::read(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("Cannot open recording metadata: "+path.string());
    if(std::filesystem::file_size(path)>1024*1024)throw std::invalid_argument("Recording metadata exceeds 1 MiB");
    RecordingMetadata result;
    result.json_={std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
    const auto document=json::parse(result.json_);
    const auto& root=document.as_object();
    if(integer(root.at("version"),1)!=1)throw std::invalid_argument("Unsupported recording metadata version");
    const auto& recording=root.at("recording").as_object();
    result.recording_rf_hz=number(recording.at("centre_hz"));
    result.sample_rate=integer(recording.at("sample_rate_hz"),128'000'000);
    result.sample_format=string(recording.at("sample_format"));
    if(const auto* order=recording.if_contains("iq_order");order && string(*order)!="I,Q")
        throw std::invalid_argument("Recording I/Q order must be I,Q");
    if(result.recording_rf_hz<=0 || result.sample_rate<8'000'000 ||
       (result.sample_format!="ci16_le" && result.sample_format!="cf32_le"))
        throw std::invalid_argument("Invalid recording RF, rate or sample format");
    if(const auto* acquisition=root.if_contains("capture")) {
        const auto& c=acquisition->as_object();SdrCaptureSettings s;
        s.sample_rate_hz=result.sample_rate;s.centre_hz=result.recording_rf_hz;
        auto text=[&](const char* key,std::string& out){if(const auto* v=c.if_contains(key))out=string(*v);};
        auto real=[&](const char* key,double& out){if(const auto* v=c.if_contains(key))out=number(*v);};
        text("device_args",s.device_args);text("antenna",s.antenna);text("clock_source",s.clock_source);text("time_source",s.time_source);
        real("sample_rate_hz",s.sample_rate_hz);real("centre_hz",s.centre_hz);real("gain_db",s.gain_db);
        real("bandwidth_hz",s.bandwidth_hz);real("buffer_seconds",s.buffer_seconds);
        if(const auto* v=c.if_contains("channel"))s.channel=integer(*v,63);
        if(s.sample_rate_hz<8e6 || s.sample_rate_hz>128e6 || s.centre_hz<=0 ||
           s.bandwidth_hz<=0 || s.buffer_seconds<.1 || s.buffer_seconds>8 || s.device_args.empty())
            throw std::invalid_argument("Invalid SDR capture settings");
        if(const auto* clock=c.if_contains("clock_model")){
            const auto& model=clock->as_object();
            auto real=[&](const char* key,double& out){if(const auto* v=model.if_contains(key))out=number(*v);};
            real("receiver_white_fm",s.clock.receiver.white_fm);real("receiver_random_walk_fm",s.clock.receiver.random_walk_fm);
            real("receiver_drift_diffusion",s.clock.receiver.drift_diffusion);
            real("transmitter_white_fm",s.clock.transmitter.white_fm);real("transmitter_random_walk_fm",s.clock.transmitter.random_walk_fm);
            real("settling_seconds",s.clock.settling_seconds);real("innovation_huber_sigma",s.clock.innovation_huber_sigma);
            real("unknown_sigma_samples",s.clock.unknown_sigma_samples);
        }
        const auto& q=s.clock;
        for(double value:{q.receiver.white_fm,q.receiver.random_walk_fm,q.receiver.drift_diffusion,q.transmitter.white_fm,q.transmitter.random_walk_fm})
            if(value<0)throw std::invalid_argument("Negative clock process noise");
        if(q.settling_seconds<.02 || q.settling_seconds>10 || q.innovation_huber_sigma<4 || q.innovation_huber_sigma>20 || q.unknown_sigma_samples<=0)
            throw std::invalid_argument("Invalid clock control parameters");
        result.capture=std::move(s);
    }
    std::set<std::string> sites,ids;
    for(const auto& item:root.at("transmitters").as_array()) {
        const auto& site=item.as_object();const auto id=string(site.at("id"));
        if(id.empty() || !sites.insert(id).second)throw std::invalid_argument("Empty or duplicate transmitter id");
        const auto& location=site.at("location").as_object();
        if(std::abs(number(location.at("latitude_deg")))>90 || std::abs(number(location.at("longitude_deg")))>180)
            throw std::invalid_argument("Invalid transmitter coordinates");
    }
    for(const auto& item:root.at("signals").as_array()) {
        const auto& s=item.as_object();
        RecordingSignal signal{string(s.at("id")),string(s.at("name")),string(s.at("transmitter_id")),
            {number(s.at("frequency_hz")),0,integer(s.at("guard_denominator"),32)}};
        signal.channel.nco_hz=signal.channel.rf_hz-result.recording_rf_hz;
        // An explicit NCO can represent a known translation without corrupting
        // physical RF metadata. Tracked receiver CFO never modifies either.
        if(const auto* nco=s.if_contains("nco_hz"))signal.channel.nco_hz=number(*nco);
        if(signal.id.empty() || signal.name.empty() || !ids.insert(signal.id).second || !sites.contains(signal.transmitter))
            throw std::invalid_argument("Invalid signal id, name or transmitter reference");
        if(number(s.at("bandwidth_hz"))!=7'000'000 || integer(s.at("fft_size"),8192)!=8192)
            throw std::invalid_argument("The pipeline supports 7 MHz, 8K DVB-T signals");
        result.signals.push_back(std::move(signal));
    }
    SarCapture capture;result.configure(capture);
    capture.input_rate=(result.sample_rate+7'999'999)/8'000'000*8'000'000;
    capture.decimation=capture.input_rate/8'000'000;capture.source_rate=result.sample_rate;
    capture.validate();
    return result;
}
void RecordingMetadata::validate_wave(const WaveInfo& wave) const {
    if(wave.sample_rate!=sample_rate || (wave.sample_bytes==4 ? "ci16_le" : "cf32_le")!=sample_format)
        throw std::invalid_argument("Recording metadata rate/format disagrees with WAV header");
}
void RecordingMetadata::configure(SarCapture& capture,std::string_view selected) const {
    if(!capture.channels.empty() || capture.recording_rf_hz)
        throw std::invalid_argument("Recording metadata conflicts with explicit channel/RF options");
    capture.recording_rf_hz=recording_rf_hz;
    for(const auto& signal:signals)if(selected.empty() || selected==signal.id)capture.channels.push_back(signal.channel);
    if(capture.channels.empty())throw std::invalid_argument("No metadata signal matches selection: "+std::string(selected));
}
void RecordingMetadata::write(const std::filesystem::path& path,std::string_view selected) const {
    auto document=json::parse(json_);json::array ids;
    for(const auto& signal:signals)if(selected.empty() || selected==signal.id)ids.emplace_back(signal.id);
    document.as_object()["processing_signal_ids"]=std::move(ids);
    std::ofstream file(path);file.exceptions(std::ios::badbit|std::ios::failbit);file<<json::serialize(document)<<'\n';
}
}
