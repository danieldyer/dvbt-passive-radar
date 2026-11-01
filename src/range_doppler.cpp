#include "range_doppler.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <sstream>
#include <stdexcept>

namespace radar {
std::string DopplerMap::json() const {
    std::ostringstream s;
    s << std::setprecision(17);
    s << "{\"version\":1,\"sequence\":" << sequence << ",\"segment\":" << segment << ",\"ranges\":" << ranges
      << ",\"frequencies\":" << frequencies << ",\"samples\":" << samples << ",\"fft_size\":" << fft_size
      << ",\"dc_removed\":" << (dc_removed ? "true" : "false") << ",\"dc_removal\":\""
      << (dc_removed ? dc_method : "none") << '"' << ",\"range_start_m\":0,\"range_step_m\":" << range_step
      << ",\"frequency_start_hz\":" << frequency_start << ",\"frequency_step_hz\":" << frequency_step
      << ",\"time_start_s\":" << time_start << ",\"time_end_s\":" << time_end
      << ",\"integration_s\":" << integration << ",\"coherence\":" << coherence
      << ",\"history_limit_s\":" << history_limit << ",\"requested_integration_s\":" << requested_integration
      << ",\"window\":\"" << window << "\",\"enbw_hz\":" << (enbw > 0 ? enbw : 1.5 / integration)
      << ",\"motion_model\":\"" << motion_model << "\",\"phase_epoch_s\":" << phase_epoch
      << ",\"physical_rf_hz\":" << rf_hz << ",\"reference_origin_samples\":" << reference_origin_samples
      << ",\"channel\":" << channel << ",\"stationary_prediction\":\"" << stationary_prediction
      << "\",\"background_start_s\":" << background_start << ",\"background_end_s\":" << background_end
      << ",\"format\":\"cf32le\",\"order\":\"range,doppler\",\"time_reference\":\"recording_relative\""
      << ",\"amplitude_units\":\"" << amplitude_units << '"';
    auto array = [&](std::string_view name, const std::vector<float> &values) {
        if (values.empty())
            return;
        s << ",\"" << name << "\":[";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i)
                s << ',';
            s << values[i];
        }
        s << ']';
    };
    array("noise_range_factor", noise_range);
    array("noise_doppler_power", noise_doppler);
    s << '}';
    return s.str();
}
void DopplerMap::write(std::ostream &out) const {
    static_assert(std::endian::native == std::endian::little);
    const auto header = json();
    const std::uint32_t length = static_cast<std::uint32_t>(header.size());
    const std::uint64_t bytes = values.size() * sizeof(C);
    out.write("RDM1", 4);
    out.write(reinterpret_cast<const char *>(&length), 4);
    out.write(reinterpret_cast<const char *>(&bytes), 8);
    out.write(header.data(), length);
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(bytes));
    if (!out)
        throw std::runtime_error("Range-Doppler stream write failed");
}
void DopplerMap::write(const std::filesystem::path &path) const {
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    write(out);
    out.close();
    if (!out)
        throw std::runtime_error("Range-Doppler output write failed");
    std::filesystem::rename(temporary, path);
}
} // namespace radar
