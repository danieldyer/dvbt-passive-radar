// Saved complex range histories -> production Vulkan focus ->
// compact GPU previews. No demodulation, clock estimation or data repair here.
#include "gpu_commands.hpp"
#include "gpu_focus.hpp"
#include "gpu_preview.hpp"
#include "matched_detector.hpp"
#include "profile_history.hpp"
#include <bit>
#include <boost/json.hpp>
#include <charconv>
#include <fstream>
#include <iostream>

namespace {
using namespace radar;
template <class T> T read(std::istream &source) {
    T value{};
    source.read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}
double number(std::string_view text) {
    double value;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid numeric option");
    return value;
}
boost::json::object json(std::istream &source, std::size_t length) {
    if (!length || length > 1024 * 1024)
        throw std::runtime_error("Invalid live metadata length");
    std::string text(length, '\0');
    source.read(text.data(), std::streamsize(length));
    return boost::json::parse(text).as_object();
}
struct Channel {
    video::ProfileHistory history;
    GpuFocus focus;
    video::GpuPreview preview;
    double interval, rf, last_time = -1;
    std::uint64_t sequence = 0;
    ComputeTimings history_timing, focus_timing;
    Channel(std::shared_ptr<gpu::Context> context, unsigned guard, double frequency, double integration,
            double update, bool erasures)
        : history(context, integration, (8192 + 8192.0 / guard) / 8e6, update, erasures), focus(context),
          preview(std::move(context)), interval((8192 + 8192.0 / guard) / 8e6), rf(frequency) {
        focus_timing.detailed = true;
    }
};
void publish(DopplerMap &map, video::GpuPreview &preview, const gpu::BufferView &values) {
    preview.calculate(map, values);
    map.noise_range.clear();
    map.noise_doppler.clear();
    auto meta = boost::json::parse(map.json()).as_object();
    meta["display_shape"] = boost::json::array{preview.height, preview.width};
    meta["preview_format"] = "power_f32le";
    meta["preview_order"] = "doppler_down,range";
    meta["source_kind"] = "live_profiles";
    meta["effective_integration_s"] = 1 / map.enbw;
    const auto text = boost::json::serialize(meta);
    const auto length = std::uint32_t(text.size());
    const auto bytes = std::uint64_t(preview.power.size() * sizeof(float));
    std::cout.write("RDP1", 4);
    std::cout.write(reinterpret_cast<const char *>(&length), 4);
    std::cout.write(reinterpret_cast<const char *>(&bytes), 8);
    std::cout.write(text.data(), std::streamsize(text.size()));
    std::cout.write(reinterpret_cast<const char *>(preview.power.data()), std::streamsize(bytes));
    if (!std::cout)
        throw std::runtime_error("Live preview output failed");
}
} // namespace
int main(int argc, char **argv) {
    try {
        static_assert(std::endian::native == std::endian::little && sizeof(radar::C) == 8);
        if (argc < 2)
            throw std::invalid_argument(
                "Usage: live_profiles_preview CAPTURE.dvblive --output NEW_DIRECTORY [--integration-seconds "
                "3 --update-seconds 0.03264 --doppler-hz 100 --range-metres 30000 --start 0 --seconds 0]");
        const std::filesystem::path input = argv[1];
        std::filesystem::path output;
        double integration = 3, update = .03264, doppler = 100, range = 30000, start = 0, seconds = 0;
        bool remove_dc = true;
        for (int i = 2; i < argc; ++i) {
            const std::string_view key = argv[i];
            if (key == "--no-remove-dc") {
                remove_dc = false;
                continue;
            }
            if (i + 1 == argc)
                throw std::invalid_argument("Missing option value");
            const std::string_view value = argv[++i];
            if (key == "--output")
                output = value;
            else if (key == "--integration-seconds")
                integration = number(value);
            else if (key == "--update-seconds")
                update = number(value);
            else if (key == "--doppler-hz")
                doppler = number(value);
            else if (key == "--range-metres")
                range = number(value);
            else if (key == "--start")
                start = number(value);
            else if (key == "--seconds")
                seconds = number(value);
            else
                throw std::invalid_argument("Unknown option: " + std::string(key));
        }
        if (output.empty() || start < 0 || seconds < 0 || doppler < 4 / integration || doppler > 200 ||
            range < 1000 || range > matched_max_range_metres)
            throw std::invalid_argument("Invalid archived detection options");
        std::ifstream source(input, std::ios::binary);
        source.exceptions(std::ios::badbit | std::ios::failbit);
        const auto total = std::filesystem::file_size(input);
        char magic[8];
        source.read(magic, 8);
        if (std::string_view(magic, 8) != "DVBTLIV1")
            throw std::runtime_error("Invalid live archive magic");
        const auto length = read<std::uint32_t>(source);
        const auto header = json(source, length);
        if (header.at("archive_mode") != "profiles" && header.at("archive_mode") != "both")
            throw std::runtime_error("Video replay needs saved complex profiles");
        const auto *policy = header.if_contains("profile_row_policy");
        const bool erasures = policy && *policy == "fixed_grid_zero_information_erasures";
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("Output directory already exists");
        auto context = std::make_shared<gpu::Context>();
        std::vector<std::unique_ptr<Channel>> channels;
        for (const auto &value : header.at("applied_channels").as_array()) {
            const auto &c = value.as_object();
            const auto guard = boost::json::value_to<unsigned>(c.at("guard"));
            if (guard != 4 && guard != 8 && guard != 16 && guard != 32)
                throw std::runtime_error("Invalid archived guard");
            channels.push_back(std::make_unique<Channel>(
                context, guard, boost::json::value_to<double>(c.at("rf_hz")), integration, update, erasures));
        }
        if (channels.empty() || channels.size() > 64)
            throw std::runtime_error("Invalid archived channel count");
        std::vector<double> information;
        double duration = 0, progress = -10;
        for (;;) {
            char tag[4];
            source.read(tag, 4);
            const auto size = read<std::uint32_t>(source);
            const auto bytes = read<std::uint64_t>(source);
            const auto meta = json(source, size);
            const auto payload = std::uint64_t(source.tellg());
            if (bytes > 64 * 1024 * 1024 || payload > total || bytes > total - payload)
                throw std::runtime_error("Invalid/truncated archive payload");
            if (std::string_view(tag, 4) == "PROF") {
                const auto channel = boost::json::value_to<std::size_t>(meta.at("channel"));
                auto &c = *channels.at(channel);
                const auto count = boost::json::value_to<std::size_t>(meta.at("count"));
                const auto segment = boost::json::value_to<std::uint64_t>(meta.at("segment"));
                const auto profile_size = boost::json::value_to<std::size_t>(meta.at("size"));
                const auto info = boost::json::value_to<double>(meta.at("information")),
                           origin = boost::json::value_to<double>(meta.at("reference_origin"));
                const auto &rows = meta.at("rows").as_array();
                const auto tail = 8ULL * (boost::json::value_to<std::size_t>(meta.at("stationary_carriers")) +
                                          boost::json::value_to<std::size_t>(meta.at("response_carriers"))) +
                                  4ULL * boost::json::value_to<std::size_t>(meta.at("noise_carriers")) +
                                  (meta.contains("cancellation_variance_carriers")
                                       ? 8ULL * boost::json::value_to<std::size_t>(
                                                    meta.at("cancellation_variance_carriers"))
                                       : 0);
                if (!count || count > 1020 || rows.size() != count || meta.at("moments") != 2 ||
                    !matched_profile_size_supported(profile_size) || meta.at("order") != "moment,row,range" ||
                    bytes != 2 * count * profile_size * sizeof(C) + tail)
                    throw std::runtime_error("Invalid archived profile layout");
                if (range > 30000 && profile_size == matched_profile_size)
                    throw std::runtime_error(
                        "Requested range exceeds this archive's original 30 km capture extent");
                std::vector<C> values(2 * count * profile_size);
                source.read(reinterpret_cast<char *>(values.data()),
                            std::streamsize(values.size() * sizeof(C)));
                if (std::ranges::any_of(
                        values, [](C z) { return !std::isfinite(z.real()) || !std::isfinite(z.imag()); }))
                    throw std::runtime_error("Nonfinite archived profile");
                c.history.stage(values, count, unsigned(profile_size), &c.history_timing);
                for (std::size_t n = 0; n < count; ++n) {
                    const auto &row = rows[n].as_array();
                    if (row.size() != 6 && row.size() != 8)
                        throw std::runtime_error("Invalid archived row coordinates");
                    const auto time = boost::json::value_to<double>(row[1]);
                    if (time <= c.last_time)
                        throw std::runtime_error("Reordered archived rows");
                    c.last_time = time;
                    duration = std::max(duration, time + c.interval);
                    const auto ready = c.history.append(time, info, row[5].as_bool(), segment);
                    if (!ready || time < start || (seconds > 0 && time > start + seconds))
                        continue;
                    const auto moments = c.history.snapshot(information, &c.history_timing);
                    auto result = c.focus.calculate_spectral_device(
                        moments, c.history.required(), c.interval, c.rf, c.history.first_time(),
                        {range, doppler, remove_dc}, &c.focus_timing, information);
                    auto &map = result.map;
                    map.channel = unsigned(channel);
                    map.sequence = ++c.sequence;
                    map.segment = segment;
                    map.requested_integration = integration;
                    map.reference_origin_samples = origin;
                    const auto frozen =
                        boost::json::value_to<std::size_t>(meta.at("stationary_carriers")) != 0;
                    map.stationary_prediction = frozen ? "archived_frozen_source_divided_transfer" : "none";
                    map.dc_method = frozen ? "frozen_stationary_prediction_and_constant_projection"
                                           : "constant_projection";
                    publish(map, c.preview, result.values);
                }
                if (c.last_time >= progress + 10) {
                    progress = c.last_time;
                    std::cerr << "Archived time " << progress << " s; maps";
                    for (auto &ch : channels)
                        std::cerr << ' ' << ch->sequence;
                    std::cerr << '\n';
                }
            }
            source.seekg(std::streamoff(payload + bytes));
            if (std::string_view(tag, 4) == "DONE") {
                if (std::uint64_t(source.tellg()) != total)
                    throw std::runtime_error("Unexpected bytes after archive completion");
                break;
            }
        }
        // Same duration schema as the WAV video tools, on the corrected sample grid.
        const auto rate = boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
        boost::json::object metadata{{"source_kind", "live_profiles"},
                                     {"input_samples", std::uint64_t(std::llround(duration * rate))},
                                     {"input_sample_rate", rate}};
        boost::json::array counts;
        for (auto &c : channels)
            counts.emplace_back(c->sequence);
        metadata["maps"] = std::move(counts);
        boost::json::array times;
        for (auto &c : channels)
            times.emplace_back(boost::json::object{
                {"maps", c->sequence},
                {"history_gpu_seconds", c->history_timing.gpu},
                {"focus_gpu_seconds", c->focus_timing.gpu},
                {"preview_gpu_seconds", c->preview.timing.gpu},
                {"focus_stages_seconds", boost::json::value_from(c->focus_timing.stages)}});
        metadata["gpu_timings"] = std::move(times);
        std::ofstream file(output / "metadata.json");
        file.exceptions(std::ios::badbit | std::ios::failbit);
        file << boost::json::serialize(metadata) << '\n';
        std::cout.flush();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Live profile replay error: " << e.what() << '\n';
        return 1;
    }
}
