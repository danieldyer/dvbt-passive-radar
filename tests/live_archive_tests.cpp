#include "live_archive.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

using namespace radar;
namespace {
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class T> T read(std::istream &file) {
    T value{};
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    check(bool(file), "Truncated archive");
    return value;
}
auto parse(std::string_view text) {
    return boost::json::parse(text, {},
                              boost::json::parse_options{.numbers = boost::json::number_precision::precise});
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Metadata path required");
        const auto metadata = RecordingMetadata::read(argv[1]);
        check(metadata.capture.has_value(), "Capture settings missing");
        SarCapture capture;
        metadata.configure(capture);
        capture.input_rate = 64'000'000;
        capture.decimation = 8;
        capture.source_rate = 61'440'000;
        capture.validate();
        AppliedCapture applied;
        applied.driver = "usdr";
        applied.rate_hz = 61'440'000.047871;
        applied.centre_hz = 202e6;
        applied.gain_db = 29.75;
        applied.bandwidth_hz = 55'999'999.875;
        applied.master_clock_hz = 61'440'000.125;
        const auto path =
            std::filesystem::temp_directory_path() /
            ("dvbt-live-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Remove {
            std::filesystem::path path;
            ~Remove() {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        } remove{path};
        constexpr std::size_t count = 37, stride = 43;
        std::vector<C> samples(capture.channels.size() * stride);
        for (std::size_t i = 0; i < samples.size(); ++i)
            samples[i] = {float(i) * .001F, -float(i) * .002F};
        const ChannelBatch batch{{{}, {}, samples.size() * sizeof(C), samples.data()},
                                 1ULL << 35,
                                 count,
                                 capture.channels.size(),
                                 stride};
        const auto check_channels = [&](std::istream &input, const boost::json::value &info,
                                        std::uint64_t bytes) {
            check(boost::json::value_to<std::uint64_t>(info.at("first")) == batch.first,
                  "Lost large channel epoch");
            check(bytes == count * capture.channels.size() * sizeof(C), "Stored stride padding as waveform");
            for (std::size_t c = 0; c < capture.channels.size(); ++c)
                for (std::size_t i = 0; i < count; ++i)
                    check(read<C>(input) == samples[c * stride + i], "Archive changed complex phase");
        };
        const auto anchor =
            device_time_anchor(200.125, 1'700'000'000'000'000'000, 1'700'000'000'000'004'000, 1000, 5000);
        check(anchor.host_monotonic_ns == 3000 && anchor.query_uncertainty_ns == 2001,
              "Incorrect UTC query bracket");
        const auto step = device_time_anchor(200, 0, 2'000'004'000, 1000, 5000);
        check(step.query_uncertainty_ns > 1'000'000'000, "Host clock step was hidden from UTC uncertainty");
        LiveArchive writer(path, metadata, applied, capture, "channels");
        writer.epoch(123456789012, applied.rate_hz);
        writer.time_anchor(anchor);
        writer.channels(batch, std::array<ClockSegment, 1>{{}});
        writer.gap({0, 998877, 23, 5});
        writer.finish();
        writer.finish();
        check(writer.queued_bytes() == 0 && writer.bytes() == std::filesystem::file_size(path),
              "Archive queue did not drain");
        bool rejected = false;
        try {
            LiveArchive duplicate(path, metadata, applied, capture, "channels");
        } catch (...) {
            rejected = true;
        }
        check(rejected, "Overwrote archive");
        std::ifstream file(path, std::ios::binary);
        char magic[8];
        file.read(magic, 8);
        check(std::memcmp(magic, "DVBTLIV1", 8) == 0, "Wrong archive header");
        const auto size = read<std::uint32_t>(file);
        std::string text(size, '\0');
        file.read(text.data(), size);
        const auto header = parse(text);
        check(!header.at("profile_stationary_subtraction").as_bool() &&
                  header.at("profile_training_rows_retained").as_bool(),
              "Archive did not declare the preserved stationary/training information");
        check(header.at("applied_capture").at("rate_hz").as_double() == applied.rate_hz,
              "Lost fractional SDR rate");
        check(header.at("applied_capture").at("driver").as_string() == applied.driver,
              "Lost archive capture driver");
        check(header.at("applied_capture").at("gain_db").as_double() == applied.gain_db &&
                  header.at("applied_capture").at("bandwidth_hz").as_double() == applied.bandwidth_hz &&
                  header.at("applied_capture").at("master_clock_hz").as_double() == applied.master_clock_hz,
              "Lost applied SDR settings");
        bool checked = false, done = false, checked_anchor = false;
        for (;;) {
            char tag[4];
            file.read(tag, 4);
            check(bool(file), "Missing final archive record");
            const auto length = read<std::uint32_t>(file);
            const auto bytes = read<std::uint64_t>(file);
            text.resize(length);
            file.read(text.data(), length);
            const auto info = parse(text);
            if (std::memcmp(tag, "UTCA", 4) == 0) {
                check(info.at("host_utc_ns").as_int64() == anchor.host_utc_ns &&
                          info.at("device_seconds").as_double() == anchor.device_seconds,
                      "UTC/device correspondence lost precision");
                checked_anchor = true;
            } else if (std::memcmp(tag, "CHIQ", 4) == 0) {
                check_channels(file, info, bytes);
                checked = true;
            } else
                file.seekg(std::streamoff(bytes), std::ios::cur);
            if (std::memcmp(tag, "DONE", 4) == 0) {
                done = true;
                break;
            }
        }
        check(checked && done && checked_anchor, "Archive has no waveform, UTC correspondence or completion");
        for (unsigned profile_size : {matched_compact_profile_size, matched_extended_profile_size})
            for (int missing_carrier : {-1, 0, 6817 + 3408, 3 * 6817 - 1}) {
                const auto profile_path = path.string() + ".both";
                Remove remove_profiles{profile_path};
                FrameBatch frame;
                frame.channel = 0;
                frame.segment = 4;
                frame.first = 1;
                frame.last = 3;
                frame.symbols.resize(3);
                frame.frames.resize(1);
                frame.carriers.count = 3;
                for (unsigned i = 0; i < 3; ++i) {
                    auto &symbol = frame.symbols[i];
                    symbol.channel = 0;
                    symbol.segment = 4;
                    symbol.modulo4 = i;
                    symbol.window.sample = (1ULL << 31) + i * 8704;
                    symbol.window.phase = .35 + i * .11;
                    symbol.window.frequency_hz = 43.125;
                }
                std::vector<C> measured(3 * 6817, C{1, 2}), reference = measured, response(6817, C{.4F, .3F});
                std::vector<std::uint32_t> flags(3 * 6817, 7);
                if (missing_carrier >= 0)
                    flags[missing_carrier] = 0;
                auto view = [](auto &values) {
                    return gpu::BufferView{{}, {}, values.size() * sizeof(values[0]), values.data()};
                };
                frame.carriers.measured = view(measured);
                SpectralBatch spectra{frame,
                                      {},
                                      {view(reference), view(flags), 3},
                                      {view(response), {}, 3, 128.375, .001},
                                      {{}, {}, {}, {}, {}, 3, 128.375}};
                SarBatch sar(capture, spectra, {{}});
                MatchedProfiles profiles;
                profiles.first = 1;
                profiles.count = 2;
                profiles.information = 12.75;
                profiles.size = profile_size;
                profiles.values.resize(2 * profiles.count * profiles.size);
                for (std::size_t i = 0; i < profiles.values.size(); ++i)
                    profiles.values[i] = {float(i) * .001F, -float(i) * .003F};
                std::vector<float> variance(6817);
                frame.symbols[1].source_offset_samples = .375;
                frame.symbols[1].source_rate_error = .04e-6;
                for (std::size_t i = 0; i < variance.size(); ++i)
                    variance[i] = .125F + float(i) * .01F;
                LiveArchive profile_writer(profile_path, metadata, applied, capture, "both");
                profile_writer.channels(batch, sar.clock);
                profile_writer.profiles(sar, profiles, variance);
                profile_writer.channels(batch, sar.clock);
                profile_writer.finish();
                check(profile_writer.queued_bytes() == 0 &&
                          profile_writer.bytes() == std::filesystem::file_size(profile_path),
                      "Combined archive queue did not drain");
                std::ifstream profile_file(profile_path, std::ios::binary);
                profile_file.seekg(8);
                const auto header_length = read<std::uint32_t>(profile_file);
                text.resize(header_length);
                profile_file.read(text.data(), header_length);
                check(parse(text).at("archive_mode") == "both", "Lost combined archive mode");
                checked = false;
                unsigned channel_records = 0, clock_records = 0;
                for (;;) {
                    char tag[4];
                    profile_file.read(tag, 4);
                    check(bool(profile_file), "Truncated profile archive");
                    const auto length = read<std::uint32_t>(profile_file);
                    const auto bytes = read<std::uint64_t>(profile_file);
                    text.resize(length);
                    profile_file.read(text.data(), length);
                    const auto info = parse(text);
                    if (std::memcmp(tag, "PROF", 4) == 0) {
                        const auto &row = info.at("rows").as_array()[0].as_array();
                        const auto &archived_rows = info.at("rows").as_array();
                        check(archived_rows.size() == profiles.count, "Archive changed profile row count");
                        for (unsigned i = 0; i < profiles.count; ++i) {
                            const auto first_carrier = i * 6817, last_carrier = (i + 2) * 6817;
                            const bool expected = missing_carrier < 0 ||
                                                  unsigned(missing_carrier) < first_carrier ||
                                                  unsigned(missing_carrier) >= last_carrier;
                            check(archived_rows[i].as_array()[5].as_bool() == expected,
                                  "Profile validity must include both adjacent rows and every carrier");
                        }
                        check(info.at("size") == profile_size &&
                                  info.at("positive_context_samples") ==
                                      matched_context_for_size(profile_size) &&
                                  info.at("negative_context_samples") == 64,
                              "Extended archive lost its finite delay support");
                        check(row[3].as_double() == frame.symbols[1].window.phase &&
                                  row[4].as_double() == 43.125,
                              "Profile archive lost pilot phase or CFO");
                        check(row[6].as_double() == .375 && row[7].as_double() == .04e-6,
                              "Profile archive lost independent source timing");
                        check(row[1].as_double() == sar.seconds(1) &&
                                  row[2].as_double() == sar.original_adc(1),
                              "Profile archive lost the common ADC coordinates");
                        check(info.at("information").as_double() == 12.75 &&
                                  bytes == (profiles.values.size() + response.size()) * sizeof(C) +
                                               variance.size() * sizeof(float),
                              "Profile archive lost weights or moments");
                        for (auto value : profiles.values)
                            check(read<C>(profile_file) == value, "Profile archive changed complex phase");
                        check(info.at("stationary_carriers") == 0 &&
                                  info.at("cancellation_variance_carriers") == 0,
                              "Production archive contains obsolete stationary estimation fields");
                        checked = true;
                        for (auto value : response)
                            check(read<C>(profile_file) == value, "Profile archive lost the response phase");
                        for (auto value : variance)
                            check(read<float>(profile_file) == value,
                                  "Profile archive lost carrier noise variance");
                    } else if (std::memcmp(tag, "CHIQ", 4) == 0) {
                        check_channels(profile_file, info, bytes);
                        ++channel_records;
                    } else if (std::memcmp(tag, "CLCK", 4) == 0) {
                        const auto &segments = info.at("segments").as_array();
                        check(!bytes && segments.size() == 1, "Invalid shared clock record");
                        const auto &clock = segments.front().as_array();
                        check(boost::json::value_to<std::uint64_t>(clock[0]) == sar.clock[0].start &&
                                  clock[1].as_double() == sar.clock[0].offset &&
                                  clock[2].as_double() == sar.clock[0].rate,
                              "Combined archive changed clock coordinates");
                        ++clock_records;
                    } else
                        profile_file.seekg(std::streamoff(bytes), std::ios::cur);
                    if (std::memcmp(tag, "DONE", 4) == 0)
                        break;
                }
                check(checked && channel_records == 2 && clock_records == 1,
                      "Missing interleaved data or duplicate common clock");
                check(profile_file.peek() == std::char_traits<char>::eof(),
                      "Records written after archive completion");
            }
        const auto reuse_path = path.string() + ".reuse";
        Remove remove_reuse{reuse_path};
        LiveArchive reuse_writer(reuse_path, metadata, applied, capture, "channels");
        for (unsigned n = 0; n < 32; ++n) {
            for (unsigned i = 0; i < samples.size(); ++i)
                samples[i] = {float(1000 * n + i), -float(i)};
            auto changing = batch;
            changing.first = n;
            changing.count = n % 2 ? 1 : count;
            reuse_writer.channels(changing, {});
            if (n % 4 == 0) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                while (reuse_writer.queued_bytes() && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                check(!reuse_writer.queued_bytes(), "Archive writer did not release reusable payloads");
            }
        }
        reuse_writer.finish();
        std::ifstream reuse_file(reuse_path, std::ios::binary);
        reuse_file.seekg(8);
        const auto reuse_header = read<std::uint32_t>(reuse_file);
        reuse_file.seekg(reuse_header, std::ios::cur);
        for (unsigned n = 0; n < 32; ++n) {
            char tag[4];
            reuse_file.read(tag, 4);
            check(std::memcmp(tag, "CHIQ", 4) == 0, "Reused archive reordered records");
            const auto length = read<std::uint32_t>(reuse_file);
            const auto bytes = read<std::uint64_t>(reuse_file);
            text.resize(length);
            reuse_file.read(text.data(), length);
            const auto info = parse(text);
            const auto count_here = n % 2 ? 1 : count;
            check(info.at("first") == n && bytes == count_here * capture.channels.size() * sizeof(C),
                  "Reused archive retained a stale extent");
            for (unsigned c = 0; c < capture.channels.size(); ++c)
                for (unsigned j = 0; j < count_here; ++j) {
                    const auto i = c * stride + j;
                    check(read<C>(reuse_file) == C{float(1000 * n + i), -float(i)},
                          "Recycled payload aliases a later input or record");
                }
        }
        const auto interrupted_path = path.string() + ".interrupted";
        Remove remove_interrupted{interrupted_path};
        try {
            LiveArchive interrupted(interrupted_path, metadata, applied, capture, "channels");
            interrupted.channels(batch, {});
            throw std::runtime_error("Injected processing failure");
        } catch (const std::runtime_error &) {
        }
        std::ifstream interrupted_file(interrupted_path, std::ios::binary);
        interrupted_file.seekg(8);
        const auto interrupted_header = read<std::uint32_t>(interrupted_file);
        interrupted_file.seekg(interrupted_header, std::ios::cur);
        char tag[4];
        interrupted_file.read(tag, 4);
        check(std::memcmp(tag, "CHIQ", 4) == 0, "Interrupted archive did not drain owned records");
        const auto length = read<std::uint32_t>(interrupted_file);
        const auto bytes = read<std::uint64_t>(interrupted_file);
        text.resize(length);
        interrupted_file.read(text.data(), length);
        check_channels(interrupted_file, parse(text), bytes);
        check(interrupted_file.peek() == std::char_traits<char>::eof(),
              "Failed processing was incorrectly certified by a DONE trailer");
        const auto clock_path = path.string() + ".clock";
        Remove remove_clock{clock_path};
        LiveArchive clock_writer(clock_path, metadata, applied, capture, "channels");
        const std::array<ClockSegment, 3> trajectory{{{0, 0, 1}, {1000, 0, 1.00001}, {2000, .01, .99999}}};
        clock_writer.channels(batch, std::span{trajectory}.first(2));
        clock_writer.channels(batch, std::span{trajectory}.subspan(1));
        clock_writer.channels(batch, std::span{trajectory}.first(1));
        clock_writer.channels(batch, std::span{trajectory}.last(1));
        clock_writer.finish();
        std::ifstream clock_file(clock_path, std::ios::binary);
        clock_file.seekg(8);
        const auto clock_header = read<std::uint32_t>(clock_file);
        clock_file.seekg(clock_header, std::ios::cur);
        unsigned clock_changes = 0;
        for (;;) {
            clock_file.read(tag, 4);
            const auto length = read<std::uint32_t>(clock_file);
            const auto bytes = read<std::uint64_t>(clock_file);
            text.resize(length);
            clock_file.read(text.data(), length);
            if (std::memcmp(tag, "CLCK", 4) == 0) {
                const auto metadata = parse(text);
                for (const auto &segment : metadata.at("segments").as_array()) {
                    check(segment.as_array()[0] == trajectory.at(clock_changes).start,
                          "Overlapping clock snapshots omitted or duplicated a change");
                    ++clock_changes;
                }
            }
            clock_file.seekg(std::streamoff(bytes), std::ios::cur);
            if (std::memcmp(tag, "DONE", 4) == 0)
                break;
        }
        check(clock_changes == trajectory.size(), "Archive lost a rolling clock update");
        std::cout
            << "Async channel/combined archives, exact phase, shared clocks and explicit completion: PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
