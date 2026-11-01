// Replays archived channelised IQ through the production receiver and detector.
// Equivalent upstream kernels run on resident input alongside it; the saved
// channels are already retimed, so RF acquisition/clock recovery is not replayed.
#include "gpu_commands.hpp"
#include "gpu_frontend.hpp"
#include "gpu_pipeline.hpp"
#include <random>
#ifdef RADAR_BENCH_ARCHIVE
#include "live_archive.hpp"
#endif
#include "matched_detector.hpp"
#include <boost/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <time.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

using namespace radar;
namespace {
using Timer = std::chrono::steady_clock;
double cpu(clockid_t id) {
    timespec t{};
    clock_gettime(id, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
template <class T> T read(std::istream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}
boost::json::object json(std::istream &input, std::size_t size) {
    if (!size || size > 1024 * 1024)
        throw std::runtime_error("Invalid archive metadata size");
    std::string text(size, '\0');
    input.read(text.data(), std::streamsize(size));
    return boost::json::parse(text).as_object();
}
} // namespace
int main(int argc, char **argv) {
    try {
#ifdef __APPLE__
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
        if (argc < 2)
            throw std::invalid_argument(
                "live_pipeline_benchmark FIXTURE.dvblive [--seconds 8 --repetitions "
                "4 --warmups 2 --workers 5 --slots 9 --archive DIR --keep-last --replicate-channel INDEX]");
        double end = 8;
        unsigned repetitions = 4, warmup_replays = 2, workers = 5, slots = 9;
        std::filesystem::path archive_dir;
        bool keep_last = false;
        int replicate = -1;
        for (int i = 2; i < argc; ++i) {
            const std::string_view key = argv[i];
            if (key == "--keep-last") {
                keep_last = true;
                continue;
            }
            if (i + 1 == argc)
                throw std::invalid_argument("Missing benchmark option value");
            const std::string value = argv[++i];
            if (key == "--seconds")
                end = std::stod(value);
            else if (key == "--repetitions")
                repetitions = std::stoul(value);
            else if (key == "--warmups")
                warmup_replays = std::stoul(value);
            else if (key == "--workers")
                workers = std::stoul(value);
            else if (key == "--slots")
                slots = std::stoul(value);
            else if (key == "--replicate-channel")
                replicate = std::stoi(value);
            else if (key == "--archive")
                archive_dir = value;
            else
                throw std::invalid_argument("Unknown benchmark option");
        }
        if (!std::isfinite(end) || end <= 0 || !repetitions || !warmup_replays || warmup_replays > 100 ||
            repetitions > 1000)
            throw std::invalid_argument("Invalid benchmark workload");
#ifndef RADAR_BENCH_ARCHIVE
        if (!archive_dir.empty())
            throw std::invalid_argument("Archive measurement requires the capture driver build");
#endif
        constexpr unsigned map_channels = 0;
        constexpr bool profiled = true, front_work = true, profile_validity = true;
        std::ifstream input(argv[1], std::ios::binary);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        char magic[8];
        input.read(magic, 8);
        if (std::string_view(magic, 8) != "DVBTLIV1")
            throw std::runtime_error("Invalid live archive");
        const auto header = json(input, read<std::uint32_t>(input));
        const auto archive_start = input.tellg();
        if (header.at("archive_mode") != "both" && header.at("archive_mode") != "channels")
            throw std::runtime_error("Channelised IQ is required");
        SarCapture capture;
        capture.input_rate = boost::json::value_to<std::uint32_t>(header.at("corrected_rate_hz"));
        capture.source_rate = boost::json::value_to<std::uint32_t>(header.at("source_nominal_rate_hz"));
        capture.decimation = boost::json::value_to<unsigned>(header.at("decimation"));
        capture.origin = boost::json::value_to<std::uint64_t>(header.at("origin"));
        capture.recording_rf_hz =
            boost::json::value_to<double>(header.at("applied_capture").as_object().at("centre_hz"));
        GpuPipelineConfig config;
        config.max_samples = 262144;
        config.residual_spectra = false;
        config.fec_workers = workers;
        config.frame_slots = slots;
        for (const auto &value : header.at("applied_channels").as_array()) {
            const auto &c = value.as_object();
            const auto guard = boost::json::value_to<unsigned>(c.at("guard"));
            capture.channels.push_back({boost::json::value_to<double>(c.at("rf_hz")),
                                        boost::json::value_to<double>(c.at("nco_hz")), guard});
            config.guards.push_back(guard);
        }
        const auto channels = capture.channels.size();
        if (replicate >= int(channels) || replicate < -1)
            throw std::invalid_argument("Invalid stress channel");
        if (replicate >= 0 && std::ranges::any_of(config.guards, [&](unsigned guard) {
                return guard != config.guards[replicate];
            }))
            throw std::invalid_argument("Replicated stress input requires matching channel guards");
        if (map_channels > channels)
            throw std::invalid_argument("Too many displayed channels");
        std::unique_ptr<GpuChannelBank> front;
        if (front_work) {
            std::vector<double> offsets;
            for (const auto &channel : capture.channels)
                offsets.push_back(channel.nco_hz);
            front = std::make_unique<GpuChannelBank>(capture.source_rate, offsets);
            auto clock = std::make_shared<SampleClock>(front->coordinate_rate(), offsets);
            clock->schedule(64, 1.000002);
            front->set_clock(clock);
        }
        auto context = front ? front->context() : std::make_shared<gpu::Context>();
        GpuPipeline pipeline(config, context, {});
        auto front_storage =
            gpu::make_buffer(context->gpu, front ? front->input_capacity() * sizeof(C) : sizeof(C));
        gpu::BufferView front_input{context, *front_storage.handle,
                                    front ? front->input_capacity() * sizeof(C) : sizeof(C), nullptr};
        if (front) {
            front_input.mapped = context->gpu.device->mapMemory(*front_storage.memory, 0, front_input.bytes);
            std::mt19937 random(8245);
            std::uniform_real_distribution<float> sample(-.1F, .1F);
            for (std::size_t i = 0; i < front->input_capacity(); ++i)
                static_cast<C *>(front_input.mapped)[i] = {sample(random), sample(random)};
        }
        ComputeTimings front_timing;
        front_timing.timestamp_queries = profiled;
        front_timing.detailed = profiled;
        std::uint64_t archive_bytes = 0;
        std::size_t archive_peak_bytes = 0;
#ifdef RADAR_BENCH_ARCHIVE
        std::unique_ptr<LiveArchive> archive;
        RecordingMetadata description;
        AppliedCapture applied;
        if (!archive_dir.empty()) {
            if (!std::filesystem::create_directory(archive_dir))
                throw std::runtime_error("Benchmark archive directory already exists");
            const auto metadata_path = archive_dir / "metadata.json";
            {
                std::ofstream metadata(metadata_path);
                metadata << boost::json::serialize(header);
            }
            description = RecordingMetadata::read(metadata_path);
            applied.driver = "benchmark";
            applied.device = "saved-IQ workload";
            applied.serial = "benchmark";
            applied.rate_hz = capture.source_rate;
            applied.centre_hz = capture.recording_rf_hz;
        }
#else
        if (!archive_dir.empty())
            throw std::runtime_error("Archive benchmark requires radar_live");
#endif
        auto storage = gpu::make_buffer(context->gpu, channels * config.max_samples * sizeof(C));
        auto *mapped =
            context->gpu.device->mapMemory(*storage.memory, 0, channels * config.max_samples * sizeof(C));
        gpu::BufferView samples{context, *storage.handle, channels * config.max_samples * sizeof(C), mapped};
        DetectionOptions options;
        options.publish_maps = map_channels > 0;
        options.update_seconds = .25;
        options.focus.doppler_hz = 100;
        options.focus.range_metres = 100000;
        // A checksum replaces the optional display projector; no full-map readback.
        double profile_checksum = 0;
        std::vector<double> map_checksums(channels);
        MatchedDetector detector(
            context, channels, options,
            [&](const auto &map, const auto &values) {
                map_checksums[map.channel] += std::norm(static_cast<const C *>(values.mapped)[17]);
            },
            [&](const auto &batch, const auto &profiles, auto variance) {
                profile_checksum += std::norm(profiles.values[profiles.values.size() / 2]);
#ifdef RADAR_BENCH_ARCHIVE
                if (archive) {
                    archive->profiles(batch, profiles, variance);
                    archive_peak_bytes = std::max(archive_peak_bytes, archive->queued_bytes());
                }
#endif
            });
        for (unsigned c = 0; c < channels; ++c)
            detector.set_publish_maps(c, c < map_channels);
        std::vector<ClockSegment> clock;
        GpuPipelineTimings processing;
        DetectionTimings detection;
        bool measuring = false;
        std::uint64_t rows = 0, good = 0, bad = 0, maps = 0, samples_processed = 0;
        std::vector<std::uint64_t> checksums(channels);
        std::vector<std::uint64_t> channel_good(channels), channel_bad(channels);
        // Optional scientific-observation accounting. Keep it in the existing
        // validation exclusion, and do not enable it in throughput comparisons.
        std::vector<std::uint64_t> profile_rows(channels), valid_profile_rows(channels);
        std::vector<double> process_latency;
        double io = 0, validation = 0, validation_cpu = 0, main_start = 0, process_start = 0;
        double drain_seconds = 0, archive_finish_seconds = 0, reset_seconds = 0;
        Timer::time_point start;
        double actual_warmup_seconds = 0;
        const auto warmup_started = Timer::now();
        auto publish = [&](std::size_t, DopplerMap &&) {
            if (measuring)
                ++maps;
        };
        auto consume = [&](const SpectralBatch &batch) {
            const auto mark = Timer::now();
            const auto active = cpu(CLOCK_THREAD_CPUTIME_ID);
            if (measuring) {
                rows += batch.frame.last - batch.frame.first;
                const auto mode = batch.frame.frames.front();
                if (mode.constellation != 2 || mode.rate != 2)
                    throw std::runtime_error("Word accounting requires 64-QAM rate 3/4");
                const auto first = (batch.frame.first * 3402 + 203) / 204;
                const auto end_word = batch.frame.last * 3402 / 204;
                const auto last = std::min(batch.packets.packets.size(), end_word > 12 ? end_word - 12 : 0);
                for (auto p = first; p < last; ++p) {
                    const auto &packet = batch.packets.packets[p];
                    if (!packet.valid) {
                        ++bad;
                        ++channel_bad[batch.frame.channel];
                        continue;
                    }
                    ++good;
                    ++channel_good[batch.frame.channel];
                    for (auto byte : packet.bytes)
                        checksums[batch.frame.channel] =
                            (checksums[batch.frame.channel] * 1099511628211ULL) ^ byte;
                }
                if (profile_validity) {
                    const auto first_row =
                        std::min(batch.frame.last, std::max<std::size_t>(1, batch.frame.first));
                    const auto *flags = static_cast<const std::uint32_t *>(batch.transmitted.valid.mapped);
                    const auto known = [&](std::size_t row) {
                        std::uint32_t missing = 0;
                        for (unsigned k = 0; k < 6817; ++k)
                            missing |= flags[row * 6817 + k] == 0;
                        return missing == 0;
                    };
                    bool previous = first_row < batch.frame.last && known(first_row - 1);
                    for (auto row = first_row; row < batch.frame.last; ++row) {
                        const bool current = known(row);
                        ++profile_rows[batch.frame.channel];
                        valid_profile_rows[batch.frame.channel] += previous && current;
                        previous = current;
                    }
                }
            }
            validation += std::chrono::duration<double>(Timer::now() - mark).count();
            validation_cpu += cpu(CLOCK_THREAD_CPUTIME_ID) - active;
            detector.append(SarBatch(capture, batch, clock), publish, measuring ? &detection : nullptr);
        };
        for (unsigned repetition = 0; repetition < warmup_replays + repetitions; ++repetition) {
            // Complete warmup replays commit scratch and can exceed the target's
            // power-control averaging period. Record their actual elapsed duration.
            // Measured replays use identical work; resets are included in time.
            if (repetition) {
                const auto reset_started = Timer::now();
                pipeline.reset();
                detector.reset();
                clock.clear();
                input.clear();
                input.seekg(archive_start);
                if (measuring)
                    reset_seconds += std::chrono::duration<double>(Timer::now() - reset_started).count();
                if (repetition == warmup_replays) {
                    actual_warmup_seconds =
                        std::chrono::duration<double>(Timer::now() - warmup_started).count();
                    measuring = true;
                    validation = validation_cpu = 0;
                    profile_checksum = 0;
                    archive_peak_bytes = 0;
                    start = Timer::now();
                    main_start = cpu(CLOCK_THREAD_CPUTIME_ID);
                    process_start = cpu(CLOCK_PROCESS_CPUTIME_ID);
                }
            }
#ifdef RADAR_BENCH_ARCHIVE
            if (!archive_dir.empty())
                archive = std::make_unique<LiveArchive>(
                    archive_dir / ("profiles-" + std::to_string(repetition) + ".dvblive"), description,
                    applied, capture, "profiles");
#endif
            for (;;) {
                const auto read_start = Timer::now();
                char tag[4];
                input.read(tag, 4);
                const auto length = read<std::uint32_t>(input);
                const auto bytes = read<std::uint64_t>(input);
                const auto meta = json(input, length);
                const auto payload = input.tellg();
                const std::string_view type(tag, 4);
                if (bytes > 64 * 1024 * 1024)
                    throw std::runtime_error("Oversized archive payload");
                if (type == "CLCK")
                    for (const auto &value : meta.at("segments").as_array()) {
                        const auto &s = value.as_array();
                        clock.push_back({boost::json::value_to<std::uint64_t>(s[0]),
                                         boost::json::value_to<double>(s[1]),
                                         boost::json::value_to<double>(s[2])});
                    }
                if (type == "CHIQ") {
                    const auto first = boost::json::value_to<std::uint64_t>(meta.at("first"));
                    const auto count = boost::json::value_to<std::size_t>(meta.at("count"));
                    if (first / 8e6 >= end)
                        break;
                    if (!count || count > config.max_samples || bytes != channels * count * sizeof(C) ||
                        boost::json::value_to<std::size_t>(meta.at("channels")) != channels ||
                        meta.at("order") != "channel,sample")
                        throw std::runtime_error("Invalid channel payload");
                    input.read(static_cast<char *>(mapped), std::streamsize(bytes));
                    if (replicate >= 0) {
                        const auto *source = static_cast<const C *>(mapped) + replicate * count;
                        for (std::size_t c = 0; c < channels; ++c)
                            if (c != std::size_t(replicate))
                                std::copy_n(source, count, static_cast<C *>(mapped) + c * count);
                    }
                    if (measuring)
                        io += std::chrono::duration<double>(Timer::now() - read_start).count();
                    const auto process_start = Timer::now();
                    if (front) {
                        const auto centre = 65536 + first * front->decimation;
                        const auto support = front->input_support(centre, count);
                        front->process_clocked_device(front_input, std::size_t(support.last - support.first),
                                                      support.first, centre, count,
                                                      measuring ? &front_timing : nullptr);
                    }
                    pipeline.process({samples, first, count, channels, count}, consume,
                                     measuring ? &processing : nullptr);
                    if (measuring)
                        process_latency.push_back(
                            std::chrono::duration<double>(Timer::now() - process_start).count());
                    if (measuring)
                        samples_processed += count;
                } else
                    input.seekg(payload + std::streamoff(bytes));
                if (type == "DONE")
                    break;
            }
            const auto drain_started = Timer::now();
            const auto validation_before_drain = validation;
            pipeline.finish(consume, measuring ? &processing : nullptr);
            detector.finish(publish, measuring ? &detection : nullptr);
            if (measuring)
                drain_seconds += std::chrono::duration<double>(Timer::now() - drain_started).count() -
                                 (validation - validation_before_drain);
#ifdef RADAR_BENCH_ARCHIVE
            if (archive) {
                const auto archive_started = Timer::now();
                archive->finish();
                if (measuring)
                    archive_bytes += archive->bytes();
                archive.reset();
                // Only our just-created benchmark output; never a supplied capture.
                if (!keep_last || repetition + 1 < warmup_replays + repetitions)
                    std::filesystem::remove(archive_dir /
                                            ("profiles-" + std::to_string(repetition) + ".dvblive"));
                if (measuring)
                    archive_finish_seconds +=
                        std::chrono::duration<double>(Timer::now() - archive_started).count();
            }
#endif
            if (!measuring) {
                processing = {};
                detection = {};
                for (auto *t :
                     {&processing.receiver, &processing.frames, &processing.reconstruction,
                      &processing.response, &detection.noise, &detection.profiles, &detection.focus}) {
                    t->timestamp_queries = profiled;
                    t->detailed = profiled;
                }
            }
        }
        const double seconds = samples_processed / 8e6, scale = 1000 / seconds;
        if (!measuring || seconds <= 0 || !rows)
            throw std::runtime_error("No measured output");
        std::ranges::sort(process_latency);
        boost::json::object result{
            {"signal_seconds", seconds},
            {"channels", channels},
            {"map_channels", map_channels},
            {"profiled", profiled},
            {"front_work", front_work},
            {"archive_bytes", archive_bytes},
            {"archive_queue_peak_bytes", archive_peak_bytes},
            {"repetitions", repetitions},
            {"full_replay_warmup", true},
            {"pilot_radius", DecoderPolicy::pilot_radius},
            {"adaptive_pilots", true},
            {"fec_passes", maximum_fec_passes},
            {"rs_rescue", true},
            {"frames_per_batch", config.frames_per_batch},
            {"workers", config.fec_workers},
            {"slots", config.frame_slots},
            {"replicate_channel", replicate},
            {"warmup_seconds", actual_warmup_seconds},
            {"warmup_replays", warmup_replays},
            {"range_metres", options.focus.range_metres},
            {"channel_rs_good", boost::json::value_from(channel_good)},
            {"channel_rs_bad", boost::json::value_from(channel_bad)},
            {"profile_validity_measured", profile_validity},
            {"channel_profile_rows", boost::json::value_from(profile_rows)},
            {"channel_valid_profile_rows", boost::json::value_from(valid_profile_rows)},
            {"process_max_ms", 1000 * process_latency.back()},
            {"process_p99_ms", 1000 * process_latency[process_latency.size() * 99 / 100]},
            {"wall_ms_per_s",
             (std::chrono::duration<double>(Timer::now() - start).count() - validation) * scale},
            {"read_ms_per_s", io * scale},
            {"drain_ms_per_s", drain_seconds * scale},
            {"archive_finish_ms_per_s", archive_finish_seconds * scale},
            {"reset_ms_per_s", reset_seconds * scale},
            {"main_cpu_ms_per_s", (cpu(CLOCK_THREAD_CPUTIME_ID) - main_start - validation_cpu) * scale},
            {"process_cpu_ms_per_s",
             (cpu(CLOCK_PROCESS_CPUTIME_ID) - process_start - validation_cpu) * scale},
            {"viterbi_worker_ms_per_s", processing.fec.viterbi * scale},
            {"rs_worker_ms_per_s", processing.fec.reed_solomon * scale},
            {"fec_prepare_ms_per_s", processing.fec.prepare * scale},
            {"initial_viterbi_worker_ms_per_s", processing.fec.initial_viterbi * scale},
            {"feedback_viterbi_worker_ms_per_s",
             (processing.fec.viterbi - processing.fec.initial_viterbi) * scale},
            {"rows", rows},
            {"rs_good", good},
            {"rs_bad", bad},
            {"maps", maps},
            {"packet_checksums", boost::json::value_from(checksums)},
            {"map_checksum", std::accumulate(map_checksums.begin(), map_checksums.end(), 0.)},
            {"profile_checksum", profile_checksum}};
        auto gpu_time = [&](const char *name, const ComputeTimings &t) { result[name] = t.gpu * scale; };
        gpu_time("channelizer_gpu_ms_per_s", front_timing);
        gpu_time("receiver_gpu_ms_per_s", processing.receiver);
        gpu_time("frames_gpu_ms_per_s", processing.frames);
        gpu_time("reconstruction_gpu_ms_per_s", processing.reconstruction);
        gpu_time("response_gpu_ms_per_s", processing.response);
        gpu_time("noise_gpu_ms_per_s", detection.noise);
        gpu_time("profiles_gpu_ms_per_s", detection.profiles);
        gpu_time("focus_gpu_ms_per_s", detection.focus);
        std::cout << boost::json::serialize(result) << '\n';
        if (front)
            context->gpu.device->unmapMemory(*front_storage.memory);
        context->gpu.device->unmapMemory(*storage.memory);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
