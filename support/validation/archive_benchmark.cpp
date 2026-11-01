#include "live_archive.hpp"
#include <chrono>
#include <ctime>
#include <iostream>
#include <sys/resource.h>
using namespace radar;
double cpu_seconds() {
    timespec t{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return double(t.tv_sec) + 1e-9 * t.tv_nsec;
}
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("Metadata path required");
        const auto metadata = RecordingMetadata::read(argv[1]);
        SarCapture capture;
        metadata.configure(capture);
        capture.input_rate = 64'000'000;
        capture.decimation = 8;
        capture.source_rate = 64'000'000;
        capture.validate();
        AppliedCapture applied;
        applied.driver = "benchmark";
        applied.rate_hz = 64e6;
        const auto path = std::filesystem::temp_directory_path() /
                          ("dvbt-archive-benchmark-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        struct Remove {
            std::filesystem::path path;
            ~Remove() {
                std::error_code e;
                std::filesystem::remove(path, e);
            }
        } remove{path};
        constexpr unsigned rows = 273, iterations = 30;
        FrameBatch frame;
        frame.first = 1;
        frame.last = rows;
        frame.symbols.resize(rows);
        frame.frames.resize(1);
        frame.carriers.count = rows;
        for (unsigned r = 0; r < rows; ++r)
            frame.symbols[r].window.sample = r * 8704ULL;
        std::vector<C> response(6817, C{.8F, .3F});
        std::vector<float> noise(6817, 1);
        std::vector<std::uint32_t> valid(rows * 6817, 1);
        const auto view = [](auto &v) { return gpu::BufferView{{}, {}, v.size() * sizeof(v[0]), v.data()}; };
        SpectralBatch spectra{frame, {}, {{}, view(valid), rows}, {view(response), {}, rows, 128.375, 0}, {}};
        spectra.residual.count = rows;
        SarBatch batch(capture, spectra, {{}});
        MatchedProfiles profiles;
        profiles.size = 4096;
        profiles.first = 1;
        profiles.count = rows - 1;
        profiles.values.resize(2 * profiles.count * profiles.size);
        for (std::size_t i = 0; i < profiles.values.size(); ++i)
            profiles.values[i] = {float(i % 4021), -float(i % 2039)};
        LiveArchive writer(path, metadata, applied, capture, "profiles");
        writer.profiles(batch, profiles, noise);
        rusage before{}, after{};
        getrusage(RUSAGE_SELF, &before);
        double cpu = 0;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned n = 0; n < iterations; ++n) {
            const auto before = cpu_seconds();
            writer.profiles(batch, profiles, noise);
            cpu += cpu_seconds() - before;
        }
        writer.finish();
        getrusage(RUSAGE_SELF, &after);
        std::cout << "{\"producer_cpu_ms\":" << cpu * 1000 / iterations << ",\"wall_ms\":"
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1000 /
                         iterations
                  << ",\"minor_faults\":" << after.ru_minflt - before.ru_minflt
                  << ",\"bytes\":" << writer.bytes() << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
