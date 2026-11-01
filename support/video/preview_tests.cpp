#include "gpu_focus.hpp"
#include "gpu_preview.hpp"
#include "matched_detector.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <time.h>

using namespace radar;
namespace {
using Timer = std::chrono::steady_clock;
struct Plane {
    std::shared_ptr<gpu::Context> context;
    gpu::Buffer storage;
    gpu::BufferView view;
    Plane(std::shared_ptr<gpu::Context> c, std::size_t bytes)
        : context(std::move(c)), storage(gpu::make_buffer(context->gpu, bytes)),
          view{context, *storage.handle, bytes, context->gpu.device->mapMemory(*storage.memory, 0, bytes)} {}
    ~Plane() {
        context->gpu.device->unmapMemory(*storage.memory);
    }
};
double cpu() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
std::vector<float> reference(const DopplerMap &map) {
    const auto rd = (map.ranges + 1623) / 1624, fd = (map.frequencies + 739) / 740;
    const auto width = (map.ranges + rd - 1) / rd, height = (map.frequencies + fd - 1) / fd;
    std::vector<float> result(width * height);
    for (std::size_t r = 0; r < map.ranges; ++r)
        for (std::size_t f = 0; f < map.frequencies; ++f) {
            auto &out = result[(height - 1 - f / fd) * width + r / rd];
            out = std::max(out, std::norm(map.values[r * map.frequencies + f]));
        }
    return result;
}
void compare(std::span<const float> a, std::span<const float> b) {
    check(a.size() == b.size(), "Preview shape mismatch");
    double maximum = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        maximum = std::max(maximum, std::abs(10 * std::log10(double(a[i]) / b[i])));
    check(maximum < .00002, "GPU peak pooling or noise scaling differs from CPU reference");
    std::cout << "Maximum GPU/CPU power difference: " << maximum << " dB\n";
}
} // namespace
int main(int argc, char **argv) {
    try {
        const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
        auto context = std::make_shared<gpu::Context>();
        video::GpuPreview preview(context);
        std::mt19937 rng(9832);
        std::normal_distribution<float> normal;
        for (const auto [nr, nf] : {std::pair{103U, 281U}, {3247U, 5021U}, {3203U, 2509U}}) {
            DopplerMap map;
            map.ranges = nr;
            map.frequencies = nf;
            map.range_step = 10;
            map.frequency_start = -100;
            map.frequency_step = 200.0 / (nf - 1);
            map.integration = 3;
            map.amplitude_units = "complex_reflectivity";
            map.values.resize(std::size_t(nr) * nf);
            for (auto &value : map.values)
                value = {normal(rng), normal(rng)};
            map.values.back() = {100, 200}; // Odd trailing cells must retain narrow peaks.
            Plane source(context, map.values.size() * sizeof(C));
            std::copy(map.values.begin(), map.values.end(), static_cast<C *>(source.view.mapped));
            preview.calculate(map, source.view);
            compare(preview.power, reference(map));
            check(
                std::equal(map.values.begin(), map.values.end(), static_cast<const C *>(source.view.mapped)),
                "GPU preview overwrote scientific amplitude or phase");
            const auto result = preview.power;
            preview.calculate(map, source.view);
            check(result == preview.power, "Reusing GPU preview storage changed output");
            static_cast<C *>(source.view.mapped)[nf + 1] = {std::numeric_limits<float>::quiet_NaN(), 0};
            bool failed = false;
            try {
                preview.calculate(map, source.view);
            } catch (const std::runtime_error &) {
                failed = true;
            }
            check(failed, "GPU preview accepted a nonfinite map");
            static_cast<C *>(source.view.mapped)[nf + 1] = {1e30F, 1e30F};
            failed = false;
            try {
                preview.calculate(map, source.view);
            } catch (const std::runtime_error &) {
                failed = true;
            }
            check(failed, "GPU preview accepted overflowing display power");
        }
        GpuFocus focus(context);
        constexpr unsigned rows = 64;
        std::vector<C> moments(2 * rows * 1024);
        for (auto &value : moments)
            value = {normal(rng), normal(rng)};
        auto copied = focus.calculate(moments, rows, .001088, 177.5e6, 12, {1000, 100, true});
        auto device = focus.calculate_device(moments, rows, .001088, 177.5e6, 12, {1000, 100, true});
        check(device.map.values.empty(), "Device focus unexpectedly copied complex samples");
        check(std::equal(copied.values.begin(), copied.values.end(), static_cast<C *>(device.values.mapped)),
              "Device focus changed scientific output");
        if (benchmark) {
            constexpr unsigned n = std::size_t(3 / .001088) + 1, repeats = 8;
            moments.resize(2 * n * 1024);
            for (auto &value : moments)
                value = {normal(rng), normal(rng)};
            focus.prepare(n);
            focus.calculate_device(moments, n, .001088, 177.5e6, 0, {30000, 100, true});
            ComputeTimings timing;
            double calibration = 0, preview_wall = 0, focus_wall = 0;
            for (unsigned i = 0; i < repeats; ++i) {
                auto mark = Timer::now();
                auto result =
                    focus.calculate_device(moments, n, .001088, 177.5e6, 0, {30000, 100, true}, &timing);
                focus_wall += std::chrono::duration<double>(Timer::now() - mark).count();
                mark = Timer::now();
                preview.calculate(result.map, result.values);
                preview_wall += std::chrono::duration<double>(Timer::now() - mark).count();
            }
            std::cout << "{\"maps\":" << repeats
                      << ",\"focus_wall_ms_per_map\":" << focus_wall * 1000 / repeats
                      << ",\"focus_gpu_ms_per_map\":" << timing.gpu * 1000 / repeats
                      << ",\"calibration_cpu_ms_per_map\":" << calibration * 1000 / repeats
                      << ",\"preview_wall_ms_per_map\":" << preview_wall * 1000 / repeats << "}\n";
        }
        std::cout << "GPU preview and device-map regression: PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
