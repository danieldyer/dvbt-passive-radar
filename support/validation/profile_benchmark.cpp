#include "gpu_compute.hpp"
#include "matched_profiles.hpp"
#include <chrono>
#include <iostream>
#include <random>

using namespace radar;
namespace {
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
    template <class T> void set(std::span<const T> values) {
        std::ranges::copy(values, static_cast<T *>(view.mapped));
    }
};
} // namespace
// Isolate production finite-waveform correlation, including both time moments,
// CP correction and the host output copy. No receiver/FEC or archive work.
int main(int argc, char **argv) {
    try {
        const unsigned tile = argc > 1 ? std::stoul(argv[1]) : 64;
        const unsigned rows = argc > 2 ? std::stoul(argv[2]) : tile + 1;
        const unsigned repeats = argc > 3 ? std::stoul(argv[3]) : 100;
        if (rows < 2 || rows > 1020 || !repeats)
            throw std::invalid_argument("Expected [tile] [rows=tile+1, 2..1020] [repeats=100]");
        constexpr unsigned bins = 6817;
        auto context = std::make_shared<gpu::Context>();
        std::vector<C> source(rows * bins), received(source.size()), response(bins, C{.8F, .3F});
        std::vector<std::uint32_t> valid(source.size(), 1);
        std::vector<float> variance(bins, 1);
        std::mt19937 random(31415);
        std::normal_distribution<float> normal;
        for (unsigned i = 0; i < source.size(); ++i) {
            source[i] = {normal(random), normal(random)};
            received[i] = response[i % bins] * source[i] + C{normal(random), normal(random)};
        }
        Plane x(context, source.size() * sizeof(C)), y(context, received.size() * sizeof(C)),
            h(context, bins * sizeof(C)), v(context, valid.size() * sizeof(std::uint32_t)),
            noise(context, bins * sizeof(float));
        x.set<C>(source);
        y.set<C>(received);
        h.set<C>(response);
        v.set<std::uint32_t>(valid);
        noise.set<float>(variance);
        FrameBatch frame;
        frame.first = 1;
        frame.last = rows;
        frame.carriers.measured = y.view;
        SpectralBatch batch{frame, {}, {x.view, v.view, rows}, {h.view, {}, rows, 128.375, 0}, {}};
        GpuMatchedProfiles processor(context, matched_compact_profile_size, tile);
        MatchedProfiles output;
        for (unsigned i = 0; i < 20; ++i)
            processor.calculate_into(output, batch, noise.view, 16);
        ComputeTimings timing;
        timing.detailed = true;
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 0; i < repeats; ++i)
            processor.calculate_into(output, batch, noise.view, 16, &timing);
        const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        double energy = 0;
        for (auto value : output.values)
            energy += std::norm(value);
        std::cout << "tile,rows,repeats,wall_ms,gpu_ms,samples_ms,prefix_ms,moment_ms,product_ms,extract_ms,"
                     "energy\n"
                  << tile << ',' << rows << ',' << repeats << ',' << wall * 1000 / repeats << ','
                  << timing.gpu * 1000 / repeats;
        for (auto stage : timing.stages)
            std::cout << ',' << stage * 1000 / repeats;
        std::cout << ',' << energy << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
