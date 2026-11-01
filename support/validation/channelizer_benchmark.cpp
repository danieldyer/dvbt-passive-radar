// Resident common-clock channelizer benchmark; no file/receive/archive timing.
#include "gpu_compute.hpp"
#include "gpu_frontend.hpp"
#include <chrono>
#include <iostream>
#include <random>
using namespace radar;
int main() {
    try {
        constexpr unsigned rate = 32'000'000;
        const std::array centres{-7e6, 0., 7e6};
        std::cout
            << "clock_ppm,outputs,repeats,gpu_ms_per_input_second,forward_ms,boundary_ms,synthesis_ms\n";
        for (double ppm : {0., 2., -20.}) {
            auto clock = std::make_shared<SampleClock>(rate, centres);
            clock->schedule(64, 1 + ppm * 1e-6);
            GpuChannelBank bank(rate, centres);
            bank.set_clock(clock);
            constexpr std::uint64_t first = 32768;
            const auto count = bank.capacity();
            const auto support = bank.input_support(first, count);
            const auto valid = std::size_t(support.last - support.first),
                       bytes = bank.input_capacity() * sizeof(C);
            auto buffer = gpu::make_buffer(bank.context()->gpu, bytes);
            auto *data = static_cast<C *>(bank.context()->gpu.device->mapMemory(*buffer.memory, 0, bytes));
            std::mt19937 random(92744);
            std::uniform_real_distribution<float> uniform(-.1F, .1F);
            for (std::size_t i = 0; i < bank.input_capacity(); ++i)
                data[i] = {uniform(random), uniform(random)};
            const gpu::BufferView input{bank.context(), *buffer.handle, bytes, data};
            using Clock = std::chrono::steady_clock;
            const auto warm = Clock::now() + std::chrono::seconds{1};
            do {
                bank.process_clocked_device(input, valid, support.first, first, count);
            } while (Clock::now() < warm);
            ComputeTimings timing;
            timing.detailed = true;
            unsigned repeats = 0;
            const auto until = Clock::now() + std::chrono::seconds{3};
            do {
                bank.process_clocked_device(input, valid, support.first, first, count, &timing);
                ++repeats;
            } while (Clock::now() < until);
            const double scale = 1000 / (repeats * double(count) / 8e6);
            std::cout << ppm << ',' << count << ',' << repeats << ',' << timing.gpu * scale << ','
                      << timing.stages[0] * scale << ',' << timing.stages[1] * scale << ','
                      << timing.stages[2] * scale << '\n';
            bank.context()->gpu.device->unmapMemory(*buffer.memory);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
