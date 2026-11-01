#include "dsp.hpp"
#include "gpu_compute.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vkFFT.h>

using namespace radar;
namespace {
constexpr std::uint32_t code[] =
#include "response_apply.inc"
    ;
struct Plan {
    VkFFTApplication app{};
    explicit Plan(VkFFTConfiguration config) {
        if (initializeVkFFT(&app, config) != VKFFT_SUCCESS)
            throw std::runtime_error("FFT plan failed");
    }
    ~Plan() {
        deleteVkFFT(&app);
    }
};
} // namespace
int main(int argc, char **argv) {
    try {
        const unsigned n = argc > 1 ? std::stoul(argv[1]) : 8192,
                       batches = argc > 2 ? std::stoul(argv[2]) : 256;
        bool lut_only = false, boost_nonpow2 = false;
        for (int i = 3; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "--lut-only")
                lut_only = true;
            else if (option == "--boost-nonpow2")
                boost_nonpow2 = true;
            else
                throw std::invalid_argument("Expected [size] [batches] [--lut-only] [--boost-nonpow2]");
        }
        gpu::Kernel kernel(code, 16);
        auto context = kernel.context();
        const auto properties = context->gpu.physical.getProperties();
        std::cerr << properties.deviceName
                  << " shared_memory=" << properties.limits.maxComputeSharedMemorySize << '\n';
        const auto memory = context->gpu.physical.getMemoryProperties();
        for (unsigned i = 0; i < memory.memoryTypeCount; ++i)
            std::cerr << "memory_type=" << i
                      << " flags=" << vk::to_string(memory.memoryTypes[i].propertyFlags) << '\n';
        VkDevice device = kernel.device();
        VkPhysicalDevice physical = kernel.physical();
        VkQueue queue = kernel.queue();
        VkCommandPool pool = kernel.pool();
        VkFence fence = kernel.fence();
        std::uint64_t bytes = std::uint64_t(n) * batches * sizeof(C);
        auto *mapped = static_cast<C *>(kernel.storage(0, bytes));
        VkBuffer buffer = kernel.buffer(0);
        std::vector<C> input(n * batches);
        std::mt19937 rng(187);
        std::uniform_real_distribution<float> random(-1, 1);
        for (auto &z : input)
            z = {random(rng), random(rng)};
        std::cout << "size,batches,lut,register_boost,boost_nonpow2,threads,actual_register_boost,actual_"
                     "threads,axis_uploads,gpu_us_per_fft_"
                     "batch,roundtrip_relative_rms\n"
                  << std::setprecision(10);
        for (int lut : {1})
            for (unsigned boost : {0U, 1U, 2U, 4U, 8U})
                for (unsigned threads : {0U, 64U, 128U, 256U}) {
                    if (!boost && threads)
                        continue;
                    if (lut_only && lut != 1)
                        continue;
                    VkFFTConfiguration config{};
                    config.FFTdim = 1;
                    config.size[0] = n;
                    config.numberBatches = batches;
                    config.device = &device;
                    config.physicalDevice = &physical;
                    config.queue = &queue;
                    config.commandPool = &pool;
                    config.fence = &fence;
                    config.buffer = &buffer;
                    config.bufferSize = &bytes;
                    config.normalize = 1;
                    config.useLUT = lut;
                    config.registerBoost = boost;
                    config.aimThreads = threads;
                    config.registerBoostNonPow2 = boost_nonpow2;
                    Plan plan(config);
                    ComputeTimings timing;
                    using Clock = std::chrono::steady_clock;
                    const auto warm_end = Clock::now() + std::chrono::milliseconds(500);
                    auto measure_end = Clock::time_point::max();
                    while (Clock::now() < measure_end) {
                        const bool measuring = Clock::now() >= warm_end;
                        if (measuring && measure_end == Clock::time_point::max())
                            measure_end = Clock::now() + std::chrono::seconds(1);
                        std::copy(input.begin(), input.end(), mapped);
                        auto cmd = kernel.begin(measuring ? &timing : nullptr);
                        VkCommandBuffer raw = cmd;
                        VkFFTLaunchParams launch{};
                        launch.commandBuffer = &raw;
                        for (unsigned repeat = 0; repeat < 10; ++repeat)
                            for (int direction : {-1, 1}) {
                                if (VkFFTAppend(&plan.app, direction, &launch) != VKFFT_SUCCESS)
                                    throw std::runtime_error("FFT dispatch failed");
                                gpu::Kernel::barrier(cmd);
                            }
                        kernel.finish(cmd);
                    }
                    double error = 0, energy = 0;
                    for (std::size_t i = 0; i < input.size(); ++i) {
                        error += std::norm(mapped[i] - input[i]);
                        energy += std::norm(input[i]);
                    }
                    if (error / energy > 1e-10)
                        throw std::runtime_error("FFT roundtrip lost precision");
                    std::cout << n << ',' << batches << ',' << lut << ',' << boost << ',' << boost_nonpow2
                              << ',' << threads << ',' << plan.app.configuration.registerBoost << ','
                              << plan.app.configuration.aimThreads << ','
                              << plan.app.localFFTPlan->numAxisUploads[0] << ','
                              << timing.gpu * 1e6 / (20 * timing.batches) << ',' << std::sqrt(error / energy)
                              << '\n'
                              << std::flush;
                }
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
