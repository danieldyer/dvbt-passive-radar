#include "gpu_ofdm.hpp"
#include "gpu_compute.hpp"
#include "gpu_fft_config.hpp"
#include <bit>
#include <cmath>
#include <numbers>

namespace radar {
namespace {
constexpr std::uint32_t window_code[] =
#include "ofdm_window.inc"
    ;
constexpr std::uint32_t pilots_code[] =
#include "ofdm_pilots.inc"
    ;
constexpr std::uint32_t cp_code[] =
#include "ofdm_cp.inc"
    ;
constexpr std::uint32_t frequency_code[] =
#include "ofdm_frequency.inc"
    ;
constexpr double tau = 2 * std::numbers::pi;
} // namespace
struct GpuOfdm::Impl {
    unsigned capacity, pilots, tps, last_count = 0;
    gpu::Kernel kernel;
    vk::UniquePipeline pilot_pipeline, cp_pipeline, frequency_pipeline;
    VkDevice device = kernel.device();
    VkPhysicalDevice physical = kernel.physical();
    VkQueue queue = kernel.queue();
    VkCommandPool pool = kernel.pool();
    VkFence fence = kernel.fence();
    VkBuffer fft_buffer{};
    std::uint64_t fft_bytes{};
    VkFFTApplication fft{};
    bool ready = false;
    Impl(const PilotPattern &pattern, unsigned cap, std::shared_ptr<gpu::Context> context)
        : capacity(cap), pilots(pattern.continual.size()), tps(pattern.tps.size()),
          kernel(window_code, 16, {}, std::move(context)) {
        if (!cap || cap > 4096 || pattern.continual.empty() || pattern.tps.empty())
            throw std::invalid_argument("Invalid OFDM batch capacity or pilot table");
        fft_bytes = cap * 8192 * sizeof(C);
        kernel.storage(2, fft_bytes);
        fft_buffer = kernel.buffer(2);
        kernel.storage(4, cap * sizeof(C));
        kernel.storage(5, cap * (pilots + 3) * sizeof(C));
        std::vector<C> table(2 * 6817 + pilots + tps);
        for (unsigned k = 0; k < 6817; ++k) {
            if (std::abs(std::abs(pattern.values[k]) - 4.0F / 3) > 1e-6F)
                throw std::invalid_argument("Invalid pilot amplitude");
            const double angle = tau * (static_cast<int>(k) - 3408) * fft_window_margin / 8192;
            table[k] = static_cast<C>(std::polar(1.0, angle)) / pattern.values[k];
            table[6817 + k] = static_cast<C>(std::polar(1.0, -angle));
        }
        for (unsigned j = 0; j < pilots; ++j) {
            if (pattern.continual[j] >= 6817)
                throw std::invalid_argument("Invalid continual pilot index");
            table[2 * 6817 + j] = {std::bit_cast<float>(pattern.continual[j]), 0};
        }
        for (unsigned j = 0; j < tps; ++j) {
            const auto k = pattern.tps[j];
            if (k >= 6817)
                throw std::invalid_argument("Invalid TPS carrier index");
            table[2 * 6817 + pilots + j] = {std::bit_cast<float>(k), pattern.values[k] > .0F ? 1.0F : -1.0F};
        }
        kernel.upload(6, std::span<const C>{table});
        pilot_pipeline = kernel.pipeline(pilots_code);
        cp_pipeline = kernel.pipeline(cp_code);
        frequency_pipeline = kernel.pipeline(frequency_code);
        VkFFTConfiguration config{};
        config.FFTdim = 1;
        config.size[0] = 8192;
        config.numberBatches = cap;
        config.device = &device;
        config.physicalDevice = &physical;
        config.queue = &queue;
        config.commandPool = &pool;
        config.fence = &fence;
        config.buffer = &fft_buffer;
        config.bufferSize = &fft_bytes;
        config.makeForwardPlanOnly = 1;
        config.useLUT = 1;
        gpu::configure_fft(config);
        const auto result = initializeVkFFT(&fft, config);
        if (result != VKFFT_SUCCESS)
            throw std::runtime_error("OFDM VkFFT initialization failed: " + std::to_string(result));
        ready = true;
    }
    ~Impl() {
        if (ready)
            deleteVkFFT(&fft);
    }
};
GpuOfdm::GpuOfdm(const PilotPattern &pattern, unsigned capacity, std::shared_ptr<gpu::Context> context)
    : impl_(std::make_unique<Impl>(pattern, capacity, std::move(context))) {}
GpuOfdm::~GpuOfdm() = default;
void GpuOfdm::wait() {
    impl_->kernel.wait();
}
std::shared_ptr<gpu::Context> GpuOfdm::context() const {
    return impl_->kernel.context();
}
OfdmBatch GpuOfdm::transform(std::span<const C> input, std::span<const OfdmWindow> windows,
                             ComputeTimings *timings) {
    if (windows.empty())
        return {};
    impl_->kernel.upload(0, input);
    return transform(impl_->kernel.view(0), windows, timings);
}
OfdmBatch GpuOfdm::transform(const gpu::BufferView &input, std::span<const OfdmWindow> windows,
                             ComputeTimings *timings, gpu::Commands *commands) {
    if (windows.empty())
        return {};
    auto &p = *impl_;
    if (windows.size() > p.capacity || input.bytes % sizeof(C))
        throw std::invalid_argument("Invalid OFDM batch dimensions");
    std::vector<C> anchors(windows.size() * 129);
    for (std::size_t w = 0; w < windows.size(); ++w) {
        const auto &x = windows[w];
        if (std::uint64_t(x.input_offset) + 8192 > input.bytes / sizeof(C) || !std::isfinite(x.phase) ||
            !std::isfinite(x.frequency_hz) || std::abs(x.frequency_hz) > 4e6)
            throw std::invalid_argument("OFDM window outside input or invalid phase coordinates");
        const double step = -tau * x.frequency_hz / 8e6;
        anchors[w * 129] = {std::bit_cast<float>(x.input_offset), static_cast<float>(step)};
        for (unsigned j = 0; j < 128; ++j)
            anchors[w * 129 + 1 + j] =
                static_cast<C>(std::polar(1.0, std::remainder(x.phase, tau) + step * (j * 64)));
    }
    p.kernel.bind(0, input);
    p.kernel.upload(1, std::span<const C>{anchors});
    const std::vector<C> identity(windows.size(), C{1, 0});
    p.kernel.upload<C>(4, identity);
    const std::array push{static_cast<unsigned>(windows.size()), p.capacity, p.pilots, p.tps};
    const bool detail = timings && timings->detailed;
    const auto cmd = p.kernel.begin(timings, detail ? 5 : 2, commands);
    p.kernel.dispatch(cmd, p.kernel.primary(), push, p.capacity * 8192);
    p.kernel.barrier(cmd);
    if (detail)
        p.kernel.stamp(cmd, 1);
    VkCommandBuffer raw = cmd;
    VkFFTLaunchParams launch{};
    launch.commandBuffer = &raw;
    const auto result = VkFFTAppend(&p.fft, -1, &launch);
    if (result != VKFFT_SUCCESS)
        throw std::runtime_error("OFDM VkFFT dispatch failed");
    p.kernel.barrier(cmd);
    if (detail)
        p.kernel.stamp(cmd, 2);
    p.kernel.groups(cmd, *p.pilot_pipeline, push, static_cast<unsigned>(windows.size()));
    if (detail)
        p.kernel.stamp(cmd, 3);
    p.kernel.finish(cmd);
    p.last_count = static_cast<unsigned>(windows.size());
    OfdmBatch output{p.kernel.view(2), {}, p.kernel.view(5), windows.size(), p.pilots + 3};
    output.carrier_stride = 8192;
    output.phase_corrections = p.kernel.view(4);
    return output;
}
std::vector<IntegerFrequency> GpuOfdm::integer_frequency(unsigned radius, ComputeTimings *timings) {
    auto &p = *impl_;
    if (radius > 256 || !p.last_count)
        throw std::invalid_argument("Invalid integer-CFO probe");
    const unsigned n = 2 * radius + 1;
    const auto *score = static_cast<const float *>(p.kernel.storage(7, p.last_count * n * sizeof(float)));
    const auto cmd = p.kernel.begin(timings);
    p.kernel.groups(cmd, *p.frequency_pipeline, std::array{p.last_count, radius, p.pilots, 0U},
                    p.last_count * n);
    p.kernel.finish(cmd);
    std::vector<IntegerFrequency> result(p.last_count);
    for (unsigned w = 0; w < p.last_count; ++w) {
        const auto *first = score + w * n;
        const auto *best = std::max_element(first, first + n);
        result[w] = {static_cast<int>(best - first) - static_cast<int>(radius), *best};
    }
    return result;
}
void GpuOfdm::rotate(std::span<const double> phases) {
    auto &p = *impl_;
    if (phases.size() != p.last_count || phases.empty())
        throw std::invalid_argument("OFDM phase correction count mismatch");
    std::vector<C> corrections;
    corrections.reserve(phases.size());
    const auto *previous = static_cast<const C *>(p.kernel.view(4).mapped);
    for (double phase : phases) {
        if (!std::isfinite(phase))
            throw std::invalid_argument("Non-finite OFDM phase correction");
        corrections.push_back(previous[corrections.size()] *
                              static_cast<C>(std::polar(1.0, std::remainder(phase, tau))));
    }
    p.kernel.upload(4, std::span<const C>{corrections});
}
std::vector<CpCorrelation> GpuOfdm::correlate(std::span<const C> input,
                                              std::span<const std::uint32_t> offsets, unsigned guard,
                                              ComputeTimings *timings) {
    if (offsets.empty())
        return {};
    impl_->kernel.upload(0, input);
    return correlate(impl_->kernel.view(0), offsets, guard, timings);
}
std::vector<CpCorrelation> GpuOfdm::correlate(const gpu::BufferView &input,
                                              std::span<const std::uint32_t> offsets, unsigned guard,
                                              ComputeTimings *timings) {
    if (offsets.empty())
        return {};
    if (guard != 4 && guard != 8 && guard != 16 && guard != 32)
        throw std::invalid_argument("Invalid guard interval");
    const unsigned length = 8192 / guard;
    for (auto at : offsets)
        if (std::uint64_t(at) + 8192 + length > input.bytes / sizeof(C))
            throw std::invalid_argument("CP correlation outside input");
    auto &p = *impl_;
    p.kernel.bind(0, input);
    p.kernel.upload(1, offsets);
    const auto *scores =
        static_cast<const CpCorrelation *>(p.kernel.storage(7, offsets.size() * sizeof(CpCorrelation)));
    static_assert(sizeof(CpCorrelation) == 4 * sizeof(float));
    const auto cmd = p.kernel.begin(timings);
    p.kernel.groups(cmd, *p.cp_pipeline, std::array{static_cast<unsigned>(offsets.size()), length, 0U, 0U},
                    static_cast<unsigned>(offsets.size()));
    p.kernel.finish(cmd);
    return {scores, scores + offsets.size()};
}
} // namespace radar
