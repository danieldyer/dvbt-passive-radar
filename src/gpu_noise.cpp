#include "gpu_noise.hpp"
#include "gpu_compute.hpp"
#include <cmath>

namespace radar {
namespace {
constexpr std::uint32_t variance_code[] =
#include "noise_variance.inc"
    ;

} // namespace
struct GpuNoise::Impl {
    gpu::Kernel kernel;
    bool ready = false;
    explicit Impl(std::shared_ptr<gpu::Context> context) : kernel(variance_code, 16, {}, std::move(context)) {
        kernel.storage(5, 6817 * sizeof(float));
        kernel.storage(6, 6817 * sizeof(float));
    }
};
GpuNoise::GpuNoise(std::shared_ptr<gpu::Context> context)
    : impl_(std::make_unique<Impl>(std::move(context))) {}
GpuNoise::~GpuNoise() = default;
void GpuNoise::reset() {
    impl_->kernel.wait();
    impl_->ready = false;
}
bool GpuNoise::ready() const {
    return impl_->ready;
}
gpu::BufferView GpuNoise::estimate(const SpectralBatch &batch, ComputeTimings *timing) {
    auto &p = *impl_;
    const auto &f = batch.frame;
    if (f.last <= f.first + 2 || f.last > batch.residual.count || batch.residual.count > 1020)
        throw std::invalid_argument("Noise estimation requires three new symbol rows");
    p.kernel.bind(0, f.carriers.measured);
    p.kernel.bind(1, batch.residual.reference);
    p.kernel.bind(2, batch.transmitted.valid);
    std::array push{unsigned(f.first), unsigned(f.last), unsigned(p.ready), 0U};
    const auto cmd = p.kernel.begin(timing);
    p.kernel.dispatch(cmd, p.kernel.primary(), push, 6817);
    p.kernel.finish(cmd);
    p.ready = true;
    const auto result = p.kernel.view(5);
    const auto values = std::span{static_cast<const float *>(result.mapped), std::size_t(6817)};
    if (std::ranges::any_of(values, [](float v) { return !std::isfinite(v) || v <= 0; }))
        throw std::runtime_error("Nonfinite carrier covariance estimate");
    return result;
}
} // namespace radar
