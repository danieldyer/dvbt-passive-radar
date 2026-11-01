#include "gpu_preview.hpp"
namespace radar::video {
namespace {
constexpr std::uint32_t code[] =
#include "preview.inc"
    ;
struct Parameters {
    unsigned ranges, frequencies, range_stride, frequency_stride, width, height;
};
} // namespace
GpuPreview::GpuPreview(std::shared_ptr<gpu::Context> context)
    : kernel_(code, sizeof(Parameters), {}, std::move(context)) {}
void GpuPreview::calculate(const DopplerMap &map, const gpu::BufferView &input) {
    if (!map.ranges || !map.frequencies || input.bytes < map.ranges * map.frequencies * sizeof(C))
        throw std::invalid_argument("Invalid GPU preview dimensions or noise calibration");
    const auto rd = unsigned((map.ranges + 1623) / 1624), fd = unsigned((map.frequencies + 739) / 740);
    width = unsigned((map.ranges + rd - 1) / rd);
    height = unsigned((map.frequencies + fd - 1) / fd);
    kernel_.bind(0, input);
    auto *output = static_cast<float *>(kernel_.storage(3, width * height * sizeof(float)));
    auto *invalid = static_cast<unsigned *>(kernel_.storage(4, sizeof(unsigned)));
    *invalid = 0;
    kernel_.run(Parameters{unsigned(map.ranges), unsigned(map.frequencies), rd, fd, width, height},
                width * height, &timing);
    if (*invalid)
        throw std::runtime_error("Nonfinite scientific map or invalid preview calibration");
    power.assign(output, output + width * height);
}
} // namespace radar::video
