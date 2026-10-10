#include "vulkan_context.hpp"
#include <vulkan/vulkan.hpp>
#include <vkFFT.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Complex = std::complex<float>;
static_assert(sizeof(Complex) == 2 * sizeof(float));

constexpr std::uint32_t cross_spectrum_spirv[] =
#include "cross_spectrum.inc"
;

using namespace radar::gpu;

void check_fft(VkFFTResult result, std::string_view operation)
{
    if (result != VKFFT_SUCCESS) {
        throw std::runtime_error(std::string{operation} + " failed: VkFFTResult=" +
                                 std::to_string(result));
    }
}

struct FftPlan {
    VkFFTApplication app{};

    explicit FftPlan(const VkFFTConfiguration& config)
    {
        // VkFFT cleans up partially initialized plans on failure.
        check_fft(initializeVkFFT(&app, config), "initializeVkFFT");
    }
    ~FftPlan() { deleteVkFFT(&app); }
    FftPlan(const FftPlan&) = delete;
    auto operator=(const FftPlan&) -> FftPlan& = delete;
};

auto sample_count(int argc, char** argv) -> std::uint32_t
{
    std::uint32_t count = 8192;
    if (argc == 2) {
        const std::string_view value{argv[1]};
        const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), count);
        if (error == std::errc{} && end == value.data() + value.size() &&
            count >= 16 && count <= 1'048'576) {
            return count;
        }
    } else if (argc == 1) {
        return count;
    }
    throw std::runtime_error("Usage: dvbt_passive_radar [FFT length: 16..1048576]");
}

void run(std::uint32_t count)
{
    const auto instance = make_instance();
    const auto gpu = make_device(*instance);
    const auto device = *gpu.device;
    const auto limits = gpu.physical.getProperties().limits;
    const auto bytes = static_cast<vk::DeviceSize>(3) * count * sizeof(Complex);
    const auto groups = (count + 127U) / 128U;
    if (bytes > limits.maxStorageBufferRange || groups > limits.maxComputeWorkGroupCount[0]) {
        throw std::runtime_error("FFT length exceeds this device's storage/dispatch limits");
    }
    std::cout << "Device: " << gpu.physical.getProperties().deviceName << '\n';

    auto buffer = make_buffer(gpu, bytes);
    std::vector<Complex> samples(3 * count);
    // Two sparse complex time signals give an independent analytic DFT at every bin.
    samples[0] = {1.0F, 0.25F};
    samples[7] = {-0.5F, 0.75F};
    samples[count + 3] = {0.25F, -1.0F};
    samples[count + 11] = {0.5F, 0.5F};
    auto* mapped = device.mapMemory(*buffer.memory, 0, bytes);
    std::memcpy(mapped, samples.data(), static_cast<std::size_t>(bytes));
    device.unmapMemory(*buffer.memory);

    const auto pool = device.createCommandPoolUnique({{}, gpu.family});
    const auto fence = device.createFenceUnique({});
    VkDevice device_handle = device;
    VkPhysicalDevice physical_handle = gpu.physical;
    VkQueue queue_handle = device.getQueue(gpu.family, 0);
    VkCommandPool pool_handle = *pool;
    VkFence fence_handle = *fence;
    VkBuffer buffer_handle = *buffer.handle;
    std::uint64_t fft_bytes = 2ULL * count * sizeof(Complex);
    VkFFTConfiguration config{};
    config.FFTdim = 1;
    config.size[0] = count;
    config.numberBatches = 2;
    config.device = &device_handle;
    config.physicalDevice = &physical_handle;
    config.queue = &queue_handle;
    config.commandPool = &pool_handle;
    config.fence = &fence_handle;
    config.buffer = &buffer_handle;
    config.bufferSize = &fft_bytes;
    FftPlan fft{config};

    const vk::DescriptorSetLayoutBinding binding{
        0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute};
    const auto set_layout = device.createDescriptorSetLayoutUnique({{}, binding});
    const vk::PushConstantRange push_constants{vk::ShaderStageFlagBits::eCompute, 0,
                                               sizeof(count)};
    const auto layout = device.createPipelineLayoutUnique({{}, *set_layout, push_constants});
    const auto shader = device.createShaderModuleUnique(
        {{}, sizeof(cross_spectrum_spirv), cross_spectrum_spirv});
    const vk::PipelineShaderStageCreateInfo stage{
        {}, vk::ShaderStageFlagBits::eCompute, *shader, "main"};
    const auto pipeline = device.createComputePipelineUnique({}, {{}, stage, *layout}).value;

    const vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer, 1};
    const auto descriptor_pool = device.createDescriptorPoolUnique({{}, 1, pool_size});
    const auto sets = device.allocateDescriptorSets({*descriptor_pool, *set_layout});
    const vk::DescriptorBufferInfo buffer_info{*buffer.handle, 0, bytes};
    const vk::WriteDescriptorSet write{
        sets.front(), 0, 0, vk::DescriptorType::eStorageBuffer, {}, buffer_info};
    device.updateDescriptorSets(write, {});

    const auto commands = device.allocateCommandBuffersUnique({*pool, vk::CommandBufferLevel::ePrimary, 1});
    const auto cmd = *commands.front();
    cmd.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    VkCommandBuffer command_handle = cmd;
    VkFFTLaunchParams launch{};
    launch.commandBuffer = &command_handle;
    check_fft(VkFFTAppend(&fft.app, -1, &launch), "VkFFTAppend");
    const vk::MemoryBarrier fft_to_product{vk::AccessFlagBits::eShaderWrite,
                                          vk::AccessFlagBits::eShaderRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, fft_to_product, {}, {});
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout, 0, sets, {});
    cmd.pushConstants<std::uint32_t>(*layout, vk::ShaderStageFlagBits::eCompute, 0, count);
    cmd.dispatch(groups, 1, 1);
    const vk::MemoryBarrier product_to_host{vk::AccessFlagBits::eShaderWrite,
                                           vk::AccessFlagBits::eHostRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eHost, {}, product_to_host, {}, {});
    cmd.end();
    device.getQueue(gpu.family, 0).submit(vk::SubmitInfo{{}, {}, cmd});
    device.waitIdle();

    mapped = device.mapMemory(*buffer.memory, 0, bytes);
    std::memcpy(samples.data(), mapped, static_cast<std::size_t>(bytes));
    device.unmapMemory(*buffer.memory);
    double max_error = 0.0;
    for (std::uint32_t k = 0; k < count; ++k) {
        const double phase = -2.0 * std::numbers::pi * k / count;
        const auto reference = std::complex{1.0, 0.25} +
                               std::complex{-0.5, 0.75} * std::polar(1.0, 7.0 * phase);
        const auto surveillance = std::complex{0.25, -1.0} * std::polar(1.0, 3.0 * phase) +
                                  std::complex{0.5, 0.5} * std::polar(1.0, 11.0 * phase);
        const auto actual = std::complex<double>{samples[2 * count + k]};
        const double error = std::abs(actual - surveillance * std::conj(reference));
        if (!std::isfinite(error)) {
            throw std::runtime_error("Cross spectrum contains non-finite values");
        }
        max_error = std::max(max_error, error);
    }
    std::cout << "FFT length: " << count << "; max cross-spectrum error: " << max_error << '\n';
    if (max_error > 1e-4) {
        throw std::runtime_error("GPU cross spectrum differs from analytic DFT");
    }
    std::cout << "FFT -> S * conj(R) -> readback: PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        run(sample_count(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
