#include "../support/video/profile_history.hpp"
#include "gpu_focus.hpp"
#include "reference/profile_history.hpp"
#include <iostream>
#include <numbers>
#include <random>
using namespace radar;
void require(bool v, const char *text) {
    if (!v)
        throw std::runtime_error(text);
}
int main() {
    try {
        auto context = std::make_shared<gpu::Context>();
        std::mt19937 random(83573);
        std::normal_distribution<float> normal;
        constexpr double dt = .001088, integration = .5;
        for (unsigned size : {1024U, 3072U, 4096U})
            for (bool erasures : {false, true}) {
                video::ProfileHistory gpu(context, integration, dt, .16, erasures);
                video::ReferenceProfileHistory cpu(integration, dt, .16, erasures);
                GpuFocus direct(context), cached(context);
                unsigned snapshots = 0;
                // Short final batches, preloaded future rows, multiple ring wraps,
                // decoder erasures, a time gap, a segment boundary and all-zero data.
                std::uint64_t row = 0;
                for (unsigned batch = 0; batch < 23; ++batch) {
                    const unsigned count = batch % 2 ? 173 : 257;
                    std::vector<C> values(2ULL * count * size);
                    for (auto &value : values)
                        value = {.01F * normal(random), .01F * normal(random)};
                    if (batch == 22)
                        std::ranges::fill(values, C{});
                    gpu.stage(values, count, size);
                    for (unsigned n = 0; n < count; ++n, ++row) {
                        const double time = 42 + row * dt + (batch >= 12 ? 1 : 0);
                        const auto segment = batch >= 18 ? 1 : 0;
                        const bool valid = !(batch >= 7 && batch <= 10 && n % 97 == 0);
                        const double info = 1.1 + .3 * (batch % 3);
                        const auto ready = gpu.append(time, info, valid, segment);
                        const auto expected =
                            cpu.append(time, info, valid, segment, {values.data() + n * size, size},
                                       {values.data() + (count + n) * size, size});
                        require(ready == expected, "GPU history changed gap/erasure/cadence admission");
                        if (!ready)
                            continue;
                        require(gpu.first_time() == cpu.first_time(), "GPU history changed the phase epoch");
                        std::vector<C> moments;
                        std::vector<double> information, expected_information;
                        cpu.snapshot(moments, expected_information);
                        const auto spectrum = gpu.snapshot(information);
                        require(information == expected_information,
                                "GPU history changed observation weights");
                        const auto a =
                            direct.calculate(moments, cpu.required(), dt, 191.625e6, cpu.first_time(),
                                             {1000, 20, true}, nullptr, information);
                        const auto b = cached.calculate_spectral_device(
                            spectrum, gpu.required(), dt, 191.625e6, gpu.first_time(), {1000, 20, true},
                            nullptr, information);
                        const auto *actual = static_cast<const C *>(b.values.mapped);
                        require(a.phase_epoch == b.map.phase_epoch &&
                                    a.values.size() == b.map.ranges * b.map.frequencies,
                                "Cached FFT changed output coordinates");
                        double error = 0, power = 0;
                        for (std::size_t i = 0; i < a.values.size(); ++i) {
                            error += std::norm(std::complex<double>(a.values[i] - actual[i]));
                            power += std::norm(std::complex<double>(a.values[i]));
                        }
                        require(std::sqrt(error / std::max(power, 1e-30)) < 5e-6,
                                "Cached range FFT altered the coherent operator");
                        ++snapshots;
                    }
                }
                require(snapshots > 10, "Insufficient ring-wrap and discontinuity coverage");
                std::cout << "Exact archived GPU history: size=" << size << " erasures=" << erasures
                          << " snapshots=" << snapshots << '\n';
            }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
