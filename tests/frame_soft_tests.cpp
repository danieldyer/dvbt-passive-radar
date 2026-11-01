#include "gpu_compute.hpp"
#include "gpu_frames.hpp"
#include "pilot_frequency_filter.hpp"
#include "reference/fec_decoder.hpp"
#include "reference/pilot_denoise_reference.hpp"
#include "reference/waveform.hpp"
#include "soft_quantizer.hpp"
#include <iostream>
#include <numbers>
#include <random>
using namespace radar;
namespace {
constexpr unsigned bins = 6817;
constexpr double tau = 2 * std::numbers::pi;
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
unsigned quantize(double x, bool minimax = false) {
    if (minimax) {
        unsigned rank = 0;
        for (auto threshold : fec::minimax_llr_thresholds)
            rank += std::abs(x) >= threshold;
        return x < 0 ? 7 - rank : 7 + rank;
    }
    return unsigned(int(std::round(std::clamp(x, -7., 7.))) + 7);
}
double boundary(unsigned a, unsigned b, bool minimax) {
    if (!minimax)
        return (int(a) + int(b)) * .5 - 7;
    const auto high = std::max(a, b);
    return high <= 7 ? -fec::minimax_llr_thresholds[7 - high] : fec::minimax_llr_thresholds[high - 8];
}
void gather(std::shared_ptr<gpu::Context> context) {
    constexpr std::uint32_t code[] =
#include "frame_demap.inc"
        ;
    constexpr unsigned rows = 5;
    const auto limits = context->gpu.physical.getProperties().limits;
    std::mt19937 random(84931);
    validation::ReferenceModem modem;
    for (unsigned qam : {2U, 4U, 6U})
        for (unsigned soft : {0U, 1U}) {
            Tps mode;
            mode.mode = 1;
            mode.constellation = (qam - 2) / 2;
            const auto mapping = modem.payload_bit_indices(mode);
            std::vector<std::uint32_t> labels(rows * bins);
            for (auto &label : labels) {
                label = soft ? 0 : random() % 17 == 0 ? 255 : random() % (1U << qam);
                if (soft)
                    for (unsigned b = 0; b < qam; ++b)
                        label |= (random() % 15) << (4 * b);
            }
            std::array<C, 2 * rows> pilots{};
            for (unsigned row = 0; row < rows; ++row)
                pilots[2 * row] = {std::bit_cast<float>((row + 3) % 4), 0};
            for (unsigned cached : {0U, 1U}) {
                if (cached &&
                    (limits.maxComputeSharedMemorySize < bins * 4 ||
                     limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 256))
                    continue;
                const unsigned threads = cached ? (limits.maxComputeWorkGroupInvocations >= 512 &&
                                                           limits.maxComputeWorkGroupSize[0] >= 512
                                                       ? 512
                                                       : 256)
                                                : 64;
                gpu::Kernel kernel(code, 16, std::array{qam, soft, cached, threads}, context);
                kernel.upload<std::uint32_t>(1, labels);
                kernel.upload<C>(5, pilots);
                kernel.upload<std::uint32_t>(10, mapping);
                kernel.storage(11, rows * 6048 * qam);
                const std::array push{rows, 2U, qam, rows};
                const auto cmd = kernel.begin();
                if (cached)
                    kernel.groups(cmd, kernel.primary(), push, rows);
                else
                    kernel.dispatch(cmd, kernel.primary(), push, rows * 6048 * qam / 4);
                kernel.finish(cmd);
                const auto *out = static_cast<const std::uint8_t *>(kernel.view(11).mapped);
                for (unsigned row = 0; row < rows; ++row)
                    for (unsigned b = 0; b < 6048 * qam; ++b) {
                        const auto index = mapping[((row + 3) % 4) * 6048 * qam + b],
                                   value = labels[row * bins + index / 8];
                        const auto expected = soft           ? (value >> (4 * (index % 8))) & 15U
                                              : value == 255 ? 2
                                                             : (value >> (index % 8)) & 1U;
                        check(out[row * 6048 * qam + b] == expected,
                              "GPU bit gather differs from exact host permutation");
                    }
            }
        }
    std::cout
        << "Cached and global bit gather match host: all constellations, hard erasures and soft metrics\n";
}
// Independent search over the actual labelled constellation, in double.
std::array<double, 6> likelihood(C z, double scale, unsigned bits) {
    std::array<double, 6> zero, one;
    zero.fill(INFINITY);
    one.fill(INFINITY);
    for (auto point : qam_constellation(bits)) {
        const auto label = bits == 2   ? hard_qam<2>(point)
                           : bits == 4 ? hard_qam<4>(point)
                                       : hard_qam<6>(point);
        const auto distance = std::norm(std::complex<double>(z) - std::complex<double>(point));
        for (unsigned b = 0; b < bits; ++b) {
            auto &v = (label >> b) & 1 ? one[b] : zero[b];
            v = std::min(v, distance);
        }
    }
    std::array<double, 6> result;
    for (unsigned b = 0; b < bits; ++b)
        result[b] = (one[b] - zero[b]) * scale;
    return result;
}
void probe(std::shared_ptr<gpu::Context> context) {
    constexpr std::uint32_t code[] =
#include "frame_soft_probe.inc"
        ;
    std::mt19937 random(57891);
    std::normal_distribution<float> normal;
    for (unsigned bits : {2U, 4U, 6U})
        for (bool minimax : {false, true}) {
            gpu::Kernel kernel(code, 4, std::array{bits, unsigned(minimax)}, context);
            std::vector<std::array<float, 4>> input;
            for (int j = -16; j <= 16; ++j)
                for (float x :
                     {j * .5F, std::nextafter(j * .5F, -INFINITY), std::nextafter(j * .5F, INFINITY)})
                    input.push_back({normal(random), normal(random), std::exp(3 * normal(random)), x});
            for (auto threshold : fec::minimax_llr_thresholds)
                for (float sign : {-1.F, 1.F})
                    for (float x : {threshold, std::nextafter(threshold, -INFINITY),
                                    std::nextafter(threshold, INFINITY)})
                        input.push_back(
                            {normal(random), normal(random), std::exp(3 * normal(random)), sign * x});
            for (unsigned j = 0; j < 10000; ++j)
                input.push_back(
                    {normal(random), normal(random), std::exp(3 * normal(random)), normal(random)});
            for (float bad : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
                input.push_back({bad, 0, 1, 0});
                input.push_back({0, bad, 1, 0});
                input.push_back({0, 0, bad, 0});
            }
            input.push_back({1, 2, 0, 0});
            kernel.upload<std::array<float, 4>>(0, input);
            kernel.storage(1, input.size() * 8);
            const auto cmd = kernel.begin();
            kernel.dispatch(cmd, kernel.primary(), std::array{unsigned(input.size())}, input.size());
            kernel.finish(cmd);
            const auto *output = static_cast<const std::array<unsigned, 2> *>(kernel.view(1).mapped);
            double worst_boundary = 0;
            for (unsigned i = 0; i < input.size(); ++i) {
                check(output[i][1] == quantize(input[i][3], minimax),
                      "GPU quantizer rounding differs from C++ round");
                if (!std::isfinite(input[i][0]) || !std::isfinite(input[i][1]) ||
                    !std::isfinite(input[i][2])) {
                    for (unsigned b = 0; b < bits; ++b)
                        check(((output[i][0] >> (4 * b)) & 15) == 7,
                              "Nonfinite GPU likelihood is not an erasure");
                    continue;
                }
                const auto llr = likelihood({input[i][0], input[i][1]}, input[i][2], bits);
                for (unsigned b = 0; b < bits; ++b) {
                    const auto actual = (output[i][0] >> (4 * b)) & 15;
                    if (actual != quantize(llr[b], minimax)) {
                        const double edge = boundary(actual, quantize(llr[b], minimax), minimax);
                        check(std::abs(int(actual) - int(quantize(llr[b], minimax))) == 1,
                              "GPU max-log differs by more than one quantization level");
                        worst_boundary = std::max(worst_boundary, std::abs(llr[b] - edge));
                    }
                }
            }
            check(worst_boundary < 2e-4, "GPU max-log disagrees away from a rounding boundary");
            std::cout << "Soft demapper QAM=" << (1U << bits)
                      << " max rounding-boundary discrepancy=" << worst_boundary << '\n';
        }
}
std::vector<float> noise_spectrum(std::span<const C> raw, unsigned rows, std::span<const double> offsets) {
    std::array<double, 128> error{}, power{}, count{};
    for (unsigned s = 4; s < rows; ++s) {
        std::complex<double> cross{};
        const auto mod = s % 4;
        for (unsigned k = 3 * mod; k < bins; k += 12)
            cross += std::complex<double>(raw[s * bins + k]) *
                     std::conj(std::complex<double>(raw[(s - 4) * bins + k])) *
                     std::polar(1., tau * (int(k) - 3408) * (offsets[s - 4] - offsets[s]) / 8192);
        const auto phase = std::polar(1., std::arg(cross));
        const double pilots = (6816 - 3 * mod) / 12 + 1;
        for (unsigned b = 0; b < 128; ++b) {
            const auto first = b * bins / 128, last = (b + 1) * bins / 128;
            for (unsigned k = first + (3 * mod + 12 - first % 12) % 12; k < last; k += 12) {
                const std::complex<double> a = raw[s * bins + k], v = raw[(s - 4) * bins + k];
                const auto e =
                    a -
                    v * phase * std::polar(1., tau * (int(k) - 3408) * (offsets[s] - offsets[s - 4]) / 8192);
                error[b] += std::norm(e) / (2 - 1 / pilots);
                power[b] += .5 * (std::norm(a) + std::norm(v));
                count[b] += 1;
            }
        }
    }
    std::array<double, 128> density;
    for (unsigned b = 0; b < 128; ++b)
        density[b] = std::max({error[b] / count[b], power[b] / count[b] * 1e-10, 1e-30});
    std::vector<float> result(bins);
    for (unsigned k = 0; k < bins; ++k) {
        const double p = std::clamp((k + .5) * 128 / bins - .5, 0., 127.);
        const auto lo = unsigned(p), hi = std::min(lo + 1, 127U);
        result[k] = std::lerp(density[lo], density[hi], p - lo);
    }
    return result;
}
} // namespace
void phase_prefix(std::shared_ptr<gpu::Context> context) {
    constexpr std::uint32_t code[] =
#include "frame_phase_scan.inc"
        ;
    gpu::Kernel kernel(code, 16, {}, context);
    std::mt19937 random(897655);
    std::uniform_real_distribution<double> angle(-3.1, 3.1);
    for (unsigned count : {8U, 136U, 1020U}) {
        const auto base = count * 2273;
        std::vector<C> data(base + 2 * count);
        std::vector<std::complex<double>> exact(count);
        std::complex<double> phase{1, 0};
        for (unsigned s = 0; s < count; ++s) {
            data[base + 2 * s] = s ? static_cast<C>(std::polar(1., angle(random))) : C{1, 0};
            data[base + 2 * s + 1] = {float(s), -float(s)};
            phase *= std::complex<double>(data[base + 2 * s]);
            phase /= std::abs(phase);
            exact[s] = phase;
        }
        kernel.upload<C>(9, data);
        const auto cmd = kernel.begin();
        kernel.groups(cmd, kernel.primary(), std::array{count, 4U, 6U, count}, 1);
        kernel.finish(cmd);
        const auto *result = static_cast<const C *>(kernel.view(9).mapped);
        double error = 0;
        for (unsigned s = 0; s < count; ++s) {
            check(result[base + 2 * s + 1] == data[base + 2 * s + 1],
                  "Prefix scan changed repeated-pilot noise phase");
            check(std::abs(std::norm(result[base + 2 * s]) - 1) < 5e-7, "Prefix phase lost unit modulus");
            for (unsigned r = s > 32 ? s - 32 : 0; r <= std::min(count - 1, s + 32); ++r) {
                const auto expected = exact[s] * std::conj(exact[r]);
                const auto actual = result[base + 2 * s] * std::conj(result[base + 2 * r]);
                error = std::max(error, std::abs(std::complex<double>(actual) - expected));
            }
        }
        std::cout << "Pilot phase prefix count=" << count << " maximum local complex error=" << error << '\n';
        check(error < 3e-6, "Parallel phase prefix disagrees with independent double products");
    }
}
void pilot_cache(std::shared_ptr<gpu::Context> context, bool benchmark = false) {
    constexpr std::uint32_t global_code[] =
#include "frame_pilots.inc"
        ;
    constexpr std::uint32_t tiled_code[] =
#include "frame_pilots_tiled.inc"
        ;
    const auto limits = context->gpu.physical.getProperties().limits;
    if (limits.maxComputeSharedMemorySize < 80 * 16 * sizeof(C) ||
        limits.maxComputeWorkGroupInvocations < 256 || limits.maxComputeWorkGroupSize[0] < 16 ||
        limits.maxComputeWorkGroupSize[1] < 16)
        return;
    validation::ReferenceModem modem;
    const auto pattern = modem.pilot_pattern();
    std::mt19937 random(48131);
    std::normal_distribution<float> normal;
    for (unsigned count : {8U, 33U, 68U, 136U, 1020U}) {
        if (benchmark && count != 1020)
            continue;
        const unsigned base = count * (2 * 2273 + 2) + 128, end = count * (3 * 2273 + 2) + 128;
        std::vector<C> data(end + 1), table(3 * bins + 192 + 49 + 177), pilots(2 * count);
        std::vector<unsigned> rows(count);
        for (unsigned k = 0; k < bins; ++k)
            table[k] = {.75F, 0};
        for (unsigned i = 0; i < 177; ++i) {
            const auto k = pattern.continual[i];
            table[2 * bins + 192 + k] = {1, 0};
            table[3 * bins + 192 + 49 + i] = {std::bit_cast<float>(k), 0};
        }
        for (unsigned r = 0; r < count; ++r) {
            rows[r] = (r + 3) % count;
            pilots[2 * rows[r]] = {std::bit_cast<float>((r + 1) % 4), 0};
            data[count * 2273 + 2 * r] = static_cast<C>(std::polar(1., .15 * r));
            for (unsigned j = 0; j < 2273; ++j)
                data[base + r * 2273 + j] = {normal(random), normal(random)};
        }
        for (unsigned radius : {4U, 8U, 16U, 32U, 64U})
            for (unsigned adaptive : {0U, 1U}) {
                if (limits.maxComputeSharedMemorySize < (16 + 2 * std::max(32U, radius)) * 16 * sizeof(C))
                    continue;
                gpu::Kernel kernel(global_code, 16, std::array{radius, adaptive}, context);
                const auto tiled = kernel.pipeline(tiled_code, std::array{radius, adaptive, 0U});
                const auto continual = kernel.pipeline(tiled_code, std::array{radius, adaptive, 1U});
                data[end] = {float(radius), 0};
                kernel.upload<C>(7, table);
                kernel.upload<C>(2, pilots);
                kernel.upload<unsigned>(6, rows);
                kernel.upload<C>(9, data);
                const std::array push{count, 2U, 6U, count};
                auto cmd = kernel.begin();
                kernel.dispatch(cmd, kernel.primary(), push, count * 2273);
                kernel.finish(cmd);
                const auto *output = static_cast<const C *>(kernel.view(9).mapped);
                std::vector<C> expected(output, output + count * 2273);
                // Independent double-precision sums, not the shared GPU fit:
                // sample the complete grid and explicitly include the end rows.
                for (unsigned s = 0; s < count; ++s)
                    for (unsigned j = 0; j < 2273; j += (s == 0 || s + 1 == count ? 1U : 137U)) {
                        const bool continual = table[2 * bins + 192 + 3 * j].real() != 0;
                        const int begin = std::max(0, int(s) - int(radius)),
                                  end = std::min(int(count) - 1, int(s) + int(radius));
                        double n = 0, t = 0, tt = 0;
                        std::complex<double> y{}, ty{};
                        for (int r = begin; r <= end; ++r)
                            if (continual || unsigned(r + 1) % 4 == j % 4) {
                                const double dt = r - int(s);
                                const std::complex<double> z = data[base + r * 2273 + j];
                                ++n;
                                t += dt;
                                tt += dt * dt;
                                y += z;
                                ty += dt * z;
                            }
                        const double determinant = n * tt - t * t;
                        const auto fit = (determinant > 0 ? (tt * y - t * ty) / determinant : y / n) *
                                         std::complex<double>(data[count * 2273 + 2 * s]) * .75;
                        check(std::abs(std::complex<double>(expected[s * 2273 + j]) - fit) <
                                  3e-6 * (1 + std::abs(fit)),
                              "GPU pilot fit differs from independent time-moment sums");
                    }
                kernel.upload<C>(9, data);
                cmd = kernel.begin();
                kernel.groups(cmd, *tiled, push, (2273 + 15) / 16, (count + 15) / 16);
                kernel.barrier(cmd);
                kernel.groups(cmd, *continual, push, (177 + 15) / 16, (count + 15) / 16);
                kernel.finish(cmd);
                output = static_cast<const C *>(kernel.view(9).mapped);
                for (unsigned i = 0; i < count * 2273; ++i)
                    check(std::abs(output[i] - expected[i]) < 3e-6F * (1 + std::abs(expected[i])),
                          "Cached pilot fit differs from global fit");
                for (unsigned i = base; i < end; ++i)
                    check(output[i] == data[i], "Cached pilot fit modified aligned observations");
                if (benchmark) {
                    ComputeTimings global_time, cached_time;
                    for (unsigned repeat = 0; repeat < 100; ++repeat) {
                        cmd = kernel.begin(&global_time);
                        kernel.dispatch(cmd, kernel.primary(), push, count * 2273);
                        kernel.finish(cmd);
                        cmd = kernel.begin(&cached_time);
                        kernel.groups(cmd, *tiled, push, (2273 + 15) / 16, (count + 15) / 16);
                        kernel.barrier(cmd);
                        kernel.groups(cmd, *continual, push, (177 + 15) / 16, (count + 15) / 16);
                        kernel.finish(cmd);
                    }
                    std::cout << "PILOT_BENCH," << count << ',' << radius << ',' << adaptive << ','
                              << global_time.gpu * 10 << ',' << cached_time.gpu * 10 << '\n';
                }
            }
        std::cout << "Cached/global pilot fit equivalent count=" << count << '\n';
    }
}
void pilot_risk(std::shared_ptr<gpu::Context> context) {
    constexpr std::uint32_t risk_code[] =
#include "frame_pilot_risk.inc"
        ;
    constexpr std::uint32_t select_code[] =
#include "frame_pilot_select.inc"
        ;
    validation::ReferenceModem modem;
    const auto pattern = modem.pilot_pattern();
    std::array<bool, bins> continual{};
    for (auto k : pattern.continual)
        continual[k] = true;
    std::mt19937 random(19432);
    std::normal_distribution<float> normal;
    for (unsigned maximum : {32U, 64U})
        for (unsigned count : {1U, 2U, 8U, 64U, 68U, 128U, 136U, 1020U})
            for (double hz : {0., 5., 20.}) {
                gpu::Kernel kernel(risk_code, 16, std::array{maximum}, context);
                const auto select = kernel.pipeline(select_code, std::array{maximum});
                const unsigned base = count * (2 * 2273 + 2) + 128, end = count * (3 * 2273 + 2) + 128;
                std::vector<C> channel(end + 1), table(3 * bins + 192), pilots(2 * count);
                std::vector<unsigned> rows(count);
                for (auto k : pattern.continual)
                    table[2 * bins + 192 + k] = {1, 0};
                for (unsigned r = 0; r < count; ++r) {
                    rows[r] = (r + 31) % count;
                    pilots[2 * rows[r]] = {std::bit_cast<float>((r + 1) % 4), 0};
                }
                for (unsigned r = 0; r < count; ++r)
                    for (unsigned j = 0; j < 2273; ++j) {
                        const auto h =
                            std::complex<double>{1, 0} + std::polar(.99, tau * hz * .001088 * r + .13 * j);
                        channel[base + r * 2273 + j] =
                            static_cast<C>(h) + .07F * C{normal(random), normal(random)};
                    }
                const unsigned margin = maximum == 64 && count > 128 ? 64 : 32,
                               choices = margin == 64 ? 5 : 4;
                std::array<double, 5> expected{};
                for (unsigned j = 0; j < 2273; ++j)
                    if (!continual[3 * j])
                        for (unsigned r = margin + (j % 4 + 3) % 4; r + margin < count; r += 4) {
                            std::complex<double> sum{};
                            for (unsigned d = 1; d <= margin / 4; ++d) {
                                sum += std::complex<double>(channel[base + (r - 4 * d) * 2273 + j]) +
                                       std::complex<double>(channel[base + (r + 4 * d) * 2273 + j]);
                                if (std::has_single_bit(d))
                                    expected[std::bit_width(d) - 1] +=
                                        std::norm(std::complex<double>(channel[base + r * 2273 + j]) -
                                                  sum / double(2 * d));
                            }
                        }
                kernel.upload<C>(9, channel);
                kernel.upload<C>(7, table);
                kernel.upload<C>(2, pilots);
                kernel.upload<unsigned>(6, rows);
                const std::array push{count, 2U, 6U, count};
                const auto cmd = kernel.begin();
                kernel.groups(cmd, kernel.primary(), push, 2273);
                kernel.barrier(cmd);
                kernel.groups(cmd, *select, push, 1);
                kernel.finish(cmd);
                const auto *output = static_cast<const C *>(kernel.view(9).mapped);
                std::array<double, 5> measured{};
                check(std::equal(channel.begin() + base, channel.begin() + end, output + base),
                      "Pilot risk selection overwrote aligned observations");
                if (count <= 64) {
                    check(output[end].real() == 8, "Short pilot batch did not retain radius 8");
                    continue;
                }
                for (unsigned j = 0; j < 2273; ++j) {
                    measured[0] += output[j].real();
                    measured[1] += output[j].imag();
                    measured[2] += output[2273 + j].real();
                    measured[3] += output[2273 + j].imag();
                    if (maximum == 64)
                        measured[4] += output[2 * 2273 + j].real();
                }
                for (unsigned j = 0; j < choices; ++j)
                    check(std::abs(measured[j] - expected[j]) < 2e-6 * (1 + expected[j]),
                          "GPU pilot prediction risk differs from independent double sums");
                const unsigned chosen =
                    count <= 64 ? 8
                                : 4U << (std::min_element(expected.begin(), expected.begin() + choices) -
                                         expected.begin());
                check(output[end].real() == chosen, "GPU pilot window selection differs from reference");
                std::cout << "Pilot risk maximum=" << maximum << " count=" << count << " echo_hz=" << hz
                          << " radius=" << chosen << '\n';
            }
}
int main(int argc, char **argv) {
    try {
        auto context = std::make_shared<gpu::Context>();
        if (argc == 2 && std::string_view(argv[1]) == "--pilot-kernel-benchmark") {
            pilot_cache(context, true);
            return 0;
        }
        phase_prefix(context);
        pilot_cache(context);
        pilot_risk(context);
        if (argc == 2 && std::string_view(argv[1]) == "--pilots-only")
            return 0;
        probe(context);
        gather(context);
        constexpr unsigned rows = 136, stride = 4;
        validation::ReferenceModem modem;
        std::mt19937 random(954312);
        std::normal_distribution<float> normal;
        std::vector<std::uint8_t> bytes(3000 * 188);
        for (auto &b : bytes)
            b = random();
        for (unsigned p = 0; p < 3000; ++p)
            bytes[188 * p] = p % 8 ? 0x47 : 0xb8;
        // Use ordinary host-visible buffers, permuted like a wrapping source ring.
        auto raw_buffer = gpu::make_buffer(context->gpu, rows * bins * sizeof(C));
        auto pilot_buffer = gpu::make_buffer(context->gpu, rows * stride * sizeof(C));
        auto *raw =
            static_cast<C *>(context->gpu.device->mapMemory(*raw_buffer.memory, 0, rows * bins * sizeof(C)));
        auto *pilots = static_cast<C *>(
            context->gpu.device->mapMemory(*pilot_buffer.memory, 0, rows * stride * sizeof(C)));
        FrameSelection selection;
        selection.frame.symbols.resize(rows);
        auto &source = selection.frame.carriers;
        source.measured = {context, *raw_buffer.handle, rows * bins * sizeof(C), raw};
        source.equalized = source.measured;
        source.pilots = {context, *pilot_buffer.handle, rows * stride * sizeof(C), pilots};
        source.count = rows;
        source.pilot_stride = stride;
        std::vector<double> offsets(rows);
        for (unsigned s = 0; s < rows; ++s) {
            selection.rows.push_back(rows - 1 - s);
            pilots[(rows - 1 - s) * stride] = {std::bit_cast<float>(s % 4), 1};
            offsets[s] = selection.frame.symbols[s].source_offset_samples = std::remainder(.137 * s, 1.0);
        }
        for (unsigned guard = 0; guard < 4; ++guard)
            for (unsigned bits : {2U, 4U, 6U}) {
                Tps mode;
                mode.mode = 1;
                mode.guard = guard;
                mode.constellation = (bits - 2) / 2;
                mode.rate = 2;
                selection.frame.frames.assign(rows / 68, mode);
                const auto tx = modem.synthesize(bytes, mode, rows);
                const auto mapping = modem.payload_bit_indices(mode);
                std::vector<C> ordered(rows * bins);
                // Coloured noise, deep two-path fading, known sample offsets, CPE and
                // smoothly changing common phase: all are removed only in the decoder.
                for (unsigned s = 0; s < rows; ++s)
                    for (unsigned k = 0; k < bins; ++k) {
                        const double omega = tau * (int(k) - 3408) / 8192, delay = 12. + guard * 32;
                        const auto h = (std::complex<double>{1, 0} + std::polar(.9, .7 - omega * delay)) *
                                       std::polar(1., .013 * s + .00003 * s * s +
                                                          omega * (offsets[s] - fft_window_margin));
                        const float sigma = std::sqrt(.003 * std::exp(2. * k / bins) / 2);
                        raw[selection.rows[s] * bins + k] = ordered[s * bins + k] =
                            tx[s * bins + k] * static_cast<C>(h) + sigma * C{normal(random), normal(random)};
                    }
                const auto noise = noise_spectrum(ordered, rows, offsets);
                validation::PilotDenoiseReference reference(modem.pilot_pattern(), guard);
                {
                    constexpr unsigned radius = 8;
                    constexpr bool smooth = true, minimax = true, adaptive = true;
                    const unsigned selected = reference.select_radius(ordered, rows, offsets, 32);
                    GpuFrameStorage storage(context);
                    const auto frame = storage.capture(selection);
                    check(frame.carriers.coded_metric == CodedMetric::minimax_likelihood,
                          "Compact metric type lost");
                    const auto h =
                        reference.estimate(ordered, rows, selected, offsets,
                                           smooth ? std::span<const float>{pilot_frequency_filters[guard]}
                                                  : std::span<const float>{});
                    const auto *measured = static_cast<const C *>(frame.carriers.measured.mapped);
                    const auto *eq = static_cast<const C *>(frame.carriers.equalized.mapped);
                    const auto *metrics = static_cast<const std::uint8_t *>(frame.carriers.coded_bits.mapped);
                    double worst_equalized = 0, worst_boundary = 0, worst_channel_noise = 0;
                    std::size_t differences = 0;
                    std::vector<std::uint8_t> cpu_metrics(rows * 6048 * bits);
                    std::vector<std::array<double, 6>> llrs(rows * bins);
                    for (unsigned i = 0; i < rows * bins; ++i) {
                        check(measured[i] == ordered[i], "Soft decoder modified scientific carrier");
                        const auto z = ordered[i] / h[i];
                        worst_equalized =
                            std::max(worst_equalized, double(std::abs(eq[i] - z)) / (1 + std::abs(z)));
                        if (std::norm(eq[i]) > 1e-20F)
                            worst_channel_noise =
                                std::max(worst_channel_noise,
                                         double(std::norm(ordered[i] / eq[i] - h[i])) / noise[i % bins]);
                        llrs[i] = likelihood(z, std::norm(h[i]) / noise[i % bins], bits);
                    }
                    for (unsigned s = 0; s < rows; ++s)
                        for (unsigned j = 0; j < 6048 * bits; ++j) {
                            const auto at = mapping[(s % 4) * 6048 * bits + j], i = s * 6048 * bits + j;
                            const double llr = llrs[s * bins + at / 8][at % 8];
                            const auto expected = quantize(llr, minimax), actual = unsigned(metrics[i]);
                            cpu_metrics[i] = expected;
                            if (actual != expected) {
                                ++differences;
                                check(std::abs(int(actual) - int(expected)) == 1,
                                      "GPU likelihood differs by multiple levels");
                                worst_boundary = std::max(
                                    worst_boundary, std::abs(llr - boundary(actual, expected, minimax)));
                            }
                        }
                    std::cout << "guard=" << guard << " bits=" << bits << " radius=" << radius
                              << " frequency=" << smooth << " minimax=" << minimax
                              << " eq_relative=" << worst_equalized
                              << " channel_error_over_noise=" << worst_channel_noise
                              << " metric_boundary=" << worst_boundary << " different=" << differences
                              << std::endl;
                    // Equalized relative error is ill-conditioned at deep channel zeros.
                    // Bound the induced unit-energy constellation error by 1e-4 of
                    // receiver noise power (<0.00044 dB if treated as additive error).
                    // Likelihood boundaries and complete decoded packets are tested too.
                    check(worst_channel_noise < 1e-4, "GPU pilot error is not negligible relative to noise");
                    check(worst_boundary < .05,
                          "GPU likelihood differs by more than 5% of a quantizer step at a boundary");
                    check(double(differences) / cpu_metrics.size() < .0003,
                          "Too many CPU/GPU likelihood differences");
                    validation::FecDecoder fec;
                    const auto gpu_decoded = fec.decode(frame.carriers, mode, modem);
                    const auto cpu_bytes = validation::ReferenceViterbi{}.decode_soft(
                        cpu_metrics, mode.rate, 0,
                        minimax ? validation::SoftMetric::minimax : validation::SoftMetric::uniform);
                    for (unsigned p = 0; p < gpu_decoded.count; ++p) {
                        fec::DvbtRs::Word word;
                        for (unsigned j = 0; j < 204; ++j) {
                            const auto i = 2244 + p * 204 + j;
                            word[j] = cpu_bytes[i - (11 - i % 12) * 204];
                        }
                        const bool valid =
                            fec::DvbtRs::decode(word) >= 0 && (word[0] == 0x47 || word[0] == 0xb8);
                        if (valid)
                            check(gpu_decoded.packets[p].valid,
                                  "Production feedback lost a valid reference packet");
                        if (valid)
                            check(word == gpu_decoded.packets[p].bytes, "GPU/CPU soft packet bytes differ");
                    }
                    if (argc > 1 && guard == 1 && bits == 6) {
                        ComputeTimings timing;
                        for (unsigned trial = 0; trial < 100; ++trial)
                            storage.capture(selection, &timing);
                        std::cout << "BENCHMARK soft_frame radius=" << radius << " gpu_ms=" << timing.gpu * 10
                                  << std::endl;
                    }
                }
            }
        context->gpu.device->unmapMemory(*raw_buffer.memory);
        context->gpu.device->unmapMemory(*pilot_buffer.memory);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
