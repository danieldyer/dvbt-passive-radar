#include "gpu_compute.hpp"
#include "gpu_frames.hpp"
#include "reference/fec_decoder.hpp"
#include "reference/waveform.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>

using namespace radar;
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
std::vector<float> decision_values(unsigned m) {
    const auto infinity = std::numeric_limits<float>::infinity();
    std::vector<float> values{0.F, -0.F, infinity, -infinity, std::numeric_limits<float>::quiet_NaN()};
    const auto scale = std::sqrt(m == 2 ? 2.F : m == 4 ? 10.F : 42.F);
    for (unsigned rank = 0; rank < (1U << (m / 2 - 1)); ++rank)
        for (float sign : {-1.F, 1.F}) {
            const float edge = sign * (2 * rank) / scale;
            values.insert(values.end(),
                          {std::nextafter(edge, -infinity), edge, std::nextafter(edge, infinity)});
        }
    return values;
}
// Exercise the production shader independently at decision boundaries and on
// erasures; an error here could otherwise be hidden by successful RS repair.
void demap_boundaries(std::shared_ptr<gpu::Context> context) {
    constexpr std::uint32_t code[] =
#include "frame_demap.inc"
        ;
    constexpr std::uint32_t label_code[] =
#include "frame_label_probe.inc"
        ;
    constexpr unsigned rows = 4, bins = 6817, stride = 2;
    validation::ReferenceModem modem;
    Tps mode;
    mode.mode = 1;
    for (unsigned m : {2U, 4U, 6U}) {
        gpu::Kernel kernel(label_code, 16, std::array{m}, context);
        const auto demap = kernel.pipeline(code, std::array{m});
        mode.constellation = (m - 2) / 2;
        const auto mapping = modem.payload_bit_indices(mode);
        const auto values = decision_values(m);
        std::vector<C> equalized(rows * bins), pilots(rows * stride);
        for (unsigned row = 0; row < rows; ++row) {
            pilots[row * stride] = {std::bit_cast<float>(row), 0};
            for (unsigned k = 0; k < bins; ++k)
                equalized[row * bins + k] = {values[k % values.size()],
                                             values[(k / values.size() + row) % values.size()]};
        }
        kernel.upload<C>(4, equalized);
        kernel.upload<C>(5, pilots);
        kernel.upload<std::uint32_t>(10, mapping);
        kernel.storage(11, rows * 6048 * m);
        kernel.storage(1, rows * bins * sizeof(std::uint32_t));
        const auto cmd = kernel.begin();
        const std::array push{rows, stride, m, rows};
        kernel.dispatch(cmd, kernel.primary(), push, rows * bins);
        kernel.barrier(cmd);
        kernel.dispatch(cmd, *demap, push, rows * 6048 * m / 4);
        kernel.finish(cmd);
        const auto *bits = static_cast<const std::uint8_t *>(kernel.view(11).mapped);
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned j = 0; j < 6048 * m; ++j) {
                const auto at = mapping[row * 6048 * m + j];
                const auto z = equalized[row * bins + at / 8];
                const auto label = m == 2 ? hard_qam<2>(z) : m == 4 ? hard_qam<4>(z) : hard_qam<6>(z);
                const auto expected = label == 255 ? 2 : (label >> (at % 8)) & 1U;
                if (bits[row * 6048 * m + j] != expected) {
                    std::cerr << "qam_bits=" << m << " z=" << std::hexfloat << z << std::defaultfloat
                              << " bit=" << at % 8 << " actual=" << unsigned(bits[row * 6048 * m + j])
                              << " expected=" << expected << '\n';
                    throw std::runtime_error("GPU demapping differs at a decision boundary or erasure");
                }
            }
    }
    std::cout << "GPU hard decision boundaries, nonfinite erasures and all permutations agree with CPU\n";
}
int main() {
    try {
        constexpr unsigned rows = 136, bins = 6817, stride = 4;
        constexpr double tau = 2 * std::numbers::pi;
        validation::ReferenceModem modem;
        Tps mode;
        mode.mode = 1;
        mode.guard = 1;
        mode.constellation = 2;
        mode.rate = 2;
        std::mt19937 random(785173);
        std::vector<std::uint8_t> bytes(2400 * 188);
        for (auto &b : bytes)
            b = random();
        for (unsigned p = 0; p < 2400; ++p)
            bytes[p * 188] = p % 8 ? 0x47 : 0xb8;
        const auto tx = modem.synthesize(bytes, mode, rows);
        auto context = std::make_shared<gpu::Context>();
        demap_boundaries(context);
        auto raw_buffer = gpu::make_buffer(context->gpu, rows * bins * sizeof(C));
        auto pilot_buffer = gpu::make_buffer(context->gpu, rows * stride * sizeof(C));
        auto *raw =
            static_cast<C *>(context->gpu.device->mapMemory(*raw_buffer.memory, 0, rows * bins * sizeof(C)));
        auto *pilots = static_cast<C *>(
            context->gpu.device->mapMemory(*pilot_buffer.memory, 0, rows * stride * sizeof(C)));
        FrameSelection selection;
        selection.frame.frames.assign(rows / 68, mode);
        auto &source = selection.frame.carriers;
        source.measured = {context, *raw_buffer.handle, rows * bins * sizeof(C), raw};
        source.equalized = source.measured;
        source.pilots = {context, *pilot_buffer.handle, rows * stride * sizeof(C), pilots};
        source.count = rows;
        source.pilot_stride = stride;
        // Noncontiguous ring gather, not the convenient identity ordering.
        for (unsigned row = 0; row < rows; ++row) {
            selection.rows.push_back(rows - row - 1);
            pilots[(rows - row - 1) * stride] = {std::bit_cast<float>(row % 4), 1};
        }
        GpuFrameStorage storage(context);
        validation::FecDecoder fec;
        selection.frame.symbols.resize(rows);
        for (unsigned row = 0; row < rows; ++row)
            selection.frame.symbols[row].source_offset_samples = std::remainder(.037 * row, 1.0);
        for (bool polynomial : {true, false}) {
            for (unsigned row = 0; row < rows; ++row)
                for (unsigned k = 0; k < bins; ++k) {
                    const double omega = tau * (int(k) - 3408) / 8192;
                    const double f = (double(k) - 3408) / 3408, t = double(row) / rows;
                    const auto h =
                        polynomial ? (std::complex<double>{1 + .2 * f + .1 * t, .15 * f * f - .1 * f * t}) *
                                         std::polar(1., -omega * 256)
                                   : std::complex<double>{1, 0} + std::polar(.8, .7 - omega * 160);
                    raw[selection.rows[row] * bins + k] =
                        tx[row * bins + k] *
                        static_cast<C>(
                            h * std::polar(1., omega * (selection.frame.symbols[row].source_offset_samples -
                                                        fft_window_margin)));
                }
            const auto frame = storage.capture(selection);
            const auto *measured = static_cast<const C *>(frame.carriers.measured.mapped);
            const auto *equalized = static_cast<const C *>(frame.carriers.equalized.mapped);
            double maximum = 0;
            for (unsigned row = 0; row < rows; ++row)
                for (unsigned k = 0; k < bins; ++k) {
                    check(measured[row * bins + k] == raw[selection.rows[row] * bins + k],
                          "Decoder equalisation altered the radar measurement");
                    if (!polynomial || (row >= 4 && row + 4 < rows))
                        maximum = std::max(maximum,
                                           double(std::abs(equalized[row * bins + k] - tx[row * bins + k])));
                }
            check(maximum < (polynomial ? .002 : .002),
                  "Pilot filtering differs from the independent analytic channel");
            if (!polynomial) {
                const auto decoded = fec.decode(frame.carriers, mode, modem);
                for (std::size_t p = 0; p < decoded.count; ++p)
                    check(decoded.packets[p].valid &&
                              std::equal(bytes.begin() + p * 188, bytes.begin() + (p + 1) * 188,
                                         decoded.packets[p].bytes.begin()),
                          "Noiseless in-GI multipath caused an FEC error floor");
            }
            std::cout << (polynomial ? "Analytic time/frequency interpolation"
                                     : "20 us echo, known-payload FEC")
                      << ": maximum constellation error " << maximum << '\n';
        }
        context->gpu.device->unmapMemory(*raw_buffer.memory);
        context->gpu.device->unmapMemory(*pilot_buffer.memory);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
