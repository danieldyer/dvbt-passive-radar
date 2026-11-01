#include "dsp.hpp" // Independent CPU minimum-phase reference.
#include "gpu_compute.hpp"
#include "gpu_response.hpp"
#include "reference/dsp_reference.hpp"
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>

using namespace radar;
void check(bool b, const char *message) {
    if (!b)
        throw std::runtime_error(message);
}
int main(int argc, char **argv) {
    try {
        constexpr unsigned bins = 6817;
        constexpr double tau = 2 * std::numbers::pi;
        const bool benchmark = argc > 1 && std::string_view(argv[1]) == "--benchmark";
        const unsigned rows = benchmark ? (argc > 2 ? std::stoul(argv[2]) : 1020) : 136;
        check(rows > 0 && rows <= 1020, "Invalid response benchmark row count");
        const auto count = rows * bins;
        auto context = std::make_shared<gpu::Context>();
        auto y_storage = gpu::make_buffer(context->gpu, count * sizeof(C));
        auto x_storage = gpu::make_buffer(context->gpu, count * sizeof(C));
        auto v_storage = gpu::make_buffer(context->gpu, count * sizeof(std::uint32_t));
        auto *y = static_cast<C *>(context->gpu.device->mapMemory(*y_storage.memory, 0, count * sizeof(C)));
        auto *x = static_cast<C *>(context->gpu.device->mapMemory(*x_storage.memory, 0, count * sizeof(C)));
        auto *valid = static_cast<std::uint32_t *>(
            context->gpu.device->mapMemory(*v_storage.memory, 0, count * sizeof(std::uint32_t)));
        const gpu::BufferView measured{context, *y_storage.handle, count * sizeof(C), y};
        const ReferenceBatch reference{{context, *x_storage.handle, count * sizeof(C), x},
                                       {context, *v_storage.handle, count * sizeof(std::uint32_t), valid},
                                       rows};
        std::mt19937 rng(93513);
        const auto points = qam_constellation(6);
        if (!benchmark) {
            // Independent conditional-origin model, with noisy deep fades.
            // Their phases must not choose the branch for stronger carriers.
            std::normal_distribution<float> noise(0, .02F);
            std::vector<C> channel(bins);
            std::vector<ResponseStatistics> statistics(bins);
            ResponseEstimate calibration;
            calibration.response.mapped = channel.data();
            calibration.response.bytes = channel.size() * sizeof(C);
            calibration.statistics.mapped = statistics.data();
            calibration.statistics.bytes = statistics.size() * sizeof(ResponseStatistics);
            for (double delay : {127.3125, -89.625}) {
                for (unsigned k = 0; k < bins; ++k) {
                    const double amplitude = 1e-7 + std::pow(std::abs(std::sin(k * .013)), 6);
                    channel[k] = static_cast<C>(std::polar(amplitude, .3 * std::sin(k * .002)));
                    const float energy = 100 + float(k % 71);
                    statistics[k] = {
                        channel[k] * energy *
                                static_cast<C>(std::polar(1.0, 2.1 - tau * (int(k) - 3408) * delay / 8192)) +
                            std::sqrt(energy) * C{noise(rng), noise(rng)},
                        energy, 1};
                }
                const double fitted = GpuResponse::direct_origin_adjustment(calibration);
                check(std::abs(fitted - delay) < .002,
                      "Faded carrier phase changed the direct-origin likelihood branch");
                std::cout << "Faded-carrier conditional direct origin error=" << fitted - delay
                          << " samples\n";
            }
        }
        for (unsigned i = 0; i < count; ++i) {
            x[i] = points[rng() % 64];
            valid[i] = (rng() % 31) != 0;
        }
        GpuResponse gpu(context);
        double largest_error = 0;
        for (unsigned order : {0U, 1U, 8U}) {
            if (benchmark && order != 8)
                continue;
            const double origin = 128.375;
            std::vector<C> rotated(count), masked_y(count);
            for (unsigned i = 0; i < count; ++i) {
                const double omega = tau * (int(i % bins) - 3408) / 8192;
                const auto h =
                    C{.7F, .4F} *
                    std::exp(order ? C{.18F, .07F} * static_cast<C>(std::polar(1.0, -omega)) : C{});
                const auto r = x[i] * static_cast<C>(std::polar(1.0, -omega * origin));
                y[i] = h * r;
                if (valid[i]) {
                    rotated[i] = r;
                    masked_y[i] = y[i];
                } else
                    y[i] = C{1e6F, -2e6F};
            }
            auto estimate = gpu.fit(measured, reference, origin, order);
            auto output = gpu.apply(measured, reference, estimate);
            if (benchmark) {
                ComputeTimings fit_time, apply_time;
                fit_time.detailed = true;
                constexpr unsigned repeats = 25;
                for (unsigned repeat = 0; repeat < repeats; ++repeat) {
                    estimate = gpu.fit(measured, reference, origin, order, 16, &fit_time);
                    output = gpu.apply(measured, reference, estimate, &apply_time);
                }
                const double scale = 1000 / repeats / (rows * 8704.0 / 8e6);
                std::cout << "rows,fit_ms_per_input_second,statistics_ms_per_input_second,solver_ms_per_"
                             "input_second,apply_ms_per_input_second\n"
                          << rows << ',' << fit_time.gpu * scale << ',' << fit_time.stages[0] * scale << ','
                          << fit_time.stages[1] * scale << ',' << apply_time.gpu * scale << '\n';
                continue;
            }
            const auto cpu = fit_minimum_phase(masked_y, rotated, bins, order, 16);
            const auto *h = static_cast<const C *>(estimate.response.mapped);
            const auto *r = static_cast<const C *>(output.reference.mapped);
            const auto *cross = static_cast<const C *>(output.cross.mapped);
            const auto *normal = static_cast<const C *>(output.normalized.mapped);
            double error = 0;
            for (unsigned k = 0; k < bins; ++k)
                error = std::max(error, double(std::abs(h[k] - cpu.response[k]) / std::abs(cpu.response[k])));
            largest_error = std::max(largest_error, error);
            std::cout << "Order " << order << ": CPU/GPU max relative response error=" << error << '\n';
            check(error < 3e-5, "GPU minimum-phase response differs from double-precision reference");
            double truth_error = 0, truth_power = 0;
            for (unsigned k = 0; k < bins; ++k) {
                const double omega = tau * (int(k) - 3408) / 8192;
                const auto truth =
                    C{.7F, .4F} *
                    std::exp(order ? C{.18F, .07F} * static_cast<C>(std::polar(1., -omega)) : C{});
                truth_error += std::norm(h[k] - truth);
                truth_power += std::norm(truth);
            }
            check(truth_error / truth_power < 1e-9,
                  "Exactly representable response retains a cancellation floor");
            check(std::abs(estimate.relative_error - cpu.relative_error) < 2e-6,
                  "GPU fit error statistic disagrees with CPU");
            for (unsigned i = 0; i < count; ++i) {
                if (!valid[i]) {
                    check(r[i] == C{} && cross[i] == C{} && normal[i] == C{},
                          "Invalid carrier leaked into residual");
                    continue;
                }
                check(std::abs(r[i] - rotated[i]) < 3e-7, "Explicit reference-origin rotation changed phase");
                const C predicted = h[i % bins] * r[i], residual = y[i] - predicted;
                check(std::abs(cross[i] - residual * std::conj(r[i])) < 5e-7,
                      "Resident residual cross product changed phase");
                check(std::abs(normal[i] - residual / predicted) < 2e-6,
                      "Target/direct normalization changed phase");
            }
            const std::vector<C> full_reference(r, r + count);
            const auto reference_only = gpu.apply(measured, reference, estimate, nullptr, nullptr, false);
            check(!reference_only.cross.bytes && !reference_only.normalized.bytes,
                  "Unrequested residual spectra were materialised");
            check(std::equal(full_reference.begin(), full_reference.end(),
                             static_cast<const C *>(reference_only.reference.mapped)),
                  "Omitting unused residual spectra changed reference phase or validity");
        }
        if (!benchmark) {
            // Two independently drifting sources share the same receive grid.
            // Integer FFT-window steering wraps their local phase, not the echo.
            std::vector<double> offsets(rows);
            std::fill_n(valid, count, 1U);
            for (double rate : {-.019, .027}) {
                for (unsigned row = 0; row < rows; ++row)
                    offsets[row] = std::remainder(rate * row, 1.0);
                for (unsigned i = 0; i < count; ++i) {
                    const double omega = tau * (int(i % bins) - 3408) / 8192;
                    y[i] = x[i] * static_cast<C>(std::polar(.8, .9 - omega * (128.375 - offsets[i / bins])));
                }
                const auto estimate = gpu.fit(measured, reference, 128.375, 8, 16, nullptr, nullptr, offsets);
                const auto aligned = gpu.apply(measured, reference, estimate);
                const auto *residual = static_cast<const C *>(aligned.normalized.mapped);
                for (unsigned i = 0; i < count; ++i)
                    check(std::abs(residual[i]) < 2e-5,
                          "Independent source timing changed direct-path phase");
            }
            bool ill_conditioned = false;
            try {
                gpu.fit(measured, reference, 0, 32);
            } catch (const std::invalid_argument &) {
                ill_conditioned = true;
            }
            check(ill_conditioned, "Unidentifiable partial-band response order was accepted");
            // A pure all-pass delay must remain outside the estimated response.
            std::fill_n(valid, count, 1U);
            for (unsigned i = 0; i < count; ++i)
                y[i] = x[i] * static_cast<C>(std::polar(.8, -tau * (int(i % bins) - 3408) * 8 / 8192 + .9));
            auto estimate = gpu.fit(measured, reference, 0, 8);
            const auto *h = static_cast<const C *>(estimate.response.mapped);
            for (unsigned k = 1; k < bins; ++k)
                check(std::abs(h[k] - h[0]) < 2e-5, "Minimum-phase estimator absorbed an all-pass delay");
            check(std::abs(GpuResponse::direct_origin_adjustment(estimate) - 8) < 2e-5,
                  "Explicit origin calibration has wrong sign/units");
            estimate = gpu.fit(measured, reference, 8, 8);
            const auto aligned = gpu.apply(measured, reference, estimate);
            const auto *residual = static_cast<const C *>(aligned.normalized.mapped);
            for (unsigned i = 0; i < count; ++i)
                check(std::abs(residual[i]) < 3e-6, "Explicit origin did not align pure delayed reference");
            // Estimate on a direct path, then inject an echo. Independent TX/gain
            // phases must cancel in the target/direct ratio; echo phase must not.
            double echo_error = 0;
            for (double phase : {-1.7, .93}) {
                for (unsigned i = 0; i < count; ++i) {
                    const double omega = tau * (int(i % bins) - 3408) / 8192;
                    const auto h = std::polar(.8, phase) *
                                   std::exp(std::complex<double>{.18, .07} * std::polar(1.0, -omega));
                    y[i] = static_cast<C>(h) * x[i];
                }
                const auto fitted = gpu.fit(measured, reference, 0, 1, 32);
                for (unsigned i = 0; i < count; ++i) {
                    const double angle = -tau * (int(i % bins) - 3408) * 37.25 / 8192 +
                                         tau * 29 * (i / bins) * 8704 / 8e6 - .63;
                    y[i] *= C{1, 0} + static_cast<C>(std::polar(.08, angle));
                }
                const auto echo = gpu.apply(measured, reference, fitted);
                const auto *actual = static_cast<const C *>(echo.normalized.mapped);
                for (unsigned i = 0; i < count; ++i) {
                    const double angle = -tau * (int(i % bins) - 3408) * 37.25 / 8192 +
                                         tau * 29 * (i / bins) * 8704 / 8e6 - .63;
                    echo_error = std::max(echo_error,
                                          std::abs(std::complex<double>(actual[i]) - std::polar(.08, angle)));
                }
            }
            check(echo_error < 5e-6,
                  "Minimum-phase residual normalization changed delayed moving-echo phase");
            std::normal_distribution<float> noise(0, .01F);
            for (unsigned i = 0; i < count; ++i)
                y[i] = C{.8F, .3F} * x[i] + C{noise(rng), noise(rng)};
            const auto noisy = gpu.fit(measured, reference, 0, 1);
            const auto *variance = static_cast<const float *>(noisy.innovation_variance.mapped);
            for (unsigned k : {0U, 51U, 1900U, 3408U, 5200U, 6816U}) {
                std::complex<double> cross{};
                double energy = 0;
                for (unsigned row = 0; row < rows; ++row) {
                    const auto at = row * bins + k;
                    cross += std::complex<double>(y[at]) * std::conj(std::complex<double>(x[at]));
                    energy += std::norm(std::complex<double>(x[at]));
                }
                double scatter = 0;
                for (unsigned row = 0; row < rows; ++row) {
                    const auto at = row * bins + k;
                    scatter +=
                        std::norm(std::complex<double>(y[at]) - cross / energy * std::complex<double>(x[at]));
                }
                check(std::abs(variance[k] / (scatter / (rows - 1)) - 1) < 1e-4,
                      "Pilot secondary covariance differs from independent residual calculation");
            }
            // Initial-guess regression: compare against the independent
            // double-precision amplitude solver with eight times the budget.
            // Include finite noise, erasures, coloured reference energy and
            // eight decades of received power; do not just fit a flat channel.
            constexpr unsigned short_rows = 16;
            const auto small_count = short_rows * bins;
            ReferenceBatch short_reference = reference;
            short_reference.count = short_rows;
            std::vector<C> cpu_x(small_count), cpu_y(small_count);
            for (float gain : {.01F, 1.F, 100.F})
                for (float sigma : {0.F, .003F}) {
                    for (unsigned k = 0; k < bins; ++k) {
                        const double omega = tau * (int(k) - 3408) / 8192;
                        std::complex<double> logarithm{};
                        for (unsigned q = 1; q <= 8; ++q)
                            logarithm += std::polar(.22 / q, .61 * q) * std::polar(1., -double(q) * omega);
                        const auto transfer =
                            static_cast<C>(std::polar(double(gain), .73) * std::exp(logarithm));
                        for (unsigned row = 0; row < short_rows; ++row) {
                            const auto i = row * bins + k;
                            x[i] = points[rng() % 64] * float(1 + .6 * std::cos(3 * omega));
                            valid[i] = rng() % 7 != 0;
                            y[i] = transfer * x[i] + gain * sigma * 100.F * C{noise(rng), noise(rng)};
                            cpu_x[i] = valid[i] ? x[i] : C{};
                            cpu_y[i] = valid[i] ? y[i] : C{};
                        }
                    }
                    // Exact log-magnitude projection should already recover a
                    // representable noiseless response, even with coloured weights.
                    // One sweep exposes a broken reduction hidden by later sweeps.
                    const auto fitted = gpu.fit(measured, short_reference, 0, 8, sigma == 0 ? 1 : 16);
                    const auto cpu = fit_minimum_phase(cpu_y, cpu_x, bins, 8, 128);
                    const auto *actual = static_cast<const C *>(fitted.response.mapped);
                    double error = 0, energy = 0;
                    for (unsigned k = 0; k < bins; ++k) {
                        error += std::norm(actual[k] - cpu.response[k]);
                        energy += std::norm(cpu.response[k]);
                    }
                    std::cout << "Convergence gain=" << gain << " noise=" << sigma
                              << " relative response error=" << std::sqrt(error / energy) << '\n';
                    check(error / energy < 1e-9,
                          "Accelerated minimum-phase solver changed the converged complex response");
                }
            std::fill_n(valid, count, 0U);
            bool rejected = false;
            try {
                gpu.fit(measured, reference, 0, 8);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            check(rejected, "Minimum-phase fit accepted an empty validity mask");
            std::cout << "Masks, all-pass rejection, explicit origin, independent gain phase and moving echo "
                         "PASS; echo max error="
                      << echo_error << '\n';
        }
        context->gpu.device->unmapMemory(*y_storage.memory);
        context->gpu.device->unmapMemory(*x_storage.memory);
        context->gpu.device->unmapMemory(*v_storage.memory);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
