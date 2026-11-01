#include "dsp_reference.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
namespace radar {
namespace {
constexpr double tau = 2 * std::numbers::pi;
// Samples and filter coefficients are finite. Explicit arithmetic lets the
// compiler vectorize without std::complex's exceptional-infinity fallback.
C multiply(C a, C b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}
} // namespace
Channelizer::Channelizer(WaveReader &input, double offset_hz, double start_seconds, double seconds)
    : input_(input), offset_(offset_hz),
      fft_size_(block_size * std::max(1U, input.info.sample_rate / 8'000'000)), hop_(0), half_(0),
      fft_(fft_size_), response_(fft_size_), work_(fft_size_), oscillator_(fft_size_), folded_(block_size) {
    if (input.info.sample_rate % 8'000'000 || input.info.sample_rate < 8'000'000)
        throw std::runtime_error("Input rate must be an integer multiple of 8 MS/s");
    if (!std::isfinite(offset_hz) || std::abs(offset_hz) + 3.5e6 > input.info.sample_rate / 2.0 ||
        !std::isfinite(start_seconds) || start_seconds < 0 || !std::isfinite(seconds) || seconds <= 0)
        throw std::runtime_error("Invalid channel offset or time interval");
    decimation = input.info.sample_rate / 8'000'000;
    const auto taps = prototype(input.info.sample_rate);
    half_ = taps.size() / 2;
    if (taps.size() >= fft_size_)
        throw std::runtime_error("Input rate exceeds channelizer design");
    hop_ = (fft_size_ - taps.size() + 1) / decimation;
    for (std::size_t i = 0; i < taps.size(); ++i)
        response_[i] = taps[i];
    fft_.transform(response_);
    // Alias the filtered spectrum before the inverse FFT. The residue accounts for
    // the FIR centre not necessarily landing on a decimation boundary.
    const auto residue = 2 * half_ % decimation;
    for (std::size_t i = 0; i < fft_size_; ++i) {
        response_[i] *= std::polar(1.0F / static_cast<float>(decimation),
                                   static_cast<float>(tau * i * residue / fft_size_));
        oscillator_[i] = static_cast<C>(std::polar(
            1.0, -tau * std::remainder(static_cast<double>(i) * offset_ / input.info.sample_rate, 1.0)));
    }
    const double first = start_seconds * input.info.sample_rate;
    if (first >= static_cast<double>(input.info.samples))
        throw std::runtime_error("Start is beyond recording");
    origin = std::max<std::uint64_t>(half_, static_cast<std::uint64_t>(first));
    origin = (origin + decimation - 1) / decimation * decimation;
    if (origin + half_ >= input.info.samples)
        throw std::runtime_error("Recording shorter than FIR support");
    output_count = std::min<std::uint64_t>(
        (input.info.samples - half_ - 1 - origin) / decimation + 1,
        static_cast<std::uint64_t>(std::min(seconds * output_rate, static_cast<double>(input.info.samples))));
}
std::vector<C> Channelizer::next(ChannelizerTimings *timings) {
    using Clock = std::chrono::steady_clock;
    auto mark = timings ? Clock::now() : Clock::time_point{};
    auto measure = [&](double ChannelizerTimings::*stage) {
        if (timings) {
            const auto now = Clock::now();
            timings->*stage += std::chrono::duration<double>(now - mark).count();
            mark = now;
        }
    };
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(hop_, output_count - produced));
    if (!count)
        return {};
    const auto first = origin + produced * decimation - half_;
    const auto required = (count - 1) * decimation + 2 * half_ + 1;
    std::fill(work_.begin(), work_.end(), C{});
    input_.read(first, std::span{work_}.first(required));
    measure(&ChannelizerTimings::read);
    const double cycles_per_sample = offset_ / input_.info.sample_rate;
    const auto phase = static_cast<C>(
        std::polar(1.0, -tau * std::remainder(static_cast<double>(first) * cycles_per_sample, 1.0)));
    // Independent table entries permit SIMD without a per-sample recurrence dependency.
    for (std::size_t i = 0; i < required; ++i)
        work_[i] = multiply(work_[i], multiply(oscillator_[i], phase));
    measure(&ChannelizerTimings::mix);
    fft_.transform(work_);
    measure(&ChannelizerTimings::forward);
    std::fill(folded_.begin(), folded_.end(), C{});
    for (std::size_t part = 0; part < decimation; ++part)
        for (std::size_t k = 0; k < folded_.size(); ++k)
            folded_[k] += multiply(work_[part * block_size + k], response_[part * block_size + k]);
    measure(&ChannelizerTimings::fold);
    inverse_.transform(folded_, true);
    std::vector<C> out(count);
    std::copy_n(folded_.begin() + static_cast<std::ptrdiff_t>(2 * half_ / decimation), count, out.begin());
    produced += count;
    measure(&ChannelizerTimings::inverse);
    return out;
}

ResponseFit fit_minimum_phase(std::span<const C> y, std::span<const C> x, std::size_t bins, unsigned order,
                              unsigned iterations) {
    if (y.size() != x.size() || y.empty() || !bins || y.size() % bins || order > 32 || !iterations)
        throw std::runtime_error("Invalid response fit dimensions/order");
    std::vector<std::complex<double>> cross(bins);
    std::vector<double> power(bins);
    // Adjacent carriers are independent accumulators. Keep their order across
    // symbols, while exposing contiguous SIMD work without a division per sample.
    for (std::size_t row = 0; row < x.size(); row += bins)
        for (std::size_t k = 0; k < bins; ++k) {
            const double yr = y[row + k].real(), yi = y[row + k].imag();
            const double xr = x[row + k].real(), xi = x[row + k].imag();
            cross[k] += std::complex<double>{yr * xr + yi * xi, yi * xr - yr * xi};
            power[k] += std::norm(x[row + k]);
        }
    double total = 0, amplitude_sum = 0;
    for (std::size_t k = 0; k < bins; ++k) {
        amplitude_sum += std::abs(cross[k]);
        total += power[k];
    }
    if (total < 1e-20 || amplitude_sum < 1e-20)
        throw std::runtime_error("No reference energy for response fit");
    const double gain = amplitude_sum / total;
    // Magnitude-only coordinate Gauss-Newton on H(z)=exp(sum c[q] z^-q).
    // Phase is reconstructed once, from the causal cepstrum; no all-pass fit.
    // Cache the basis and iterate real amplitudes, avoiding millions of trig calls.
    const auto dimensions = 2 * order + 1;
    struct Basis {
        std::size_t bins = 0;
        unsigned order = 99;
        std::vector<double> real, imag;
    };
    thread_local Basis basis;
    if (basis.bins != bins || basis.order != order) {
        const auto orthogonal = minimum_phase_basis(bins, order);
        basis.bins = bins;
        basis.order = order;
        basis.real.resize(dimensions * bins);
        basis.imag.resize(dimensions * bins);
        for (std::size_t i = 0; i < orthogonal.size(); ++i) {
            basis.real[i] = orthogonal[i].real();
            basis.imag[i] = orthogonal[i].imag();
        }
    }
    std::vector<double> coefficient(dimensions), target(bins), amplitude(bins, gain);
    coefficient[0] = std::log(gain);
    for (std::size_t k = 0; k < bins; ++k)
        target[k] = power[k] > 1e-12 ? std::abs(cross[k]) / power[k] : 0;
    for (unsigned it = 0; it < iterations; ++it) {
        for (unsigned d = 0; d < dimensions; ++d) {
            const auto *real = basis.real.data() + d * bins;
            double numerator = 0, denominator = 1e-12;
            for (std::size_t k = 0; k < bins; ++k) {
                const double derivative = amplitude[k] * real[k];
                numerator += derivative * (target[k] - amplitude[k]) * power[k];
                denominator += derivative * derivative * power[k];
            }
            const double update = std::clamp(numerator / denominator, -0.25, 0.25);
            coefficient[d] += update;
            for (std::size_t k = 0; k < bins; ++k)
                amplitude[k] *= std::exp(update * real[k]);
        }
        for (std::size_t k = 0; k < bins; ++k) {
            double magnitude = 0;
            for (unsigned d = 0; d < dimensions; ++d)
                magnitude += coefficient[d] * basis.real[d * bins + k];
            amplitude[k] = std::exp(magnitude);
        }
    }
    std::vector<std::complex<double>> h(bins);
    for (std::size_t k = 0; k < bins; ++k) {
        double angle = 0;
        for (unsigned d = 0; d < dimensions; ++d)
            angle += coefficient[d] * basis.imag[d * bins + k];
        h[k] = std::polar(amplitude[k], angle);
    }
    std::complex<double> phase{};
    for (std::size_t k = 0; k < bins; ++k)
        phase += cross[k] * std::conj(h[k]);
    if (std::abs(phase) > 0)
        for (auto &value : h)
            value *= phase / std::abs(phase);
    ResponseFit result;
    result.response.resize(bins);
    for (std::size_t k = 0; k < bins; ++k)
        result.response[k] = static_cast<C>(h[k]);
    double error = 0, energy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        error += std::norm(y[i] - result.response[i % bins] * x[i]);
        energy += std::norm(y[i]);
    }
    result.relative_error = error / std::max(energy, 1e-30);
    return result;
}
} // namespace radar
