#include "dsp.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fftw3.h>
#include <limits>
#include <mutex>
#include <numbers>
#include <stdexcept>

namespace radar {
namespace {
constexpr double tau = 2 * std::numbers::pi;
std::mutex fftw_planner_mutex;
template <class T> T read_le(std::istream &in) {
    static_assert(std::endian::native == std::endian::little);
    T value{};
    if (!in.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error("Truncated WAV header");
    return value;
}
double bessel0(double x) {
    double sum = 1, term = 1;
    for (unsigned k = 1; k < 80; ++k) {
        term *= x * x / (4.0 * k * k);
        sum += term;
        if (term < sum * 1e-16)
            break;
    }
    return sum;
}
} // namespace
struct Fft::Impl {
    std::size_t size;
    fftwf_plan forward{}, inverse{}, unaligned_forward{}, unaligned_inverse{};
    int alignment{};
    explicit Impl(std::size_t n) : size(n) {
        const std::lock_guard lock(fftw_planner_mutex);
        auto *tmp = fftwf_alloc_complex(n);
        if (!tmp)
            throw std::bad_alloc();
        alignment = fftwf_alignment_of(reinterpret_cast<float *>(tmp));
        forward = fftwf_plan_dft_1d(static_cast<int>(n), tmp, tmp, FFTW_FORWARD, FFTW_ESTIMATE);
        inverse = fftwf_plan_dft_1d(static_cast<int>(n), tmp, tmp, FFTW_BACKWARD, FFTW_ESTIMATE);
        unaligned_forward =
            fftwf_plan_dft_1d(static_cast<int>(n), tmp, tmp, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED);
        unaligned_inverse =
            fftwf_plan_dft_1d(static_cast<int>(n), tmp, tmp, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_UNALIGNED);
        fftwf_free(tmp);
        if (!forward || !inverse || !unaligned_forward || !unaligned_inverse)
            throw std::runtime_error("FFTW plan failed");
    }
    ~Impl() {
        const std::lock_guard lock(fftw_planner_mutex);
        fftwf_destroy_plan(forward);
        fftwf_destroy_plan(inverse);
        fftwf_destroy_plan(unaligned_forward);
        fftwf_destroy_plan(unaligned_inverse);
    }
};
Fft::Fft(std::size_t size) : impl_(std::make_unique<Impl>(size)) {}
Fft::~Fft() = default;
void Fft::transform(std::span<C> data, bool inverse) const {
    if (data.size() != impl_->size)
        throw std::runtime_error("FFT size mismatch");
    auto *p = reinterpret_cast<fftwf_complex *>(data.data());
    const bool aligned = fftwf_alignment_of(reinterpret_cast<float *>(p)) == impl_->alignment;
    fftwf_execute_dft(aligned ? (inverse ? impl_->inverse : impl_->forward)
                              : (inverse ? impl_->unaligned_inverse : impl_->unaligned_forward),
                      p, p);
    if (inverse)
        for (auto &v : data)
            v /= static_cast<float>(data.size());
}

WaveReader::WaveReader(const std::filesystem::path &path) : file_buffer_(64 * 1024) {
    file_.rdbuf()->pubsetbuf(file_buffer_.data(), static_cast<std::streamsize>(file_buffer_.size()));
    file_.open(path, std::ios::binary);
    if (!file_)
        throw std::runtime_error("Cannot open WAV: " + path.string());
    const auto length = std::filesystem::file_size(path);
    const auto riff = read_le<std::uint32_t>(file_);
    const auto riff_size = read_le<std::uint32_t>(file_);
    if ((riff != 0x46464952 && riff != 0x34364652) || read_le<std::uint32_t>(file_) != 0x45564157)
        throw std::runtime_error("Expected RIFF/RF64 WAVE");
    std::uint64_t rf64_data = 0;
    bool format_seen = false;
    while (file_ && static_cast<std::uint64_t>(file_.tellg()) + 8 <= length) {
        const auto id = read_le<std::uint32_t>(file_);
        const auto size = read_le<std::uint32_t>(file_);
        const auto pos = static_cast<std::uint64_t>(file_.tellg());
        if (size > length - pos) {
            if (id != 0x61746164 || riff != 0x34364652)
                throw std::runtime_error("WAV chunk exceeds file size");
        }
        if (id == 0x34367364 && size >= 28) { // ds64
            (void)read_le<std::uint64_t>(file_);
            rf64_data = read_le<std::uint64_t>(file_);
        } else if (id == 0x20746d66 && size >= 16) {
            const auto format = read_le<std::uint16_t>(file_);
            const auto channels = read_le<std::uint16_t>(file_);
            info.sample_rate = read_le<std::uint32_t>(file_);
            const auto byte_rate = read_le<std::uint32_t>(file_);
            const auto align = read_le<std::uint16_t>(file_);
            const auto bits = read_le<std::uint16_t>(file_);
            const bool floating = format == 3 && align == 8 && bits == 32;
            const bool integer = format == 1 && align == 4 && bits == 16;
            if ((!floating && !integer) || channels != 2 ||
                byte_rate != std::uint64_t(info.sample_rate) * align || info.sample_rate == 0)
                throw std::runtime_error("WAV must contain little-endian float32 or signed int16 stereo I,Q");
            info.sample_bytes = align;
            format_seen = true;
        } else if (id == 0x61746164) {
            if (!format_seen)
                throw std::runtime_error("WAV data precedes format");
            info.data_offset = pos;
            std::uint64_t bytes = size;
            if (riff == 0x34364652) {
                if (!rf64_data)
                    throw std::runtime_error("RF64 missing ds64 data length");
                bytes = rf64_data;
            } else if (length - pos > UINT32_MAX && static_cast<std::uint32_t>(length - pos) == size &&
                       static_cast<std::uint32_t>(length - 8) == riff_size) {
                // Recognise the recorder's exact modulo-2^32 overflow, not arbitrary trailing chunks.
                bytes = length - pos;
                info.wrapped_size = true;
            }
            if (bytes > length - pos || bytes % info.sample_bytes)
                throw std::runtime_error("Invalid WAV data length");
            info.samples = bytes / info.sample_bytes;
            return;
        }
        file_.seekg(static_cast<std::streamoff>(pos + size + (size & 1)));
    }
    throw std::runtime_error("WAV has no sample data");
}
void WaveReader::read(std::uint64_t first, std::span<C> output) {
    if (first > info.samples || output.size() > info.samples - first)
        throw std::runtime_error("Read beyond WAV samples");
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(info.data_offset + first * info.sample_bytes));
    if (info.sample_bytes == 4) {
        integer_samples_.resize(2 * output.size());
        if (!file_.read(reinterpret_cast<char *>(integer_samples_.data()),
                        static_cast<std::streamsize>(output.size() * 4)))
            throw std::runtime_error("Short WAV sample read");
        // Every int16 value is represented exactly in float32; power-of-two
        // normalization preserves all ADC bits and introduces no quantization.
        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = {integer_samples_[2 * i] * (1.f / 32768),
                         integer_samples_[2 * i + 1] * (1.f / 32768)};
        return;
    }
    if (!file_.read(reinterpret_cast<char *>(output.data()),
                    static_cast<std::streamsize>(output.size_bytes())))
        throw std::runtime_error("Short WAV sample read");
    // Integer reduction allows SIMD validation, including a partial final vector.
    unsigned invalid = 0;
    for (auto x : output)
        invalid |= unsigned(!std::isfinite(x.real())) | unsigned(!std::isfinite(x.imag()));
    if (invalid)
        throw std::runtime_error("Non-finite input I/Q sample");
}
std::vector<float> kaiser_prototype(double fs, double cutoff, unsigned half_at_56mhz) {
    // 120 dB design target, generous flat band beyond the outermost +/-3.328125 MHz carrier.
    auto half = static_cast<std::size_t>(std::ceil(half_at_56mhz * fs / 56e6));
    std::vector<float> taps(2 * half + 1);
    double sum = 0;
    for (std::size_t i = 0; i < taps.size(); ++i) {
        const double x = static_cast<double>(i) - static_cast<double>(half);
        const double z = x / static_cast<double>(half);
        const double f = cutoff / fs;
        const double sinc = x == 0 ? 2 * f : std::sin(tau * f * x) / (std::numbers::pi * x);
        taps[i] = static_cast<float>(sinc * bessel0(12 * std::sqrt(std::max(0.0, 1 - z * z))) / bessel0(12));
        sum += taps[i];
    }
    for (auto &t : taps)
        t /= static_cast<float>(sum);
    return taps;
}
std::vector<std::complex<double>> minimum_phase_basis(std::size_t bins, unsigned order) {
    if (!bins || bins > 8192 || order > 32)
        throw std::invalid_argument("Invalid minimum-phase basis dimensions");
    std::vector<std::complex<double>> basis((2 * order + 1) * bins);
    for (unsigned d = 0; d <= 2 * order; ++d) {
        auto column = std::span{basis}.subspan(d * bins, bins);
        for (std::size_t k = 0; k < bins; ++k)
            column[k] = std::polar(1., -tau * ((d + 1) / 2) * (double(k) - double(bins / 2)) / 8192) *
                        (d && !(d & 1) ? std::complex<double>{0, 1} : std::complex<double>{1, 0});
        // Reorthogonalised modified Gram-Schmidt on the occupied carriers.
        // Fourier coordinates are not orthogonal on this partial band. Their
        // conditioning, not lack of iterations, caused the cancellation floor.
        for (unsigned pass = 0; pass < 2; ++pass)
            for (unsigned previous = 0; previous < d; ++previous) {
                const auto other = std::span{basis}.subspan(previous * bins, bins);
                double projection = 0;
                for (std::size_t k = 0; k < bins; ++k)
                    projection += column[k].real() * other[k].real();
                projection /= bins;
                for (std::size_t k = 0; k < bins; ++k)
                    column[k] -= projection * other[k];
            }
        double norm = 0;
        for (auto z : column)
            norm += z.real() * z.real();
        norm = std::sqrt(norm / bins);
        if (!(norm > 1e-12))
            throw std::invalid_argument("Rank-deficient minimum-phase basis");
        for (auto &z : column)
            z /= norm;
        double amplification = 0;
        for (auto z : column)
            amplification += std::norm(z);
        // Phase continuation outside a partially observed Fourier band can be
        // numerically unidentifiable even when the real columns are orthogonal.
        // Reject models that lose half the float significand to that inversion;
        // do not hide the problem by fitting a spurious phase or adding a ridge.
        if (amplification / bins > 1 / std::numeric_limits<float>::epsilon())
            throw std::invalid_argument(
                "Minimum-phase order is numerically unidentifiable on the occupied band; reduce the order");
    }
    return basis;
}

} // namespace radar
