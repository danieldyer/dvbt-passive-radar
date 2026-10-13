#include "dsp.hpp"
#include <fftw3.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <mutex>
#include <stdexcept>

namespace radar {
namespace {
constexpr double tau = 2 * std::numbers::pi;
std::mutex fftw_planner_mutex;
// Samples and filter coefficients are finite. Explicit arithmetic lets the
// compiler vectorize without std::complex's exceptional-infinity fallback.
C multiply(C a,C b) {
    return {a.real()*b.real()-a.imag()*b.imag(),a.real()*b.imag()+a.imag()*b.real()};
}
template<class T> T read_le(std::istream& in) {
    static_assert(std::endian::native == std::endian::little);
    T value{};
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("Truncated WAV header");
    return value;
}
double bessel0(double x) {
    double sum = 1, term = 1;
    for (unsigned k = 1; k < 80; ++k) {
        term *= x*x / (4.0*k*k);
        sum += term;
        if (term < sum*1e-16) break;
    }
    return sum;
}
}
struct Fft::Impl {
    std::size_t size;
    fftwf_plan forward{}, inverse{}, unaligned_forward{}, unaligned_inverse{};
    int alignment{};
    explicit Impl(std::size_t n) : size(n) {
        const std::lock_guard lock(fftw_planner_mutex);
        auto* tmp = fftwf_alloc_complex(n);
        if (!tmp) throw std::bad_alloc();
        alignment=fftwf_alignment_of(reinterpret_cast<float*>(tmp));
        forward = fftwf_plan_dft_1d(static_cast<int>(n),tmp,tmp,FFTW_FORWARD,FFTW_ESTIMATE);
        inverse = fftwf_plan_dft_1d(static_cast<int>(n),tmp,tmp,FFTW_BACKWARD,FFTW_ESTIMATE);
        unaligned_forward = fftwf_plan_dft_1d(static_cast<int>(n),tmp,tmp,FFTW_FORWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
        unaligned_inverse = fftwf_plan_dft_1d(static_cast<int>(n),tmp,tmp,FFTW_BACKWARD,FFTW_ESTIMATE | FFTW_UNALIGNED);
        fftwf_free(tmp);
        if (!forward || !inverse || !unaligned_forward || !unaligned_inverse) throw std::runtime_error("FFTW plan failed");
    }
    ~Impl() { const std::lock_guard lock(fftw_planner_mutex);fftwf_destroy_plan(forward); fftwf_destroy_plan(inverse);fftwf_destroy_plan(unaligned_forward);fftwf_destroy_plan(unaligned_inverse); }
};
Fft::Fft(std::size_t size) : impl_(std::make_unique<Impl>(size)) {}
Fft::~Fft() = default;
void Fft::transform(std::span<C> data, bool inverse) const {
    if (data.size() != impl_->size) throw std::runtime_error("FFT size mismatch");
    auto* p = reinterpret_cast<fftwf_complex*>(data.data());
    const bool aligned=fftwf_alignment_of(reinterpret_cast<float*>(p))==impl_->alignment;
    fftwf_execute_dft(aligned ? (inverse ? impl_->inverse : impl_->forward) :
        (inverse ? impl_->unaligned_inverse : impl_->unaligned_forward),p,p);
    if (inverse) for (auto& v : data) v /= static_cast<float>(data.size());
}

WaveReader::WaveReader(const std::filesystem::path& path) : file_buffer_(64*1024) {
    file_.rdbuf()->pubsetbuf(file_buffer_.data(),static_cast<std::streamsize>(file_buffer_.size()));
    file_.open(path,std::ios::binary);
    if (!file_) throw std::runtime_error("Cannot open WAV: " + path.string());
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
        if (size > length-pos) {
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
            const bool floating=format==3 && align==8 && bits==32;
            const bool integer=format==1 && align==4 && bits==16;
            if ((!floating && !integer) || channels != 2 ||
                byte_rate != std::uint64_t(info.sample_rate)*align || info.sample_rate == 0)
                throw std::runtime_error("WAV must contain little-endian float32 or signed int16 stereo I,Q");
            info.sample_bytes=align;
            format_seen = true;
        } else if (id == 0x61746164) {
            if (!format_seen) throw std::runtime_error("WAV data precedes format");
            info.data_offset = pos;
            std::uint64_t bytes = size;
            if (riff == 0x34364652) {
                if (!rf64_data) throw std::runtime_error("RF64 missing ds64 data length");
                bytes = rf64_data;
            } else if (length-pos > UINT32_MAX &&
                       static_cast<std::uint32_t>(length-pos) == size &&
                       static_cast<std::uint32_t>(length-8) == riff_size) {
                // Recognise the recorder's exact modulo-2^32 overflow, not arbitrary trailing chunks.
                bytes = length-pos;
                info.wrapped_size = true;
            }
            if (bytes > length-pos || bytes % info.sample_bytes) throw std::runtime_error("Invalid WAV data length");
            info.samples = bytes/info.sample_bytes;
            return;
        }
        file_.seekg(static_cast<std::streamoff>(pos + size + (size & 1)));
    }
    throw std::runtime_error("WAV has no sample data");
}
void WaveReader::read(std::uint64_t first, std::span<C> output) {
    if (first > info.samples || output.size() > info.samples-first)
        throw std::runtime_error("Read beyond WAV samples");
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(info.data_offset + first*info.sample_bytes));
    if(info.sample_bytes==4) {
        integer_samples_.resize(2*output.size());
        if(!file_.read(reinterpret_cast<char*>(integer_samples_.data()),static_cast<std::streamsize>(output.size()*4)))
            throw std::runtime_error("Short WAV sample read");
        // Every int16 value is represented exactly in float32; power-of-two
        // normalization preserves all ADC bits and introduces no quantization.
        for(std::size_t i=0;i<output.size();++i)
            output[i]={integer_samples_[2*i]*(1.f/32768),integer_samples_[2*i+1]*(1.f/32768)};
        return;
    }
    if (!file_.read(reinterpret_cast<char*>(output.data()),static_cast<std::streamsize>(output.size_bytes())))
        throw std::runtime_error("Short WAV sample read");
    // Integer reduction allows SIMD validation, including a partial final vector.
    unsigned invalid=0;
    for(auto x:output)invalid|=unsigned(!std::isfinite(x.real()))|unsigned(!std::isfinite(x.imag()));
    if(invalid)throw std::runtime_error("Non-finite input I/Q sample");
}
std::vector<float> Channelizer::prototype(double fs,double cutoff,unsigned half_at_56mhz) {
    // 120 dB design target, generous flat band beyond the outermost +/-3.328125 MHz carrier.
    auto half = static_cast<std::size_t>(std::ceil(half_at_56mhz * fs / 56e6));
    std::vector<float> taps(2*half+1);
    double sum = 0;
    for (std::size_t i = 0; i < taps.size(); ++i) {
        const double x = static_cast<double>(i)-static_cast<double>(half);
        const double z = x/static_cast<double>(half);
        const double f = cutoff/fs;
        const double sinc = x == 0 ? 2*f : std::sin(tau*f*x)/(std::numbers::pi*x);
        taps[i] = static_cast<float>(sinc*bessel0(12*std::sqrt(std::max(0.0,1-z*z)))/bessel0(12));
        sum += taps[i];
    }
    for (auto& t : taps) t /= static_cast<float>(sum);
    return taps;
}
Channelizer::Channelizer(WaveReader& input, double offset_hz, double start_seconds, double seconds)
    : input_(input), offset_(offset_hz), fft_size_(block_size*std::max(1U,input.info.sample_rate/8'000'000)), hop_(0), half_(0), fft_(fft_size_),
      response_(fft_size_), work_(fft_size_), oscillator_(fft_size_), folded_(block_size) {
    if (input.info.sample_rate % 8'000'000 || input.info.sample_rate < 8'000'000)
        throw std::runtime_error("Input rate must be an integer multiple of 8 MS/s");
    if (!std::isfinite(offset_hz) || std::abs(offset_hz)+3.5e6 > input.info.sample_rate/2.0 ||
        !std::isfinite(start_seconds) || start_seconds < 0 || !std::isfinite(seconds) || seconds <= 0)
        throw std::runtime_error("Invalid channel offset or time interval");
    decimation = input.info.sample_rate/8'000'000;
    const auto taps = prototype(input.info.sample_rate);
    half_ = taps.size()/2;
    if (taps.size() >= fft_size_) throw std::runtime_error("Input rate exceeds channelizer design");
    hop_ = (fft_size_-taps.size()+1)/decimation;
    for (std::size_t i=0;i<taps.size();++i) response_[i] = taps[i];
    fft_.transform(response_);
    // Alias the filtered spectrum before the inverse FFT. The residue accounts for
    // the FIR centre not necessarily landing on a decimation boundary.
    const auto residue=2*half_%decimation;
    for(std::size_t i=0;i<fft_size_;++i) {
        response_[i]*=std::polar(1.0F/static_cast<float>(decimation),
            static_cast<float>(tau*i*residue/fft_size_));
        oscillator_[i]=static_cast<C>(std::polar(1.0,-tau*std::remainder(
            static_cast<double>(i)*offset_/input.info.sample_rate,1.0)));
    }
    const double first = start_seconds*input.info.sample_rate;
    if (first >= static_cast<double>(input.info.samples)) throw std::runtime_error("Start is beyond recording");
    origin = std::max<std::uint64_t>(half_,static_cast<std::uint64_t>(first));
    origin = (origin+decimation-1)/decimation*decimation;
    if (origin+half_ >= input.info.samples) throw std::runtime_error("Recording shorter than FIR support");
    output_count = std::min<std::uint64_t>((input.info.samples-half_-1-origin)/decimation+1,
                                         static_cast<std::uint64_t>(std::min(seconds*output_rate,static_cast<double>(input.info.samples))));
}
std::vector<C> Channelizer::next(ChannelizerTimings* timings) {
    using Clock=std::chrono::steady_clock;
    auto mark=timings ? Clock::now() : Clock::time_point{};
    auto measure=[&](double ChannelizerTimings::* stage) {
        if(timings) {const auto now=Clock::now();timings->*stage+=std::chrono::duration<double>(now-mark).count();mark=now;}
    };
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(hop_,output_count-produced));
    if (!count) return {};
    const auto first = origin+produced*decimation-half_;
    const auto required = (count-1)*decimation+2*half_+1;
    std::fill(work_.begin(),work_.end(),C{});
    input_.read(first,std::span{work_}.first(required));
    measure(&ChannelizerTimings::read);
    const double cycles_per_sample = offset_/input_.info.sample_rate;
    const auto phase = static_cast<C>(std::polar(1.0,-tau*std::remainder(static_cast<double>(first)*cycles_per_sample,1.0)));
    // Independent table entries permit SIMD without a per-sample recurrence dependency.
    for (std::size_t i=0;i<required;++i) work_[i] = multiply(work_[i],multiply(oscillator_[i],phase));
    measure(&ChannelizerTimings::mix);
    fft_.transform(work_);
    measure(&ChannelizerTimings::forward);
    std::fill(folded_.begin(),folded_.end(),C{});
    for(std::size_t part=0;part<decimation;++part)
        for(std::size_t k=0;k<folded_.size();++k)
            folded_[k]+=multiply(work_[part*block_size+k],response_[part*block_size+k]);
    measure(&ChannelizerTimings::fold);
    inverse_.transform(folded_,true);
    std::vector<C> out(count);
    std::copy_n(folded_.begin()+static_cast<std::ptrdiff_t>(2*half_/decimation),count,out.begin());
    produced += count;
    measure(&ChannelizerTimings::inverse);
    return out;
}

std::vector<std::complex<double>> minimum_phase_basis(std::size_t bins,unsigned order) {
    if(!bins || bins>8192 || order>32)throw std::invalid_argument("Invalid minimum-phase basis dimensions");
    std::vector<std::complex<double>> basis((2*order+1)*bins);
    for(unsigned d=0;d<=2*order;++d) {
        auto column=std::span{basis}.subspan(d*bins,bins);
        for(std::size_t k=0;k<bins;++k)column[k]=std::polar(1.,-tau*((d+1)/2)*(double(k)-double(bins/2))/8192)*
            (d && !(d&1) ? std::complex<double>{0,1} : std::complex<double>{1,0});
        // Reorthogonalised modified Gram-Schmidt on the occupied carriers.
        // Fourier coordinates are not orthogonal on this partial band. Their
        // conditioning, not lack of iterations, caused the cancellation floor.
        for(unsigned pass=0;pass<2;++pass)for(unsigned previous=0;previous<d;++previous) {
            const auto other=std::span{basis}.subspan(previous*bins,bins);
            double projection=0;for(std::size_t k=0;k<bins;++k)projection+=column[k].real()*other[k].real();
            projection/=bins;
            for(std::size_t k=0;k<bins;++k)column[k]-=projection*other[k];
        }
        double norm=0;for(auto z:column)norm+=z.real()*z.real();
        norm=std::sqrt(norm/bins);
        if(!(norm>1e-12))throw std::invalid_argument("Rank-deficient minimum-phase basis");
        for(auto& z:column)z/=norm;
        double amplification=0;for(auto z:column)amplification+=std::norm(z);
        // Phase continuation outside a partially observed Fourier band can be
        // numerically unidentifiable even when the real columns are orthogonal.
        // Reject models that lose half the float significand to that inversion;
        // do not hide the problem by fitting a spurious phase or adding a ridge.
        if(amplification/bins>1/std::numeric_limits<float>::epsilon())
            throw std::invalid_argument("Minimum-phase order is numerically unidentifiable on the occupied band; reduce the order");
    }
    return basis;
}

ResponseFit fit_minimum_phase(std::span<const C> y, std::span<const C> x,
                             std::size_t bins, unsigned order, unsigned iterations) {
    if (y.size()!=x.size() || y.empty() || !bins || y.size()%bins || order>32 || !iterations)
        throw std::runtime_error("Invalid response fit dimensions/order");
    std::vector<std::complex<double>> cross(bins);
    std::vector<double> power(bins);
    // Adjacent carriers are independent accumulators. Keep their order across
    // symbols, while exposing contiguous SIMD work without a division per sample.
    for (std::size_t row=0;row<x.size();row+=bins)for(std::size_t k=0;k<bins;++k) {
        const double yr=y[row+k].real(),yi=y[row+k].imag();
        const double xr=x[row+k].real(),xi=x[row+k].imag();
        cross[k] += std::complex<double>{yr*xr+yi*xi,yi*xr-yr*xi};
        power[k] += std::norm(x[row+k]);
    }
    double total=0,amplitude_sum=0;
    for (std::size_t k=0;k<bins;++k) { amplitude_sum+=std::abs(cross[k]); total+=power[k]; }
    if (total<1e-20 || amplitude_sum<1e-20) throw std::runtime_error("No reference energy for response fit");
    const double gain=amplitude_sum/total;
    // Magnitude-only coordinate Gauss-Newton on H(z)=exp(sum c[q] z^-q).
    // Phase is reconstructed once, from the causal cepstrum; no all-pass fit.
    // Cache the basis and iterate real amplitudes, avoiding millions of trig calls.
    const auto dimensions=2*order+1;
    struct Basis {std::size_t bins=0;unsigned order=99;std::vector<double> real,imag;};
    thread_local Basis basis;
    if(basis.bins!=bins || basis.order!=order) {
        const auto orthogonal=minimum_phase_basis(bins,order);
        basis.bins=bins;basis.order=order;basis.real.resize(dimensions*bins);basis.imag.resize(dimensions*bins);
        for(std::size_t i=0;i<orthogonal.size();++i){basis.real[i]=orthogonal[i].real();basis.imag[i]=orthogonal[i].imag();}
    }
    std::vector<double> coefficient(dimensions),target(bins),amplitude(bins,gain);
    coefficient[0]=std::log(gain);
    for(std::size_t k=0;k<bins;++k) target[k]=power[k]>1e-12 ? std::abs(cross[k])/power[k] : 0;
    for(unsigned it=0;it<iterations;++it){for(unsigned d=0;d<dimensions;++d) {
        const auto* real=basis.real.data()+d*bins;
        double numerator=0,denominator=1e-12;
        for(std::size_t k=0;k<bins;++k) {
            const double derivative=amplitude[k]*real[k];
            numerator+=derivative*(target[k]-amplitude[k])*power[k];
            denominator+=derivative*derivative*power[k];
        }
        const double update=std::clamp(numerator/denominator,-0.25,0.25);
        coefficient[d]+=update;
        for(std::size_t k=0;k<bins;++k)amplitude[k]*=std::exp(update*real[k]);
    }
        for(std::size_t k=0;k<bins;++k){double magnitude=0;
            for(unsigned d=0;d<dimensions;++d)magnitude+=coefficient[d]*basis.real[d*bins+k];
            amplitude[k]=std::exp(magnitude);
        }
    }
    std::vector<std::complex<double>> h(bins);
    for(std::size_t k=0;k<bins;++k) {
        double angle=0;
        for(unsigned d=0;d<dimensions;++d)angle+=coefficient[d]*basis.imag[d*bins+k];
        h[k]=std::polar(amplitude[k],angle);
    }
    std::complex<double> phase{};
    for(std::size_t k=0;k<bins;++k)phase+=cross[k]*std::conj(h[k]);
    if(std::abs(phase)>0)for(auto& value:h)value*=phase/std::abs(phase);
    ResponseFit result;
    result.response.resize(bins);
    for (std::size_t k=0;k<bins;++k) result.response[k]=static_cast<C>(h[k]);
    double error=0, energy=0;
    for (std::size_t i=0;i<x.size();++i) { error+=std::norm(y[i]-result.response[i%bins]*x[i]); energy+=std::norm(y[i]); }
    result.relative_error=error/std::max(energy,1e-30);
    return result;
}
} // namespace radar
