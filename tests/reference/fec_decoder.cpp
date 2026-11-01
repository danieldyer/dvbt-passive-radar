#include "fec_decoder.hpp"
#include <bit>
#include <chrono>
#include <ctime>
namespace radar::validation {
namespace {
// Count CPU consumed by this worker, excluding preemption by other workers or
// unrelated processes. Summing these measurements gives core-seconds of work.
struct Clock {
    using duration = std::chrono::nanoseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<Clock>;
    [[maybe_unused]] static constexpr bool is_steady = true;
    static time_point now() {
        timespec value{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
        return time_point{std::chrono::seconds{value.tv_sec} + duration{value.tv_nsec}};
    }
};
double seconds(Clock::time_point from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
}
template <unsigned Bits> std::array<std::uint8_t, 6817> hard_labels(std::span<const C> carriers) {
    std::array<std::uint8_t, 6817> labels;
    for (unsigned k = 0; k < 6817; ++k)
        labels[k] = hard_qam<Bits>(carriers[k]);
    return labels;
}

} // namespace
void FecDecoder::configure(std::size_t count, const Tps &mode, const Modem &tables) {
    const auto bits = qam_bits(mode);
    if (count > 1020)
        throw std::invalid_argument("FEC requires <=1020 non-hierarchical 8K DVB-T symbols");
    if (constellation_ != mode.constellation) {
        indices_ = tables.payload_bit_indices(mode);
        constellation_ = mode.constellation;
    }
    bits_.resize(count * 6048 * bits);
}
void FecDecoder::demap(std::span<const C> carriers, unsigned mod, unsigned bits, std::size_t symbol) {
    if (carriers.size() != 6817 || mod > 3)
        throw std::invalid_argument("Invalid FEC carriers");
    const auto labels = bits == 2   ? hard_labels<2>(carriers)
                        : bits == 4 ? hard_labels<4>(carriers)
                                    : hard_labels<6>(carriers);
    const auto *gather = indices_.data() + mod * 6048 * bits;
    auto *out = bits_.data() + symbol * 6048 * bits;
    for (unsigned i = 0; i < 6048 * bits; ++i) {
        const auto index = gather[i];
        const auto label = labels[index / 8];
        out[i] = label == 255 ? 2 : (label >> (index % 8)) & 1;
    }
}
FecBatch FecDecoder::finish(std::span<const std::uint8_t> bits, const Tps &mode, FecTimings *timing,
                            std::size_t overlap, CodedMetric metric) {
    if (metric == CodedMetric::minimax_likelihood) {
        if (overlap)
            throw std::invalid_argument("Production compact batches are independent");
        return production_.decode(bits, mode.rate, timing);
    }
    if (metric != CodedMetric::hard_bits && metric != CodedMetric::likelihood_plus_7)
        throw std::invalid_argument("Invalid reference metric encoding");
    auto start = timing ? Clock::now() : Clock::time_point{};
    const auto bytes = metric == CodedMetric::hard_bits
                           ? viterbi_.decode(bits, mode.rate, overlap)
                           : viterbi_.decode_soft(bits, mode.rate, overlap, SoftMetric::uniform);
    if (timing) {
        timing->viterbi += seconds(start);
        start = Clock::now();
    }
    packets_.resize(bytes.size() / 204 > 11 ? bytes.size() / 204 - 11 : 0);
    for (std::size_t p = 0; p < packets_.size(); ++p) {
        auto &packet = packets_[p];
        for (unsigned j = 0; j < 204; ++j) {
            const auto i = 2244 + p * 204 + j;
            packet.bytes[j] = bytes[i - (11 - i % 12) * 204];
        }
        const int errors = fec::DvbtRs::decode(packet.bytes);
        packet.valid = errors >= 0 && (packet.bytes[0] == 0x47 || packet.bytes[0] == 0xb8);
        packet.corrected = packet.valid ? unsigned(errors) : 0;
    }
    if (timing)
        timing->reed_solomon += seconds(start);
    const auto pattern = fec::puncture_pattern(mode.rate);
    return {packets_,
            packets_.size(),
            bits.size() / std::count(pattern.begin(), pattern.end(), 1U) * (pattern.size() / 2),
            viterbi_.reused_steps(),
            1,
            0,
            0};
}
FecBatch FecDecoder::decode(std::span<const Symbol> symbols, const Tps &mode, const Modem &tables,
                            FecTimings *timing) {
    const auto start = timing ? Clock::now() : Clock::time_point{};
    configure(symbols.size(), mode, tables);
    for (std::size_t s = 0; s < symbols.size(); ++s)
        demap(symbols[s].equalized, symbols[s].modulo4, 2 + 2 * mode.constellation, s);
    if (timing)
        timing->prepare += seconds(start);
    return finish(bits_, mode, timing, 0, CodedMetric::hard_bits);
}
FecBatch FecDecoder::decode(const OfdmBatch &batch, const Tps &mode, const Modem &tables, FecTimings *timing,
                            std::size_t overlap_symbols) {
    if (overlap_symbols > 1020)
        throw std::invalid_argument("Invalid FEC overlap");
    const auto overlap = overlap_symbols * 6048 * qam_bits(mode);
    if (batch.coded_bits.bytes) {
        const auto count = batch.count * 6048 * qam_bits(mode);
        if (batch.count > 1020 || !batch.coded_bits.mapped || batch.coded_bits.bytes < count)
            throw std::invalid_argument("Invalid resident coded-metric batch");
        static_assert(std::endian::native == std::endian::little);
        return finish({static_cast<const std::uint8_t *>(batch.coded_bits.mapped), count}, mode, timing,
                      overlap, batch.coded_metric);
    }
    const auto start = timing ? Clock::now() : Clock::time_point{};
    configure(batch.count, mode, tables);
    if (batch.count && (!batch.equalized.mapped || !batch.pilots.mapped || batch.pilot_stride < 2 ||
                        batch.equalized.bytes < batch.count * 6817 * sizeof(C) ||
                        batch.pilots.bytes < batch.count * batch.pilot_stride * sizeof(C)))
        throw std::invalid_argument("Invalid mapped OFDM FEC batch");
    const auto *stats = static_cast<const C *>(batch.pilots.mapped);
    const auto *carriers = static_cast<const C *>(batch.equalized.mapped);
    for (std::size_t s = 0; s < batch.count; ++s)
        demap({carriers + s * 6817, 6817}, std::bit_cast<std::uint32_t>(stats[s * batch.pilot_stride].real()),
              2 + 2 * mode.constellation, s);
    if (timing)
        timing->prepare += seconds(start);
    return finish(bits_, mode, timing, overlap, CodedMetric::hard_bits);
}
} // namespace radar::validation
