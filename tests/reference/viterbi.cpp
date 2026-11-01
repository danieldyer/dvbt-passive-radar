#include "viterbi.hpp"
#include "soft_quantizer.hpp"
#include <limits>
namespace radar::validation {
using namespace radar::fec;
namespace {
constexpr auto costs = []() consteval {
    std::array<std::array<std::uint8_t, 32>, 16> table{};
    for (unsigned mask = 0; mask < 4; ++mask)
        for (unsigned observed = 0; observed < 4; ++observed)
            for (unsigned p = 0; p < 32; ++p)
                table[mask * 4 + observed][p] = std::popcount((DvbtCode::branches[2 * p] ^ observed) & mask);
    return table;
}();
template <bool Companded>
constexpr auto soft_levels = []() consteval {
    std::array<int, 15> result{};
    for (int code = 0; code < 15; ++code) {
        const int signed_rank = code - 7, rank = signed_rank < 0 ? -signed_rank : signed_rank;
        const int value = Companded ? minimax_llr_levels[rank] : rank;
        result[code] = signed_rank < 0 ? -value : value;
    }
    return result;
}();
template <bool Companded>
constexpr auto soft_costs = []() consteval {
    std::array<std::array<std::uint8_t, 32>, 225> table{};
    for (unsigned a = 0; a < 15; ++a)
        for (unsigned b = 0; b < 15; ++b)
            for (unsigned p = 0; p < 32; ++p) {
                const auto label = DvbtCode::branches[2 * p];
                const int x = soft_levels<Companded>[a], y = soft_levels<Companded>[b];
                table[a * 15 + b][p] = std::max(label & 1 ? x : -x, 0) + std::max(label & 2 ? y : -y, 0);
            }
    return table;
}();
struct Metrics {
    std::array<std::uint8_t, 64> metric{};
    // Inlining also keeps the metric array from escaping through the tail path.
    [[gnu::always_inline]] std::uint64_t advance(const std::uint8_t *cost, unsigned weight) {
        std::array<std::uint8_t, 64> next;
        std::array<std::uint8_t, 32> decisions;
        for (unsigned p = 0; p < 32; ++p) {
            const auto branch = cost[p];
            const auto complement = static_cast<std::uint8_t>(weight - branch);
            const auto lo0 = static_cast<std::uint8_t>(metric[p] + branch);
            const auto hi0 = static_cast<std::uint8_t>(metric[p + 32] + complement);
            const auto lo1 = static_cast<std::uint8_t>(metric[p] + complement);
            const auto hi1 = static_cast<std::uint8_t>(metric[p + 32] + branch);
            next[2 * p] = std::min(lo0, hi0);
            next[2 * p + 1] = std::min(lo1, hi1);
            decisions[p] = (hi0 < lo0 ? 1U << (2 * p % 8) : 0) | (hi1 < lo1 ? 2U << (2 * p % 8) : 0);
        }
        // Each byte contains two survivor bits in disjoint positions. Two
        // pairwise sums pack four butterflies into each output byte. Keep these
        // short loops vectorized: eager unrolling otherwise scalarizes packing.
        std::array<std::uint8_t, 16> pairs;
#if defined(__clang__)
#pragma clang loop vectorize(enable) unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
        for (unsigned i = 0; i < 16; ++i)
            pairs[i] = decisions[2 * i] + decisions[2 * i + 1];
        std::array<std::uint8_t, 8> packed;
#if defined(__clang__)
#pragma clang loop vectorize(enable) unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
        for (unsigned i = 0; i < 8; ++i)
            packed[i] = pairs[2 * i] + pairs[2 * i + 1];
        auto result = std::bit_cast<std::uint64_t>(packed);
        if constexpr (std::endian::native == std::endian::big)
            result = std::byteswap(result);
        metric = next;
        return result;
    }
    [[gnu::always_inline]] void normalize() {
        std::uint8_t minimum = std::numeric_limits<std::uint8_t>::max();
        for (auto value : metric)
            minimum = std::min(minimum, value);
        for (auto &value : metric)
            value -= minimum;
    }
    std::array<std::uint8_t, 64> values() const {
        return metric;
    }
};

} // namespace

template <unsigned Rate, bool Erasures, bool Soft, bool Companded, bool Cache>
std::vector<std::uint8_t> ReferenceViterbi::decode_rate(std::span<const std::uint8_t> bits,
                                                        std::size_t overlap) {
    static_assert(!Companded || Soft);
    constexpr unsigned key = Rate + (Companded ? 10 : Soft ? 5 : 0);
    constexpr unsigned maximum_llr = Companded ? minimax_llr_levels.back() : 7;
    constexpr unsigned maximum_metric = std::numeric_limits<std::uint8_t>::max();
    constexpr auto pattern = punctures[Rate];
    constexpr unsigned period = puncture_lengths[Rate] / 2;
    constexpr unsigned sent = [] {
        unsigned n = 0;
        for (auto bit : punctures[Rate])
            n += bit;
        return n;
    }();
    if (bits.size() % sent)
        throw std::invalid_argument("Incomplete puncture cycle");
    const auto count = bits.size() / sent * period;
    // Power-of-two puncture-cycle blocks align with DVB-T frame boundaries.
    // Checkpoint spacing stays independent of the more frequent soft-metric
    // normalisation, preserving the bounded overlap-cache footprint.
    constexpr unsigned cycles = std::bit_floor(96U / period), block = period * cycles;
    // Any two K=7 states merge after six transitions. Their metric spread
    // is at most the largest six-step observation cost. A puncture cycle adds
    // at most maximum_llr times its transmitted-bit count. Normalise before
    // the selected unsigned width overflows, preserving exact comparisons.
    constexpr unsigned spread = [=]() consteval {
        unsigned maximum = 0;
        for (unsigned phase = 0; phase < period; ++phase) {
            unsigned observed = 0;
            for (unsigned t = 0; t < 6; ++t)
                for (unsigned j = 0; j < 2; ++j)
                    observed += pattern[(2 * (phase + t) + j) % (2 * period)];
            maximum = std::max(maximum, maximum_llr * observed);
        }
        return maximum;
    }();
    constexpr unsigned normal_cycles = Soft ? (maximum_metric - spread) / (maximum_llr * sent) : cycles;
    static_assert(!Soft || spread + normal_cycles * maximum_llr * sent <= maximum_metric);
    const bool aligned = Cache && overlap && overlap <= bits.size() && overlap % (sent * cycles) == 0;
    const bool retain =
        aligned && overlap <= bits.size() / 2 && (bits.size() - overlap) % (sent * cycles) == 0;
    const auto repeated = aligned ? overlap / sent * period : 0, tail = count - repeated;
    const bool reuse = aligned && cached_rate_ == key && cached_bits_.size() == overlap;
    cached_rate_ = 15; // An allocation/decoder failure must not leave a reusable partial trellis.
    reused_steps_ = 0;
    const auto previous_tail = reuse ? survivors_.size() - repeated : 0;
    // Preserve the old suffix until reuse is decided. It starts beyond the
    // repeated prefix, so forward decoding cannot overwrite its source.
    if constexpr (Cache)
        survivors_.resize(reuse ? std::max(survivors_.size(), count) : count);
    else if (survivors_.size() < count)
        survivors_.resize(count);
    next_checkpoints_.resize(retain ? repeated / block : 0);
    Metrics metrics;
    std::size_t at = 0, t = 0;
    // Keep both template layers in the time loop. Clang otherwise outlines a
    // puncture cycle and reloads captured metrics/counters on every cycle.
    auto advance = [&]<std::size_t Phase> [[gnu::always_inline]] (Metrics &state) {
        if constexpr (Soft) {
            unsigned a = 7, b = 7;
            if constexpr (pattern[2 * Phase])
                a = bits[at++];
            if constexpr (pattern[2 * Phase + 1])
                b = bits[at++];
            const unsigned weight = std::abs(soft_levels<Companded>[a]) + std::abs(soft_levels<Companded>[b]);
            survivors_[t++] = state.advance(soft_costs<Companded>[a * 15 + b].data(), weight);
        } else {
            unsigned observed = 0, mask = 0;
            if constexpr (pattern[2 * Phase]) {
                const auto bit = bits[at++];
                if (!Erasures || bit < 2) {
                    observed = bit;
                    mask = 1;
                }
            }
            if constexpr (pattern[2 * Phase + 1]) {
                const auto bit = bits[at++];
                if (!Erasures || bit < 2) {
                    observed |= bit << 1;
                    mask |= 2;
                }
            }
            survivors_[t++] = state.advance(costs[mask * 4 + observed].data(), (mask & 1) + (mask >> 1));
        }
    };
    // Canonical checkpoints enable exact overlap reuse. Hard metrics cannot
    // overflow within this block; soft metrics normalise within smaller blocks.
    while (count - t >= block) {
        const auto end = t + block;
        // Keep the hot state local to this block. The checkpoint comparison
        // must not make its address escape and force per-step stack spills.
        auto state = metrics;
        while (t < end) {
            const auto normal_end = std::min(end, t + period * normal_cycles);
            while (t < normal_end)
                [&]<std::size_t... P> [[gnu::always_inline]] (std::index_sequence<P...>) {
                    (advance.template operator()<P>(state), ...);
                }(std::make_index_sequence<period>{});
            state.normalize();
        }
        metrics = state;
        if (reuse && t < repeated && metrics.metric == checkpoints_[t / block - 1]) {
            // Equalisation can change a few boundary decisions. Skip only a
            // contiguous, bit-identical run ending on a stored checkpoint.
            const auto begin = bits.begin() + at, end_bits = bits.begin() + overlap;
            const auto old = cached_bits_.cbegin() + at;
            const auto same = std::equal(begin, end_bits, old)
                                  ? overlap - at
                                  : std::size_t(std::mismatch(begin, end_bits, old).first - begin);
            const auto blocks = same / (sent * cycles), steps = blocks * block;
            if (steps) {
                std::copy_n(survivors_.data() + previous_tail + t, steps, survivors_.data() + t);
                reused_steps_ += steps;
                t += steps;
                at += blocks * sent * cycles;
                metrics.metric = checkpoints_[t / block - 1];
            }
        }
        if (retain && t > tail)
            next_checkpoints_[(t - tail) / block - 1] = metrics.metric;
    }
    while (at < bits.size()) {
        [&]<std::size_t... P> [[gnu::always_inline]] (std::index_sequence<P...>) {
            (advance.template operator()<P>(metrics), ...);
        }(std::make_index_sequence<period>{});
        metrics.normalize();
    }

    // Independent batches borrow worker-owned storage. A short tail or gap
    // must not force zero-initialising a large survivor buffer on the next batch.
    if constexpr (Cache)
        survivors_.resize(count);
    if (retain) {
        cached_bits_.assign(bits.end() - overlap, bits.end());
        checkpoints_.swap(next_checkpoints_);
        cached_rate_ = key;
    } else {
        cached_bits_.clear();
        checkpoints_.clear();
        cached_rate_ = 15;
    }

    metrics.normalize(); // Canonical metrics for final-state selection.
    const auto final = metrics.values();
    unsigned state = static_cast<unsigned>(std::min_element(final.begin(), final.end()) - final.begin());
    std::vector<std::uint8_t> result(count / 8);
    auto predecessor = [&] {
        --t;
        const auto bit = state & 1U;
        state = (state >> 1) | (((survivors_[t] >> state) & 1U) << 5);
        return bit;
    };
    while (t % 8)
        predecessor(); // Partial terminal byte still contributes to the path.
    while (t) {
        const auto byte = t / 8 - 1;
        result[byte] = [&]<std::size_t... I>(std::index_sequence<I...>) {
            std::uint8_t value = 0;
            ((value |= predecessor() << I), ...);
            return value;
        }(std::make_index_sequence<8>{});
    }
    return result;
}
std::vector<std::uint8_t> ReferenceViterbi::decode(std::span<const std::uint8_t> bits, unsigned rate,
                                                   std::size_t overlap) {
    if (rate >= 5)
        throw std::invalid_argument("Invalid DVB-T code rate");
    // Most batches contain only binary observations. Select once per batch so
    // puncture masks and branch weights become constants in that time loop.
    // The same trellis still handles arbitrary erased observations exactly.
    std::uint8_t combined = 0;
    for (auto bit : bits)
        combined |= bit;
    const bool erased = combined > 1;
    switch (rate) {
    case 0:
        return erased ? decode_rate<0, true>(bits, overlap) : decode_rate<0, false>(bits, overlap);
    case 1:
        return erased ? decode_rate<1, true>(bits, overlap) : decode_rate<1, false>(bits, overlap);
    case 2:
        return erased ? decode_rate<2, true>(bits, overlap) : decode_rate<2, false>(bits, overlap);
    case 3:
        return erased ? decode_rate<3, true>(bits, overlap) : decode_rate<3, false>(bits, overlap);
    case 4:
        return erased ? decode_rate<4, true>(bits, overlap) : decode_rate<4, false>(bits, overlap);
    default:
        throw std::invalid_argument("Invalid DVB-T code rate");
    }
}
std::vector<std::uint8_t> ReferenceViterbi::decode_soft(std::span<const std::uint8_t> metrics, unsigned rate,
                                                        std::size_t overlap, SoftMetric encoding) {
    if (encoding != SoftMetric::uniform && encoding != SoftMetric::minimax)
        throw std::invalid_argument("Invalid soft metric encoding");
    std::uint8_t maximum = 0;
    for (auto value : metrics)
        maximum = std::max(maximum, value);
    if (maximum > 14)
        throw std::invalid_argument("Invalid quantised likelihood");
    // Independent feedback batches never reuse a preceding trellis. Select
    // that shape once so checkpoint comparisons cannot keep the hot metrics
    // addressable inside the recursion. Both paths retain full traceback.
    const auto decode = [&]<unsigned Rate>() {
        if (encoding == SoftMetric::minimax)
            return overlap ? decode_rate<Rate, false, true, true>(metrics, overlap)
                           : decode_rate<Rate, false, true, true, false>(metrics, 0);
        return overlap ? decode_rate<Rate, false, true>(metrics, overlap)
                       : decode_rate<Rate, false, true, false, false>(metrics, 0);
    };
    switch (rate) {
    case 0:
        return decode.template operator()<0>();
    case 1:
        return decode.template operator()<1>();
    case 2:
        return decode.template operator()<2>();
    case 3:
        return decode.template operator()<3>();
    case 4:
        return decode.template operator()<4>();
    default:
        throw std::invalid_argument("Invalid DVB-T code rate");
    }
}

} // namespace radar::validation
