#pragma once
#include "fec.hpp"
#include "fec_metrics.hpp"

namespace radar {
class CpuFec;
}

namespace radar::fec {
// Exact constrained ML recursion: modular metrics for punctured rates,
// normalised uint16 metrics for rate 1/2. Preparation, interval reuse and full
// traceback are shared; no likelihood is clipped or discarded.
class ConstrainedViterbi {
    template <bool Narrow> using Metric = std::conditional_t<Narrow, std::uint8_t, std::uint16_t>;
    template <bool Narrow> using Metrics = std::array<Metric<Narrow>, 64>;
    bool narrow_{};
    static constexpr unsigned maximum_llr = minimax_llr_levels.back();
    // The wide path normalises every 128 transitions. Include the initial
    // six-step spread and terminal six-step collapse in its signed bound.
    static_assert((128 + 12) * 2 * maximum_llr < 32768);
    static constexpr auto &branches = trellis_costs;
    static constexpr auto &weights = trellis_weights;
    std::vector<std::uint8_t> observations_;
    std::vector<std::uint64_t> survivors_;
    bool have_solution_{};
    std::vector<std::int16_t> previous_;
    std::vector<std::uint8_t> bytes_;
    std::size_t decoded_steps_{};
    template <unsigned Rate> void prepare_rate(std::span<const std::uint8_t> metrics) {
        constexpr auto pattern = punctures[Rate];
        constexpr unsigned period = puncture_lengths[Rate] / 2;

        // Any two state metrics differ by at most six observations' cost.
        // Competing ACS predecessors differ ONLY in their oldest state bit.
        // Flip that input bit six transitions earlier: the two paths differ
        // only in the punctured seven-step impulse response of the encoder.
        // This tighter pairwise bound, not all seven observations, controls
        // the signed modular comparison. Known-start sections expand the
        // first six transitions explicitly, before every state is reachable.
        constexpr auto bounds = [=]() consteval {
            std::array<unsigned, 2> maximum{};
            for (unsigned start = 0; start < period; ++start) {
                unsigned spread = 0, paired = 0;
                for (unsigned t = 0; t < 7; ++t)
                    for (unsigned j = 0; j < 2; ++j) {
                        const auto observed = pattern[(2 * (start + t) + j) % (2 * period)];
                        if (t < 6)
                            spread += observed;
                        paired += observed * ((DvbtCode::symbol(1U << t) >> j) & 1);
                    }
                maximum[0] = std::max(maximum[0], spread * maximum_llr);
                maximum[1] = std::max(maximum[1], paired * maximum_llr);
            }
            return maximum;
        }();
        static_assert(Rate == 0 || (bounds[0] < 128 && bounds[1] < 128));
        constexpr unsigned sent = []() consteval {
            unsigned n = 0;
            for (auto bit : punctures[Rate])
                n += bit;
            return n;
        }();
        if (metrics.size() % sent)
            throw std::invalid_argument("Incomplete compact puncture cycle");
        observations_.resize(metrics.size() / sent * period);
        if (survivors_.size() < observations_.size())
            survivors_.resize(observations_.size());
        if (observations_.size() / 8 >= UINT32_MAX)
            throw std::invalid_argument("Oversized constrained trellis");
        previous_.assign(observations_.size() / 8, -1);
        bytes_.resize(previous_.size());
        have_solution_ = false;
        std::size_t at = 0, t = 0;
        auto append = [&]<std::size_t Phase> [[gnu::always_inline]] {
            unsigned a = 7, b = 7;
            if constexpr (pattern[2 * Phase])
                a = metrics[at++];
            if constexpr (pattern[2 * Phase + 1])
                b = metrics[at++];
            observations_[t++] = a * 15 + b;
        };
        while (at < metrics.size())
            [&]<std::size_t... P>(std::index_sequence<P...>) {
                (append.template operator()<P>(), ...);
            }(std::make_index_sequence<period>{});
        narrow_ = Rate != 0;
    }
    // Each fixed input bit halves the reachable state set. After six bits,
    // all paths have merged; the last two byte bits add only a common cost.
    // Trace the six shrinking choices to preserve exact predecessor tie order.
    unsigned collapse(std::array<std::int16_t, 64> state, std::size_t t, unsigned byte) const {
        std::array<std::uint32_t, 6> choices{};
        [&]<std::size_t... J>(std::index_sequence<J...>) {
            (
                [&] {
                    constexpr unsigned n = 32 >> J;
                    const auto observation = observations_[t + J];
                    const auto &costs = branches[observation];
                    const auto weight = weights[observation];
                    const unsigned prefix = (byte >> (8 - J)) & ((1U << J) - 1), bit = (byte >> (7 - J)) & 1;
                    std::array<std::int16_t, n> next;
                    std::array<std::uint8_t, n> decisions;
                    for (unsigned p = 0; p < n; ++p) {
                        const auto base = costs[(p << J) | prefix];
                        const auto c = bit ? weight - base : base;
                        const auto a = std::int16_t(state[p] + c),
                                   b = std::int16_t(state[p + n] + weight - c);
                        next[p] = std::min(a, b);
                        decisions[p] = b < a;
                    }
                    std::uint32_t mask = 0;
                    for (unsigned p = 0; p < n; ++p)
                        mask |= std::uint32_t(decisions[p]) << p;
                    choices[J] = mask;
                    std::copy(next.begin(), next.end(), state.begin());
                }(),
                ...);
        }(std::make_index_sequence<6>{});
        unsigned predecessor = 0;
        for (unsigned j = 6; j-- > 0;)
            predecessor += ((choices[j] >> predecessor) & 1) * (32 >> j);
        return predecessor;
    }
    template <bool Narrow>
    [[gnu::always_inline]] static std::uint64_t advance(Metrics<Narrow> &state, unsigned observation) {
        const auto &cost = branches[observation];
        const auto weight = weights[observation];
        Metrics<Narrow> next;
        std::array<std::uint8_t, 32> decisions;
        for (unsigned p = 0; p < 32; ++p) {
            const auto c = cost[p], other = std::uint8_t(weight - c);
            const auto a = Metric<Narrow>(state[p] + c), b = Metric<Narrow>(state[p + 32] + other);
            const auto d = Metric<Narrow>(state[p] + other), e = Metric<Narrow>(state[p + 32] + c);
            bool upper0, upper1;
            if constexpr (Narrow) {
                const auto delta0 = std::bit_cast<std::int8_t>(std::uint8_t(b - a));
                const auto delta1 = std::bit_cast<std::int8_t>(std::uint8_t(e - d));
                upper0 = delta0 < 0;
                upper1 = delta1 < 0;
                next[2 * p] = a + std::min(delta0, std::int8_t(0));
                next[2 * p + 1] = d + std::min(delta1, std::int8_t(0));
            } else {
                upper0 = b < a;
                upper1 = e < d;
                next[2 * p] = std::min(a, b);
                next[2 * p + 1] = std::min(d, e);
            }
            decisions[p] = (upper0 ? 1U << (2 * p % 8) : 0) | (upper1 ? 2U << (2 * p % 8) : 0);
        }
        state = next;
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
        return result;
    }

  public:
    void prepare(std::span<const std::uint8_t> metrics, unsigned rate) {
        unsigned invalid = 0;
        for (auto x : metrics)
            invalid |= x >= 15;
        if (invalid)
            throw std::invalid_argument("Invalid compact coded metrics");
        switch (rate) {
        case 0:
            return prepare_rate<0>(metrics);
        case 1:
            return prepare_rate<1>(metrics);
        case 2:
            return prepare_rate<2>(metrics);
        case 3:
            return prepare_rate<3>(metrics);
        case 4:
            return prepare_rate<4>(metrics);
        default:
            throw std::invalid_argument("Invalid DVB-T code rate");
        }
    }

  private:
    template <bool Narrow>
    void decode_range(std::size_t begin, std::size_t end, std::span<const std::int16_t> known) {
        decoded_steps_ += (end == known.size() ? observations_.size() : 8 * end) - 8 * begin;
        Metrics<Narrow> state{};
        std::size_t t = 8 * begin;
        if (begin) {
            const unsigned initial = known[begin - 1] & 63;
            [&]<std::size_t... J>(std::index_sequence<J...>) {
                (
                    [&] {
                        constexpr unsigned n = 1U << J;
                        const auto observation = observations_[t + J];
                        const auto weight = weights[observation];
                        std::array<Metric<Narrow>, 2 * n> next;
                        for (unsigned p = 0; p < n; ++p) {
                            const auto parent = ((initial << J) | p) & 63;
                            const auto base = branches[observation][parent & 31];
                            const auto c = parent & 32 ? weight - base : base;
                            next[2 * p] = state[p] + c;
                            next[2 * p + 1] = state[p] + weight - c;
                        }
                        std::copy(next.begin(), next.end(), state.begin());
                        survivors_[t + J] = (initial >> (5 - J)) & 1 ? UINT64_MAX : 0;
                    }(),
                    ...);
            }(std::make_index_sequence<6>{});
            t += 6;
        }
        const bool closed = end && known[end - 1] >= 0;
        const auto stop = closed ? 8 * (end - 1) : observations_.size();
        for (; t % 8 && t < stop; ++t)
            survivors_[t] = advance<Narrow>(state, observations_[t]);
        for (; t + 8 <= stop; t += 8) {
            [&]<std::size_t... J> [[gnu::always_inline]] (std::index_sequence<J...>) {
                ((survivors_[t + J] = advance<Narrow>(state, observations_[t + J])), ...);
            }(std::make_index_sequence<8>{});
            if constexpr (!Narrow)
                if (t % 128 == 0) {
                    const auto minimum = *std::min_element(state.begin(), state.end());
                    for (auto &value : state)
                        value -= minimum;
                }
        }
        for (; t < stop; ++t)
            survivors_[t] = advance<Narrow>(state, observations_[t]);
        std::array<std::int16_t, 64> relative;
        for (unsigned p = 0; p < 64; ++p) {
            if constexpr (Narrow)
                relative[p] = std::bit_cast<std::int8_t>(std::uint8_t(state[p] - state[0]));
            else
                relative[p] = state[p];
        }
        unsigned s = closed ? collapse(relative, t, known[end - 1])
                            : std::min_element(relative.begin(), relative.end()) - relative.begin();
        auto predecessor = [&] [[gnu::always_inline]] {
            --t;
            const auto bit = s & 1;
            s = (s >> 1) | (((survivors_[t] >> s) & 1) << 5);
            return bit;
        };
        while (t % 8)
            predecessor();
        while (t > 8 * begin) {
            const auto byte = t / 8 - 1;
            bytes_[byte] = [&]<std::size_t... B>(std::index_sequence<B...>) {
                std::uint8_t value = 0;
                ((value |= predecessor() << B), ...);
                return value;
            }(std::make_index_sequence<8>{});
        }
    }

  public:
    // A known byte fixes the complete K=7 encoder state. Newly pinned bytes
    // often agree with the previous ML path: compatible boundary subpaths stay
    // optimal by the splice argument, even when an interval splits. Revisions
    // invalidate this certificate; prepare() always starts a fresh batch.
    std::size_t decoded_steps() const {
        return decoded_steps_;
    }

  private:
    template <bool Narrow> const std::vector<std::uint8_t> &decode(std::span<const std::int16_t> known) {
        decoded_steps_ = 0;
        if (known.size() != observations_.size() / 8)
            throw std::invalid_argument("Invalid constrained-byte sequence");
        // Both reductions scan the same resident bytes. Integer flags expose
        // SIMD validation without changing the accepted range or cache rules.
        unsigned invalid = 0, revised = 0;
        for (std::size_t i = 0; i < known.size(); ++i) {
            invalid |= std::uint16_t(known[i] + 1) > 256;
            revised |= unsigned(previous_[i] >= 0) & unsigned(previous_[i] != known[i]);
        }
        if (invalid)
            throw std::invalid_argument("Invalid constrained-byte sequence");
        const bool certified = have_solution_ && !revised;
        std::size_t at = 0;
        bool start_matches = true;
        while (at < known.size()) {
            if (known[at] >= 0) {
                start_matches = (known[at] & 63) == (bytes_[at] & 63);
                bytes_[at] = known[at];
                ++at;
                continue;
            }
            const auto begin = at;
            while (at < known.size() && known[at] < 0)
                ++at;
            // A cheaper subpath between states on the previous ML path could
            // replace that subpath and contradict its optimality. Thus newly
            // split sections need no ACS when both boundaries still agree.
            const bool closed = at < known.size();
            const bool end_matches = !closed || known[at] == bytes_[at];
            const bool next_start_matches = closed && (known[at] & 63) == (bytes_[at] & 63);
            if (!certified || !start_matches || !end_matches)
                decode_range<Narrow>(begin, closed ? at + 1 : at, known);
            // Traceback includes the closing byte. Save its old state comparison
            // before decoding overwrites it, then consume that fixed byte here.
            if (closed) {
                bytes_[at] = known[at];
                ++at;
                start_matches = next_start_matches;
            }
        }
        previous_.assign(known.begin(), known.end());
        have_solution_ = true;
        return bytes_;
    }

    // Only CpuFec may certify its initial full-traceback minimax solution for
    // these exact observations. An arbitrary tentative path is not a certificate.
    friend class radar::CpuFec;
    void seed(std::span<const std::uint8_t> path) {
        if (path.size() != bytes_.size())
            throw std::invalid_argument("Invalid initial trellis path");
        bytes_.assign(path.begin(), path.end());
        have_solution_ = true;
    }

  public:
    const std::vector<std::uint8_t> &operator()(std::span<const std::int16_t> known) {
        return narrow_ ? decode<true>(known) : decode<false>(known);
    }
};
} // namespace radar::fec
