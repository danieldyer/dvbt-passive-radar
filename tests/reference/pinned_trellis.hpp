#pragma once
// Exact integer reference for RS-constrained, quantised convolutional decoding.
// No posterior approximation: unreachable states use a separate sentinel.
#include "soft_quantizer.hpp"
#include "soft_trellis.hpp"
#include <type_traits>

namespace radar::validation {
// Fixed minimax posterior-log-loss quantiser. See the reproducible design;
// the bound does not cover likelihood mismatch from channel/noise estimation.
inline int companded_feedback(float value) {
    const float magnitude = std::abs(value);
    unsigned rank = 0;
    for (auto threshold : fec::minimax_llr_thresholds)
        rank += magnitude >= threshold;
    return value < 0 ? -fec::minimax_llr_levels[rank] : fec::minimax_llr_levels[rank];
}
template <unsigned Scale, bool Companded = false>
inline std::vector<ByteCost> feedback_costs(std::span<const float> llr, unsigned rate) {
    static_assert(Scale == 1 || Scale == 2 || Scale == 4);
    static_assert(!Companded || Scale == 2);
    const auto pattern = fec::puncture_pattern(rate);
    const unsigned sent = std::count(pattern.begin(), pattern.end(), 1U);
    if (llr.size() % sent || !std::ranges::all_of(llr, [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Invalid feedback likelihoods");
    std::vector<ByteCost> result(llr.size() / sent * (pattern.size() / 2));
    std::size_t at = 0;
    const auto quantize = [](float x) {
        if constexpr (Companded)
            return companded_feedback(x);
        else
            return int(std::round(Scale * std::clamp(x, -7.F, 7.F)));
    };
    for (std::size_t t = 0; t < result.size(); ++t) {
        const auto phase = 2 * t % pattern.size();
        const int a = pattern[phase] ? quantize(llr[at++]) : 0;
        const int b = pattern[phase + 1] ? quantize(llr[at++]) : 0;
        result[t] = {
            std::uint8_t(std::max(-a, 0) + std::max(-b, 0)), std::uint8_t(std::max(a, 0) + std::max(-b, 0)),
            std::uint8_t(std::max(-a, 0) + std::max(b, 0)), std::uint8_t(std::max(a, 0) + std::max(b, 0))};
    }
    return result;
}
template <unsigned Maximum = 7, unsigned Rate = 5, bool FinitePenalty = false>
inline std::vector<std::uint8_t> pinned_viterbi(std::span<const ByteCost> costs,
                                                std::span<const std::int16_t> known) {
    static_assert(Maximum == 7 || Maximum == 14 || Maximum == 28);
    static_assert(Rate <= 5);
    static constexpr unsigned normalize_period = []() consteval {
        if constexpr (FinitePenalty)
            return 96U;
        else if constexpr (Rate == 5 || Maximum == 28)
            return 8U;
        else {
            constexpr unsigned period = fec::puncture_lengths[Rate] / 2;
            const auto observations = [](unsigned start, unsigned count) {
                unsigned n = 0;
                for (unsigned t = 0; t < count; ++t)
                    for (unsigned b = 0; b < 2; ++b)
                        n += fec::punctures[Rate][(2 * (start + t) + b) % fec::puncture_lengths[Rate]];
                return n;
            };
            unsigned spread = 0;
            for (unsigned phase = 0; phase < period; ++phase)
                spread = std::max(spread, observations(phase, 6));
            unsigned n = 0;
            while (Maximum * (spread + observations(0, n + period)) < 255)
                n += period;
            if (!n)
                throw "Puncture-aware byte metric has no safe normalization period";
            return n;
        }
    }();
    using Metric = std::conditional_t<!FinitePenalty && (Maximum == 7 || (Maximum == 14 && Rate < 5)),
                                      std::uint8_t, std::uint16_t>;
    if (known.size() != costs.size() / 8 ||
        !std::ranges::all_of(known, [](auto x) { return x >= -1 && x <= 255; }))
        throw std::invalid_argument("Pinned byte coordinates or values invalid");
    static constexpr auto table = []() consteval {
        // Factor the two coded observations so increasing precision does not
        // grow the table quadratically. Each row is contiguous for vector loads.
        std::array<std::array<std::array<Metric, 32>, 2 * Maximum + 1>, 2> result{};
        for (unsigned bit = 0; bit < 2; ++bit)
            for (int value = -int(Maximum); value <= int(Maximum); ++value)
                for (unsigned p = 0; p < 32; ++p)
                    result[bit][value + Maximum][p] =
                        std::max(fec::DvbtCode::branches[2 * p] & (1U << bit) ? value : -value, 0);
        return result;
    }();
    constexpr Metric infinity = []() consteval {
        if constexpr (FinitePenalty)
            return Metric{32768};
        else
            return std::numeric_limits<Metric>::max();
    }();
    static_assert(Rate < 5 || 28 * Maximum < infinity);
    static_assert(!FinitePenalty || 2 * (2 * normalize_period + 12) * Maximum < infinity);
    std::array<Metric, 64> state{};
    std::vector<std::uint64_t> survivors(costs.size());
    for (std::size_t t = 0; t < costs.size(); ++t) {
        // A complete known byte fixes the K=7 state. Consecutive known bytes
        // therefore have exactly one permitted path: its common cost cancels
        // under normalisation, and its predecessor is known during traceback.
        if (t % 8 == 0 && t >= 8 && t / 8 < known.size() && known[t / 8] >= 0 && known[t / 8 - 1] >= 0) {
            state.fill(infinity);
            state[known[t / 8] & 63] = 0;
            t += 7;
            continue;
        }
        const auto &c = costs[t];
        const int a = int(c[1]) - c[0], b = int(c[2]) - c[0];
        if constexpr (Rate < 5)
            if ((!fec::punctures[Rate][2 * t % fec::puncture_lengths[Rate]] && a) ||
                (!fec::punctures[Rate][(2 * t + 1) % fec::puncture_lengths[Rate]] && b))
                throw std::invalid_argument("Branch cost violates the declared puncture pattern");
        if (std::abs(a) > Maximum || std::abs(b) > Maximum || c[0] != std::max(-a, 0) + std::max(-b, 0) ||
            c[0] + c[3] > 2 * Maximum || c[0] + c[3] != c[1] + c[2])
            throw std::invalid_argument("Invalid bounded branch cost");
        const auto &first = table[0][a + Maximum];
        const auto &second = table[1][b + Maximum];
        const auto weight = c[0] + c[3];
        const auto byte = t / 8 < known.size() ? known[t / 8] : -1;
        const int bit = byte < 0 ? -1 : (byte >> (7 - t % 8)) & 1;
        const Metric forbid_zero = bit == 1 ? infinity : 0, forbid_one = bit == 0 ? infinity : 0;
        const Metric zero_mask = Metric(0U - unsigned(bit == 1)), one_mask = Metric(0U - unsigned(bit == 0));
        std::array<Metric, 64> next;
        std::array<std::uint8_t, 32> decisions;
        for (unsigned p = 0; p < 32; ++p) {
            const auto cost = Metric(first[p] + second[p]), complement = Metric(weight - cost);
            // OR with an all-ones mask preserves unreachable states without
            // control flow inside the independent butterfly loop.
            const auto lo_mask = FinitePenalty ? Metric{0} : Metric(0U - unsigned(state[p] == infinity));
            const auto hi_mask = FinitePenalty ? Metric{0} : Metric(0U - unsigned(state[p + 32] == infinity));
            const auto lo0 = Metric(Metric(state[p] + cost) | lo_mask);
            const auto hi0 = Metric(Metric(state[p + 32] + complement) | hi_mask);
            const auto lo1 = Metric(Metric(state[p] + complement) | lo_mask);
            const auto hi1 = Metric(Metric(state[p + 32] + cost) | hi_mask);
            if constexpr (FinitePenalty) {
                next[2 * p] = (std::min(lo0, hi0) & Metric(~zero_mask)) | forbid_zero;
                next[2 * p + 1] = (std::min(lo1, hi1) & Metric(~one_mask)) | forbid_one;
            } else {
                next[2 * p] = std::min(lo0, hi0) | forbid_zero;
                next[2 * p + 1] = std::min(lo1, hi1) | forbid_one;
            }
            decisions[p] = (hi0 < lo0 ? 1U << (2 * p % 8) : 0) | (hi1 < lo1 ? 2U << (2 * p % 8) : 0);
        }
        survivors[t] = pack_survivors(decisions);
        state = next;
        if (t % normalize_period == normalize_period - 1) {
            // Every reachable final state is defined by six permitted input
            // bits. Follow those bits from the best state six steps earlier:
            // reachable spread <=12*Maximum, even with pins. Eight more
            // steps add <=16*Maximum, below the selected sentinel.
            const auto minimum = *std::min_element(state.begin(), state.end());
            for (auto &value : state)
                if (FinitePenalty || value != infinity)
                    value -= minimum;
        }
    }
    unsigned s = std::min_element(state.begin(), state.end()) - state.begin();
    std::vector<std::uint8_t> bytes(costs.size() / 8);
    for (std::size_t t = costs.size(); t-- > 0;) {
        if (t % 8 == 7 && t >= 15 && known[t / 8] >= 0 && known[t / 8 - 1] >= 0) {
            bytes[t / 8] = known[t / 8];
            s = known[t / 8 - 1] & 63;
            t -= 7;
            continue;
        }
        if (t < bytes.size() * 8)
            bytes[t / 8] |= (s & 1) << (7 - t % 8);
        s = (s >> 1) | (((survivors[t] >> s) & 1) << 5);
    }
    return bytes;
}
} // namespace radar::validation
