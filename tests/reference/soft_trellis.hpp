#pragma once
// Experimental CPU references. Production FEC and scientific samples unchanged.
#include "fec.hpp"
#include <cmath>
#include <limits>

namespace radar::validation {
using StateMetrics = std::array<float, 64>;
using BranchMetrics = std::array<float, 4>;
inline constexpr auto survivor_masks = []() consteval {
    std::array<std::array<std::uint8_t, 32>, 2> masks{};
    for (unsigned p = 0; p < 32; ++p) {
        masks[0][p] = 1U << (2 * p % 8);
        masks[1][p] = 2U << (2 * p % 8);
    }
    return masks;
}();
inline std::uint64_t pack_survivors(const std::array<std::uint8_t, 32> &decisions) {
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
    auto mask = std::bit_cast<std::uint64_t>(packed);
    if constexpr (std::endian::native == std::endian::big)
        mask = std::byteswap(mask);
    return mask;
}
inline std::vector<BranchMetrics> soft_branches(std::span<const float> llr, unsigned rate) {
    const auto pattern = fec::puncture_pattern(rate);
    const auto sent = std::count(pattern.begin(), pattern.end(), 1U);
    if (llr.size() % sent)
        throw std::invalid_argument("Incomplete soft puncture cycle");
    std::vector<BranchMetrics> out(llr.size() / sent * (pattern.size() / 2));
    std::size_t at = 0;
    for (std::size_t t = 0; t < out.size(); ++t) {
        const auto phase = 2 * t % pattern.size();
        const float a = pattern[phase] ? llr[at++] : 0;
        const float b = pattern[phase + 1] ? llr[at++] : 0;
        if (!std::isfinite(a) || !std::isfinite(b))
            throw std::invalid_argument("Nonfinite likelihood");
        out[t] = {.5F * (a + b), .5F * (-a + b), .5F * (a - b), .5F * (-a - b)};
    }
    return out;
}
template <class Score> inline void normalize(std::array<Score, 64> &values) {
    const auto maximum = *std::max_element(values.begin(), values.end());
    for (auto &v : values)
        v -= maximum;
}
template <class Score>
inline std::vector<std::uint8_t> soft_viterbi_t(std::span<const std::array<Score, 4>> branches) {
    std::array<Score, 64> state{};
    std::vector<std::uint64_t> survivors(branches.size());
    for (std::size_t t = 0; t < branches.size(); ++t) {
        std::array<Score, 64> next;
        std::array<std::uint8_t, 32> decisions;
        for (unsigned p = 0; p < 32; ++p) {
            const auto branch = branches[t][fec::DvbtCode::branches[2 * p]];
            const Score a = state[p] + branch, b = state[p + 32] - branch;
            const Score c = state[p] - branch, d = state[p + 32] + branch;
            next[2 * p] = std::max(a, b);
            next[2 * p + 1] = std::max(c, d);
            decisions[p] = (b > a ? 1U << (2 * p % 8) : 0) | (d > c ? 2U << (2 * p % 8) : 0);
        }
        survivors[t] = pack_survivors(decisions);
        state = next;
        if (t % 32 == 31)
            normalize(state);
    }
    unsigned s = std::max_element(state.begin(), state.end()) - state.begin();
    std::vector<std::uint8_t> bytes(branches.size() / 8);
    for (std::size_t t = branches.size(); t-- > 0;) {
        if (t < bytes.size() * 8)
            bytes[t / 8] |= (s & 1) << (7 - t % 8);
        s = (s >> 1) | (((survivors[t] >> s) & 1) << 5);
    }
    return bytes;
}
inline std::vector<std::uint8_t> soft_viterbi(std::span<const BranchMetrics> branches) {
    return soft_viterbi_t<float>(branches);
}
using QuantizedBranch = std::array<std::int16_t, 4>;
using ByteCost = std::array<std::uint8_t, 4>;
// Bounded rounding, ties away from zero; conversion is safe after validation.
inline int byte_quantize(float x) {
    const auto bounded = std::clamp(x, -7.F, 7.F);
    return int(double(bounded) + std::copysign(.5, double(bounded)));
}
template <unsigned Rate> inline std::vector<ByteCost> byte_costs_rate(std::span<const float> llr) {
    constexpr auto pattern = fec::punctures[Rate];
    constexpr unsigned period = fec::puncture_lengths[Rate] / 2;
    constexpr unsigned sent = [] {
        unsigned n = 0;
        for (auto bit : fec::punctures[Rate])
            n += bit;
        return n;
    }();
    if (llr.size() % sent)
        throw std::invalid_argument("Incomplete byte-metric puncture cycle");
    std::vector<ByteCost> out(llr.size() / sent * period);
    std::size_t at = 0, t = 0;
    auto advance = [&]<std::size_t Phase>() {
        int a = 0, b = 0;
        if constexpr (pattern[2 * Phase])
            a = byte_quantize(llr[at++]);
        if constexpr (pattern[2 * Phase + 1])
            b = byte_quantize(llr[at++]);
        out[t++] = {
            std::uint8_t(std::max(-a, 0) + std::max(-b, 0)), std::uint8_t(std::max(a, 0) + std::max(-b, 0)),
            std::uint8_t(std::max(-a, 0) + std::max(b, 0)), std::uint8_t(std::max(a, 0) + std::max(b, 0))};
    };
    while (t < out.size())
        [&]<std::size_t... P>(std::index_sequence<P...>) {
            (advance.template operator()<P>(), ...);
        }(std::make_index_sequence<period>{});
    return out;
}
inline std::vector<ByteCost> byte_costs(std::span<const float> llr, unsigned rate) {
    if (!std::ranges::all_of(llr, [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Nonfinite LLR");
    switch (rate) {
    case 0:
        return byte_costs_rate<0>(llr);
    case 1:
        return byte_costs_rate<1>(llr);
    case 2:
        return byte_costs_rate<2>(llr);
    case 3:
        return byte_costs_rate<3>(llr);
    case 4:
        return byte_costs_rate<4>(llr);
    default:
        throw std::invalid_argument("Invalid DVB-T code rate");
    }
}
inline std::vector<std::uint8_t> byte_viterbi(std::span<const ByteCost> costs) {
    // Expand the 225 quantized observation pairs once. Contiguous butterfly
    // costs avoid an indexed byte gather inside every trellis step.
    static constexpr auto table = []() consteval {
        std::array<std::array<std::uint8_t, 32>, 225> result{};
        for (int a = -7; a <= 7; ++a)
            for (int b = -7; b <= 7; ++b)
                for (unsigned p = 0; p < 32; ++p) {
                    const auto label = fec::DvbtCode::branches[2 * p];
                    result[(a + 7) * 15 + b + 7][p] =
                        std::max(label & 1 ? a : -a, 0) + std::max(label & 2 ? b : -b, 0);
                }
        return result;
    }();
    std::array<std::uint8_t, 64> state{};
    std::vector<std::uint64_t> survivors(costs.size());
    const auto less = [](std::uint8_t a, std::uint8_t b) {
        return std::bit_cast<std::int8_t>(std::uint8_t(a - b)) < 0;
    };
    // Each branch costs 0..14. Any two K=7 states can be connected in six
    // steps, so state spread <=84, candidate spread <=98<128. Signed modular
    // differences therefore compare exact unbounded costs: no normalization.
    for (std::size_t t = 0; t < costs.size(); ++t) {
        std::array<std::uint8_t, 64> next;
        std::array<std::uint8_t, 32> decisions;
        const unsigned weight = costs[t][0] + costs[t][3];
        const auto key = (int(costs[t][1]) - costs[t][0] + 7) * 15 + int(costs[t][2]) - costs[t][0] + 7;
        const auto *branches = table[key].data();
        for (unsigned p = 0; p < 32; ++p) {
            const auto branch = branches[p];
            const std::uint8_t complement = weight - branch;
            const std::uint8_t a = state[p] + branch, b = state[p + 32] + complement,
                               c = state[p] + complement, d = state[p + 32] + branch;
            const std::uint8_t hi0 = 0U - unsigned(less(b, a)), hi1 = 0U - unsigned(less(d, c));
            next[2 * p] = a ^ ((a ^ b) & hi0);
            next[2 * p + 1] = c ^ ((c ^ d) & hi1);
            decisions[p] = (hi0 & survivor_masks[0][p]) | (hi1 & survivor_masks[1][p]);
        }
        survivors[t] = pack_survivors(decisions);
        state = next;
    }
    unsigned s = 0;
    for (unsigned j = 1; j < 64; ++j)
        if (less(state[j], state[s]))
            s = j;
    std::vector<std::uint8_t> bytes(costs.size() / 8);
    for (std::size_t t = costs.size(); t-- > 0;) {
        if (t < bytes.size() * 8)
            bytes[t / 8] |= (s & 1) << (7 - t % 8);
        s = (s >> 1) | (((survivors[t] >> s) & 1) << 5);
    }
    return bytes;
}
inline std::vector<QuantizedBranch> quantized_branches(std::span<const float> llr, unsigned rate) {
    const auto pattern = fec::puncture_pattern(rate);
    const auto sent = std::count(pattern.begin(), pattern.end(), 1U);
    if (llr.size() % sent)
        throw std::invalid_argument("Incomplete quantized puncture cycle");
    std::vector<QuantizedBranch> out(llr.size() / sent * (pattern.size() / 2));
    std::size_t at = 0;
    const auto quantize = [](float x) {
        if (!std::isfinite(x))
            throw std::invalid_argument("Nonfinite LLR");
        return int(std::round(std::clamp(4 * x, -127.F, 127.F)));
    };
    for (std::size_t t = 0; t < out.size(); ++t) {
        const auto phase = 2 * t % pattern.size();
        const int a = pattern[phase] ? quantize(llr[at++]) : 0;
        const int b = pattern[phase + 1] ? quantize(llr[at++]) : 0;
        out[t] = {std::int16_t(a + b), std::int16_t(-a + b), std::int16_t(a - b), std::int16_t(-a - b)};
    }
    // |branch| <=254; 32 steps between normalizations plus the six-step
    // state spread stays well within signed 16-bit metric range.
    return out;
}
inline float log_add(float a, float b) {
    const auto hi = std::max(a, b), lo = std::min(a, b);
    if (!std::isfinite(hi))
        return hi;
    return hi + std::log1p(std::exp(lo - hi));
}
inline StateMetrics forward_map(const StateMetrics &state, const BranchMetrics &branch, float prior = 0) {
    StateMetrics next;
    for (unsigned s = 0; s < 64; ++s) {
        const auto label = fec::DvbtCode::branches[s];
        next[s] = log_add(state[s / 2] + branch[label], state[s / 2 + 32] + branch[label ^ 3]) +
                  (s & 1 ? -.5F : .5F) * prior;
    }
    normalize(next);
    return next;
}
// Exact full-block log-MAP (up to floating-point error), unknown endpoint states.
// Checkpoints bound memory; each block is recomputed, not independently decoded.
template <bool Coded = false>
inline std::vector<float> bcjr_map(std::span<const BranchMetrics> branches,
                                   std::span<const float> prior = {}) {
    if (!prior.empty() && (prior.size() != branches.size() ||
                           !std::ranges::all_of(prior, [](float x) { return std::isfinite(x); })))
        throw std::invalid_argument("Invalid information-bit prior");
    constexpr unsigned block = 4096;
    std::vector<StateMetrics> checkpoints((branches.size() + block - 1) / block);
    StateMetrics alpha{};
    for (std::size_t t = 0; t < branches.size(); ++t) {
        if (t % block == 0)
            checkpoints[t / block] = alpha;
        alpha = forward_map(alpha, branches[t], prior.empty() ? 0 : prior[t]);
    }
    std::vector<float> output(branches.size() * (Coded ? 2 : 1));
    std::vector<StateMetrics> scratch(block);
    StateMetrics beta{};
    for (std::size_t group = checkpoints.size(); group-- > 0;) {
        const auto begin = group * block, end = std::min(begin + block, branches.size());
        alpha = checkpoints[group];
        for (auto t = begin; t < end; ++t) {
            scratch[t - begin] = alpha;
            alpha = forward_map(alpha, branches[t], prior.empty() ? 0 : prior[t]);
        }
        for (auto t = end; t-- > begin;) {
            StateMetrics next;
            float zero = -std::numeric_limits<float>::infinity(), one = zero;
            BranchMetrics encoded;
            encoded.fill(-INFINITY);
            for (unsigned s = 0; s < 64; ++s) {
                const auto d = (2 * s) & 63;
                const auto label = fec::DvbtCode::symbol(2 * s);
                const float input = prior.empty() ? 0 : .5F * prior[t];
                const float a = branches[t][label] + beta[d] + input;
                const float b = branches[t][label ^ 3] + beta[d + 1] - input;
                next[s] = log_add(a, b);
                if constexpr (Coded) {
                    encoded[label] = log_add(encoded[label], scratch[t - begin][s] + a);
                    encoded[label ^ 3] = log_add(encoded[label ^ 3], scratch[t - begin][s] + b);
                } else {
                    zero = log_add(zero, scratch[t - begin][s] + a);
                    one = log_add(one, scratch[t - begin][s] + b);
                }
            }
            if constexpr (Coded) {
                output[2 * t] = log_add(encoded[0], encoded[2]) - log_add(encoded[1], encoded[3]);
                output[2 * t + 1] = log_add(encoded[0], encoded[1]) - log_add(encoded[2], encoded[3]);
            } else
                output[t] = zero - one;
            normalize(next);
            beta = next;
        }
    }
    return output;
}
inline std::vector<float> bcjr(std::span<const BranchMetrics> branches, std::span<const float> prior = {}) {
    return bcjr_map(branches, prior);
}
inline std::vector<std::uint8_t> posterior_bytes(std::span<const float> posterior) {
    std::vector<std::uint8_t> bytes(posterior.size() / 8);
    for (std::size_t i = 0; i < bytes.size() * 8; ++i)
        bytes[i / 8] |= unsigned(posterior[i] < 0) << (7 - i % 8);
    return bytes;
}
} // namespace radar::validation
