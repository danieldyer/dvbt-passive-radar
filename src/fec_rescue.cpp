#include "fec_rescue.hpp"
#include "fec_metrics.hpp"
#include <numeric>

namespace radar::fec {
bool RsRescue::eligible(std::span<const std::int16_t> known, std::size_t failed) {
    if (!failed || failed > maximum_failed)
        return false;
    return std::ranges::count(known, -1) <= maximum_unknown_bytes;
}
void RsRescue::prepare(std::span<const std::uint8_t> coded, unsigned rate) {
    const auto pattern = puncture_pattern(rate);
    const auto sent = std::count(pattern.begin(), pattern.end(), 1U);
    std::uint8_t maximum = 0;
    for (auto value : coded)
        maximum = std::max(maximum, value);
    if (coded.size() % sent || maximum >= 15)
        throw std::invalid_argument("Invalid RS rescue compact metrics");
    coded_ = coded;
    rate_ = rate;
    steps_ = coded.size() / sent * (pattern.size() / 2);
}
template <unsigned Rate> void RsRescue::estimate_rate(std::span<const std::int16_t> known) {
    constexpr auto pattern = punctures[Rate];
    constexpr unsigned period = puncture_lengths[Rate] / 2;
    constexpr unsigned sent = []() consteval {
        unsigned n = 0;
        for (auto x : punctures[Rate])
            n += x;
        return n;
    }();
    static constexpr auto offsets = [=]() consteval {
        std::array<unsigned, period> out{};
        for (unsigned p = 1; p < period; ++p)
            out[p] = out[p - 1] + pattern[2 * p - 2] + pattern[2 * p - 1];
        return out;
    }();
    static constexpr auto table = []() consteval {
        std::array<Cost, 225> out{};
        constexpr std::array<unsigned, 4> position{0, 1, 3, 2};
        for (unsigned observation = 0; observation < out.size(); ++observation)
            for (unsigned label = 0; label < 4; ++label)
                out[observation][label] = trellis_costs[observation][position[label]];
        return out;
    }();
    alternatives_.resize(known.size());
    for (std::size_t at = 0; at < known.size();) {
        if (known[at] >= 0) {
            alternatives_[at] = {std::uint8_t(known[at]), std::uint8_t(known[at]), 1000000};
            ++at;
            continue;
        }
        const auto begin = at;
        while (at < known.size() && known[at] < 0)
            ++at;
        // Six bits of a known right byte fix its state; the last two add a
        // common cost and cannot change an unknown byte's candidate ranking.
        const auto end = at < known.size() ? 8 * at + 6 : steps_;
        costs_.resize(end - 8 * begin);
        for (std::size_t j = 0; j < costs_.size(); ++j) {
            const auto t = 8 * begin + j, phase = t % period;
            auto input = t / period * sent + offsets[phase];
            const unsigned a = pattern[2 * phase] ? coded_[input++] : 7;
            const unsigned b = pattern[2 * phase + 1] ? coded_[input] : 7;
            costs_[j] = table[a * 15 + b];
        }
        interval(begin, begin ? known[begin - 1] & 63 : -1, at < known.size() ? known[at] >> 2 : -1);
    }
}
void RsRescue::estimate(std::span<const std::int16_t> known) {
    // One vectorisable scan validates constraints and the bounded work size.
    std::int16_t minimum = 0, maximum = 0;
    std::size_t unknown = 0;
    for (auto value : known) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        unknown += value == -1;
    }
    if (known.size() != steps_ / 8 || minimum < -1 || maximum > 255 || unknown > maximum_unknown_bytes)
        throw std::invalid_argument("Invalid RS rescue byte coordinates or work bound");
    switch (rate_) {
    case 0:
        return estimate_rate<0>(known);
    case 1:
        return estimate_rate<1>(known);
    case 2:
        return estimate_rate<2>(known);
    case 3:
        return estimate_rate<3>(known);
    case 4:
        return estimate_rate<4>(known);
    default:
        throw std::invalid_argument("Invalid RS rescue rate");
    }
}
void RsRescue::interval(std::size_t byte_begin, int initial, int final) {
    // Exact integer max-log recursion. The work cap bounds cumulative costs,
    // so no per-transition normalisation is needed; common offsets cancel in
    // the byte likelihood gap. Six transitions make every state reachable.
    constexpr std::int32_t unreachable = -(1 << 24);
    constexpr auto maximum_cost = 2LL * minimax_llr_levels.back();
    constexpr auto maximum_steps = 8LL * maximum_unknown_bytes + 7;
    static_assert(6 * maximum_cost < -unreachable);
    static_assert(2 * (-std::int64_t(unreachable) + maximum_steps * maximum_cost) + 2 * maximum_cost <
                  INT32_MAX);
    constexpr unsigned block = 4096;
    // A branch cost separates into its two coded-bit costs. Compile-time
    // masks expose contiguous butterfly arithmetic instead of indexed gathers.
    static constexpr auto masks = []() consteval {
        std::array<std::array<int, 32>, 2> out{};
        for (unsigned p = 0; p < 32; ++p)
            for (unsigned b = 0; b < 2; ++b)
                out[b][p] = -int((DvbtCode::branches[2 * p] >> b) & 1);
        return out;
    }();
    const auto forward = [&](const State &old, const Cost &cost) {
        State next;
        const int base = cost[0], dx = cost[1] - base, dy = cost[2] - base, total = cost[0] + cost[3];
        for (unsigned p = 0; p < 32; ++p) {
            const int c = base + (dx & masks[0][p]) + (dy & masks[1][p]), d = total - c;
            next[2 * p] = std::max(old[p] - c, old[p + 32] - d);
            next[2 * p + 1] = std::max(old[p] - d, old[p + 32] - c);
        }
        return next;
    };
    State alpha{};
    if (initial >= 0) {
        alpha.fill(unreachable);
        alpha[initial] = 0;
    }
    checkpoints_.resize((costs_.size() + block - 1) / block);
    for (std::size_t group = 0; group < checkpoints_.size(); ++group) {
        checkpoints_[group] = alpha;
        if (group + 1 < checkpoints_.size())
            for (auto t = group * block; t < (group + 1) * block; ++t)
                alpha = forward(alpha, costs_[t]);
    }
    // Byte scores use only alpha after the six-bit prefix. Keep those
    // states alone: 128 KiB per checkpoint block instead of 1 MiB.
    scratch_.resize((std::min<std::size_t>(block, costs_.size()) + 7) / 8);
    State beta{}, byte_end{};
    if (final >= 0) {
        beta.fill(unreachable);
        beta[final] = 0;
    }
    for (std::size_t group = checkpoints_.size(); group-- > 0;) {
        const auto begin = group * block, end = std::min(begin + block, costs_.size());
        auto state = checkpoints_[group];
        for (auto t = begin; t < end; ++t) {
            state = forward(state, costs_[t]);
            if ((t - begin) % 8 == 5)
                scratch_[(t - begin) / 8] = state;
        }
        for (auto t = end; t-- > begin;) {
            if (t % 8 == 7)
                byte_end = beta;
            if (t % 8 == 0 && t + 8 <= costs_.size()) {
                std::array<std::int32_t, 256> score;
                static constexpr auto byte_masks = []() consteval {
                    std::array<std::array<int, 256>, 4> out{};
                    for (unsigned w = 0; w < 256; ++w)
                        for (unsigned b = 0; b < 2; ++b) {
                            out[b][w] = -int((DvbtCode::symbol(w >> 1) >> b) & 1);
                            out[b + 2][w] = -int((DvbtCode::symbol(w & 127) >> b) & 1);
                        }
                    return out;
                }();
                const auto &a = costs_[t + 6];
                const auto &b = costs_[t + 7];
                const int ax = a[1] - a[0], ay = a[2] - a[0], bx = b[1] - b[0], by = b[2] - b[0],
                          base = a[0] + b[0];
                // Four suffixes share one alpha value and contiguous beta
                // values. This layout auto-vectorises without intrinsics.
                for (unsigned prefix = 0; prefix < 64; ++prefix) {
                    const int common = scratch_[(t - begin) / 8][prefix] - base;
                    for (unsigned tail = 0; tail < 4; ++tail) {
                        const auto word = 4 * prefix + tail;
                        score[word] = common + byte_end[4 * (prefix & 15) + tail] -
                                      (ax & byte_masks[0][word]) - (ay & byte_masks[1][word]) -
                                      (bx & byte_masks[2][word]) - (by & byte_masks[3][word]);
                    }
                }
                auto best_score = score[0];
                for (auto value : score)
                    best_score = std::max(best_score, value);
                const unsigned best = std::find(score.begin(), score.end(), best_score) - score.begin();
                score[best] = std::numeric_limits<std::int32_t>::min();
                auto second_score = score[0];
                for (auto value : score)
                    second_score = std::max(second_score, value);
                const unsigned second = std::find(score.begin(), score.end(), second_score) - score.begin();
                alternatives_[byte_begin + t / 8] = {std::uint8_t(best), std::uint8_t(second),
                                                     best_score - second_score};
            }
            State next;
            const auto &cost = costs_[t];
            const int base = cost[0], dx = cost[1] - base, dy = cost[2] - base, total = cost[0] + cost[3];
            for (unsigned p = 0; p < 32; ++p) {
                const int c = base + (dx & masks[0][p]) + (dy & masks[1][p]), d = total - c;
                next[p] = std::max(beta[2 * p] - c, beta[2 * p + 1] - d);
                next[p + 32] = std::max(beta[2 * p] - d, beta[2 * p + 1] - c);
            }
            beta = next;
        }
    }
}
unsigned RsRescue::correct(std::span<RsResult> packets, std::span<const std::uint8_t> selected) const {
    if (selected.size() != alternatives_.size() ||
        (selected.size() / 204 > 11 ? selected.size() / 204 - 11 : 0) != packets.size() ||
        std::ranges::count_if(packets, [](const auto &p) { return !p.valid; }) > maximum_failed)
        throw std::invalid_argument("Invalid RS rescue packet coordinates or work bound");
    unsigned added = 0;
    for (std::size_t p = 0; p < packets.size(); ++p)
        if (!packets[p].valid) {
            DvbtRs::Word received, alternative;
            std::array<std::int32_t, 204> gap;
            for (unsigned j = 0; j < 204; ++j) {
                const auto i = 2244 + p * 204 + j, source = i - (11 - i % 12) * 204;
                received[j] = selected[source];
                const auto &a = alternatives_[source];
                const bool same = a.best == received[j];
                alternative[j] = same ? a.second : a.best;
                gap[j] = same ? a.gap : 0;
            }
            std::array<unsigned, 204> order;
            std::iota(order.begin(), order.end(), 0U);
            std::partial_sort(order.begin(), order.begin() + positions, order.end(),
                              [&](auto a, auto b) { return gap[a] != gap[b] ? gap[a] < gap[b] : a < b; });
            DvbtRs::Word winner{};
            bool found = false, ambiguous = false;
            for (unsigned mask = 0; mask < (1U << positions); ++mask) {
                auto candidate = received;
                for (unsigned bit = 0; bit < positions; ++bit)
                    if (mask & (1U << bit))
                        candidate[order[bit]] = alternative[order[bit]];
                if (DvbtRs::decode(candidate) < 0 || (candidate[0] != 0x47 && candidate[0] != 0xb8))
                    continue;
                if (found && candidate != winner) {
                    ambiguous = true;
                    break;
                }
                winner = candidate;
                found = true;
            }
            if (!found || ambiguous)
                continue;
            auto &packet = packets[p];
            packet.bytes = winner;
            packet.valid = true;
            packet.corrected = 0;
            for (unsigned j = 0; j < 204; ++j)
                packet.corrected += winner[j] != received[j];
            ++added;
        }
    return added;
}
} // namespace radar::fec
