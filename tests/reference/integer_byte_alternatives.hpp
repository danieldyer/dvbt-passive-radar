#pragma once
#include "soft_rs_reference.hpp"

namespace radar::validation {
// Exact max-log recursion for integer branch costs. The K=7 encoder forgets
// an endpoint after six transitions, so finite unreachable metrics cannot win
// against a reachable path even with the largest uint8 branch costs.
inline ByteAlternatives integer_byte_alternatives(std::span<const ByteCost> costs, int initial_state = -1,
                                                  int final_state = -1) {
    if (initial_state < -1 || initial_state >= 64 || final_state < -1 || final_state >= 64)
        throw std::invalid_argument("Invalid integer posterior endpoint");
    using State = std::array<std::int32_t, 64>;
    constexpr std::int32_t unreachable = -(1 << 24);
    constexpr unsigned block = 4096;
    const auto normalize = [](State &state) {
        auto maximum = state[0];
        for (auto value : state)
            maximum = std::max(maximum, value);
        for (auto &value : state)
            value -= maximum;
    };
    const auto forward = [&](const State &old, const ByteCost &cost) {
        State next;
        for (unsigned s = 0; s < 64; ++s) {
            const auto label = fec::DvbtCode::branches[s];
            next[s] = std::max(old[s / 2] - cost[label], old[s / 2 + 32] - cost[label ^ 3]);
        }
        normalize(next);
        return next;
    };
    State alpha{};
    if (initial_state >= 0) {
        alpha.fill(unreachable);
        alpha[initial_state] = 0;
    }
    std::vector<State> checkpoints((costs.size() + block - 1) / block);
    for (std::size_t group = 0; group < checkpoints.size(); ++group) {
        checkpoints[group] = alpha;
        // The final block is reconstructed below; do not compute it twice.
        if (group + 1 < checkpoints.size())
            for (auto t = group * block; t < (group + 1) * block; ++t)
                alpha = forward(alpha, costs[t]);
    }
    ByteAlternatives out;
    out.map_bytes.resize(costs.size() / 8);
    out.second_bytes.resize(out.map_bytes.size());
    out.alternative_gap.resize(out.map_bytes.size());
    std::vector<State> scratch(std::min<std::size_t>(block, costs.size()) + 1);
    State beta{}, byte_end{};
    if (final_state >= 0) {
        beta.fill(unreachable);
        beta[final_state] = 0;
    }
    for (std::size_t group = checkpoints.size(); group-- > 0;) {
        const auto begin = group * block, end = std::min(begin + block, costs.size());
        scratch[0] = checkpoints[group];
        for (auto t = begin; t < end; ++t)
            scratch[t - begin + 1] = forward(scratch[t - begin], costs[t]);
        for (auto t = end; t-- > begin;) {
            if (t % 8 == 7)
                byte_end = beta;
            if (t % 8 == 0 && t + 8 <= costs.size()) {
                std::array<std::int32_t, 256> score;
                for (unsigned word = 0; word < 256; ++word)
                    score[word] = scratch[t - begin + 6][word >> 2] -
                                  costs[t + 6][fec::DvbtCode::symbol(word >> 1)] -
                                  costs[t + 7][fec::DvbtCode::symbol(word & 127)] + byte_end[word & 63];
                auto best_score = score[0];
                for (auto value : score)
                    best_score = std::max(best_score, value);
                const unsigned best = std::find(score.begin(), score.end(), best_score) - score.begin();
                score[best] = std::numeric_limits<std::int32_t>::min();
                auto second_score = score[0];
                for (auto value : score)
                    second_score = std::max(second_score, value);
                const unsigned second = std::find(score.begin(), score.end(), second_score) - score.begin();
                out.map_bytes[t / 8] = best;
                out.second_bytes[t / 8] = second;
                out.alternative_gap[t / 8] = best_score - second_score;
            }
            State next;
            for (unsigned s = 0; s < 64; ++s) {
                const unsigned dest = 2 * s & 63, label = fec::DvbtCode::symbol(2 * s);
                next[s] = std::max(beta[dest] - costs[t][label], beta[dest + 1] - costs[t][label ^ 3]);
            }
            normalize(next);
            beta = next;
        }
    }
    return out;
}
} // namespace radar::validation
