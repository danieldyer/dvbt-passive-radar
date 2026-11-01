#pragma once
// Exact checkpointed convolutional posteriors and bounded GMD RS reference.
#include "rs_erasures_reference.hpp"
#include "soft_trellis.hpp"
#include <numeric>
namespace radar::validation {
struct ByteAlternatives {
    std::vector<std::uint8_t> map_bytes, second_bytes;
    std::vector<float> alternative_gap;
};
struct SoftInformation : ByteAlternatives {
    std::vector<float> bits, byte_log_probability, map_log_probability;
};
// Six input bits overwrite the K=7 state. Alpha after those six bits already
// sums every pre-byte history conditional on this prefix. Only two transitions
// remain, so all 256 joint byte probabilities require no path enumeration.
inline std::array<float, 256> byte_scores(const StateMetrics &prefix, const BranchMetrics &penultimate,
                                          const BranchMetrics &last, const StateMetrics &suffix) {
    std::array<float, 256> score;
    for (unsigned word = 0; word < 256; ++word) {
        const auto seventh = word >> 1;
        score[word] = prefix[word >> 2] + penultimate[fec::DvbtCode::symbol(seventh)] +
                      last[fec::DvbtCode::symbol(word & 127)] + suffix[word & 63];
    }
    return score;
}
inline std::array<float, 256> byte_posterior(const StateMetrics &prefix, const BranchMetrics &penultimate,
                                             const BranchMetrics &last, const StateMetrics &suffix) {
    auto score = byte_scores(prefix, penultimate, last, suffix);
    const auto maximum = *std::max_element(score.begin(), score.end());
    double sum = 0;
    for (auto x : score)
        sum += std::exp(double(x - maximum));
    const auto normalization = float(std::log(sum));
    for (auto &x : score)
        x = (x - maximum) - normalization;
    return score;
}
inline SoftInformation bcjr_information(std::span<const BranchMetrics> branches,
                                        std::span<const std::uint8_t> selected, int initial_state = -1,
                                        int final_state = -1) {
    if (selected.size() != branches.size() / 8)
        throw std::invalid_argument("Byte posterior coordinates mismatch");
    if (initial_state < -1 || initial_state >= 64 || final_state < -1 || final_state >= 64)
        throw std::invalid_argument("Invalid posterior endpoint state");
    constexpr unsigned block = 4096;
    std::vector<StateMetrics> checkpoints((branches.size() + block - 1) / block);
    StateMetrics alpha{};
    if (initial_state >= 0) {
        alpha.fill(-INFINITY);
        alpha[initial_state] = 0;
    }
    for (std::size_t t = 0; t < branches.size(); ++t) {
        if (t % block == 0)
            checkpoints[t / block] = alpha;
        alpha = forward_map(alpha, branches[t]);
    }
    SoftInformation out;
    out.bits.resize(branches.size());
    out.byte_log_probability.resize(selected.size());
    out.map_bytes.resize(selected.size());
    out.map_log_probability.resize(selected.size());
    out.second_bytes.resize(selected.size());
    out.alternative_gap.resize(selected.size());
    std::vector<StateMetrics> scratch(std::min<std::size_t>(block, branches.size()) + 1);
    StateMetrics beta{}, byte_end_beta{};
    if (final_state >= 0) {
        beta.fill(-INFINITY);
        beta[final_state] = 0;
    }
    for (std::size_t group = checkpoints.size(); group-- > 0;) {
        const auto begin = group * block, end = std::min(begin + block, branches.size());
        scratch[0] = checkpoints[group];
        for (auto t = begin; t < end; ++t) {
            const auto local = t - begin;
            const auto &old = scratch[local];
            auto &next = scratch[local + 1];
            for (unsigned s = 0; s < 64; ++s) {
                const auto label = fec::DvbtCode::branches[s];
                next[s] = log_add(old[s / 2] + branches[t][label], old[s / 2 + 32] + branches[t][label ^ 3]);
            }
            const auto maximum = *std::max_element(next.begin(), next.end());
            for (auto &x : next)
                x -= maximum;
        }
        for (auto t = end; t-- > begin;) {
            if (t % 8 == 7)
                byte_end_beta = beta;
            StateMetrics next;
            float zero = -INFINITY, one = -INFINITY;
            for (unsigned s = 0; s < 64; ++s) {
                const auto d = 2 * s & 63;
                const auto label = fec::DvbtCode::symbol(2 * s);
                const float a = branches[t][label] + beta[d], b = branches[t][label ^ 3] + beta[d + 1];
                next[s] = log_add(a, b);
                zero = log_add(zero, scratch[t - begin][s] + a);
                one = log_add(one, scratch[t - begin][s] + b);
            }
            out.bits[t] = zero - one;
            if (t % 8 == 0 && t + 8 <= branches.size()) {
                const auto probability =
                    byte_posterior(scratch[t - begin + 6], branches[t + 6], branches[t + 7], byte_end_beta);
                const auto best = std::max_element(probability.begin(), probability.end());
                out.byte_log_probability[t / 8] = probability[selected[t / 8]];
                out.map_bytes[t / 8] = std::uint8_t(best - probability.begin());
                out.map_log_probability[t / 8] = *best;
                unsigned second = best == probability.begin() ? 1 : 0;
                for (unsigned value = 0; value < 256; ++value)
                    if (probability.begin() + value != best && probability[value] > probability[second])
                        second = value;
                out.second_bytes[t / 8] = std::uint8_t(second);
                out.alternative_gap[t / 8] = *best - probability[second];
            }
            normalize(next);
            beta = next;
        }
    }
    return out;
}
// Max-log alternatives are conditional best-path scores, not calibrated
// byte probabilities. Keep that distinction explicit in the return type.
inline ByteAlternatives max_log_byte_alternatives(std::span<const BranchMetrics> branches,
                                                  int initial_state = -1, int final_state = -1) {
    if (initial_state < -1 || initial_state >= 64 || final_state < -1 || final_state >= 64)
        throw std::invalid_argument("Invalid max-log endpoint state");
    constexpr unsigned block = 4096;
    auto forward = [](const StateMetrics &old, const BranchMetrics &branch) {
        StateMetrics next;
        for (unsigned s = 0; s < 64; ++s) {
            const auto label = fec::DvbtCode::branches[s];
            next[s] = std::max(old[s / 2] + branch[label], old[s / 2 + 32] + branch[label ^ 3]);
        }
        normalize(next);
        return next;
    };
    std::vector<StateMetrics> checkpoints((branches.size() + block - 1) / block);
    StateMetrics alpha{};
    if (initial_state >= 0) {
        alpha.fill(-INFINITY);
        alpha[initial_state] = 0;
    }
    for (std::size_t t = 0; t < branches.size(); ++t) {
        if (t % block == 0)
            checkpoints[t / block] = alpha;
        alpha = forward(alpha, branches[t]);
    }
    ByteAlternatives out;
    out.map_bytes.resize(branches.size() / 8);
    out.second_bytes.resize(out.map_bytes.size());
    out.alternative_gap.resize(out.map_bytes.size());
    std::vector<StateMetrics> scratch(std::min<std::size_t>(block, branches.size()) + 1);
    StateMetrics beta{}, byte_end{};
    if (final_state >= 0) {
        beta.fill(-INFINITY);
        beta[final_state] = 0;
    }
    for (std::size_t group = checkpoints.size(); group-- > 0;) {
        const auto begin = group * block, end = std::min(begin + block, branches.size());
        scratch[0] = checkpoints[group];
        for (auto t = begin; t < end; ++t)
            scratch[t - begin + 1] = forward(scratch[t - begin], branches[t]);
        for (auto t = end; t-- > begin;) {
            if (t % 8 == 7)
                byte_end = beta;
            if (t % 8 == 0 && t + 8 <= branches.size()) {
                const auto score =
                    byte_scores(scratch[t - begin + 6], branches[t + 6], branches[t + 7], byte_end);
                unsigned best = 0, second = 1;
                if (score[second] > score[best])
                    std::swap(best, second);
                for (unsigned value = 2; value < 256; ++value) {
                    if (score[value] > score[best]) {
                        second = best;
                        best = value;
                    } else if (score[value] > score[second])
                        second = value;
                }
                out.map_bytes[t / 8] = best;
                out.second_bytes[t / 8] = second;
                out.alternative_gap[t / 8] = score[best] - score[second];
            }
            StateMetrics next;
            for (unsigned s = 0; s < 64; ++s) {
                const unsigned dest = 2 * s & 63, label = fec::DvbtCode::symbol(2 * s);
                next[s] = std::max(branches[t][label] + beta[dest], branches[t][label ^ 3] + beta[dest + 1]);
            }
            normalize(next);
            beta = next;
        }
    }
    return out;
}
// Exact conditional max-log alternatives after accepted RS words have fixed
// encoder bytes. A known left byte fixes the starting K=7 state; the first six
// bits of a known right byte fix the ending state. Its last two bits have a
// common cost and cannot affect any unknown-byte ranking. This reference only
// visits unknown intervals and retains the original punctured likelihoods.
template <class CostAt, class Decode>
inline ByteAlternatives conditional_byte_alternatives_with(std::size_t steps,
                                                           std::span<const std::int16_t> known,
                                                           CostAt cost_at, Decode decode) {
    if (known.size() != steps / 8 || !std::ranges::all_of(known, [](auto x) { return x >= -1 && x <= 255; }))
        throw std::invalid_argument("Invalid conditional byte coordinates");
    ByteAlternatives out;
    out.map_bytes.resize(known.size());
    out.second_bytes.resize(known.size());
    out.alternative_gap.assign(known.size(), 1e6F);
    for (std::size_t at = 0; at < known.size();) {
        if (known[at] >= 0) {
            out.map_bytes[at] = out.second_bytes[at] = known[at];
            ++at;
            continue;
        }
        const auto begin = at;
        while (at < known.size() && known[at] < 0)
            ++at;
        const auto end = at < known.size() ? 8 * at + 6 : steps;
        std::vector<ByteCost> costs(end - 8 * begin);
        for (std::size_t t = 0; t < costs.size(); ++t)
            costs[t] = cost_at(8 * begin + t);
        const auto part =
            decode(costs, begin ? known[begin - 1] & 63 : -1, at < known.size() ? known[at] >> 2 : -1);
        std::copy(part.map_bytes.begin(), part.map_bytes.end(), out.map_bytes.begin() + begin);
        std::copy(part.second_bytes.begin(), part.second_bytes.end(), out.second_bytes.begin() + begin);
        std::copy(part.alternative_gap.begin(), part.alternative_gap.end(),
                  out.alternative_gap.begin() + begin);
    }
    return out;
}
template <class CostAt>
inline ByteAlternatives
conditional_byte_alternatives_at(std::size_t steps, std::span<const std::int16_t> known, CostAt cost_at) {
    return conditional_byte_alternatives_with(steps, known, cost_at,
                                              [](std::span<const ByteCost> costs, int initial, int final) {
                                                  std::vector<BranchMetrics> branch(costs.size());
                                                  for (std::size_t t = 0; t < costs.size(); ++t)
                                                      for (unsigned k = 0; k < 4; ++k)
                                                          branch[t][k] = -float(costs[t][k]);
                                                  return max_log_byte_alternatives(branch, initial, final);
                                              });
}
inline ByteAlternatives conditional_byte_alternatives(std::span<const ByteCost> costs,
                                                      std::span<const std::int16_t> known) {
    return conditional_byte_alternatives_at(costs.size(), known, [&](std::size_t t) { return costs[t]; });
}
// Sum-product counterpart of the interval reference above. Quantised costs
// are in half-LLR units, so multiplying by -0.5 gives log likelihoods. This is
// a diagnostic reference, not an unbounded addition to the capture decoder.
inline SoftInformation conditional_information(std::span<const ByteCost> costs,
                                               std::span<const std::int16_t> known,
                                               std::span<const std::uint8_t> selected) {
    if (known.size() != costs.size() / 8 || selected.size() != known.size() ||
        !std::ranges::all_of(known, [](auto x) { return x >= -1 && x <= 255; }))
        throw std::invalid_argument("Invalid conditional posterior coordinates");
    SoftInformation out;
    out.bits.resize(8 * known.size());
    out.byte_log_probability.resize(known.size());
    out.map_log_probability.resize(known.size());
    out.map_bytes.resize(known.size());
    out.second_bytes.resize(known.size());
    out.alternative_gap.assign(known.size(), INFINITY);
    for (std::size_t at = 0; at < known.size();) {
        if (known[at] >= 0) {
            out.map_bytes[at] = out.second_bytes[at] = known[at];
            out.byte_log_probability[at] = selected[at] == known[at] ? 0 : -INFINITY;
            for (unsigned bit = 0; bit < 8; ++bit)
                out.bits[8 * at + bit] = (known[at] >> (7 - bit)) & 1 ? -INFINITY : INFINITY;
            ++at;
            continue;
        }
        const auto begin = at;
        while (at < known.size() && known[at] < 0)
            ++at;
        const auto end = at < known.size() ? 8 * at + 6 : costs.size();
        std::vector<BranchMetrics> branch(end - 8 * begin);
        for (std::size_t t = 0; t < branch.size(); ++t)
            for (unsigned k = 0; k < 4; ++k)
                branch[t][k] = -.5F * costs[8 * begin + t][k];
        const auto part =
            bcjr_information(branch, selected.subspan(begin, at - begin), begin ? known[begin - 1] & 63 : -1,
                             at < known.size() ? known[at] >> 2 : -1);
        const auto copy = [&](const auto &source, auto &destination) {
            std::copy(source.begin(), source.end(), destination.begin() + begin);
        };
        copy(part.byte_log_probability, out.byte_log_probability);
        copy(part.map_log_probability, out.map_log_probability);
        copy(part.map_bytes, out.map_bytes);
        copy(part.second_bytes, out.second_bytes);
        copy(part.alternative_gap, out.alternative_gap);
        std::copy_n(part.bits.begin(), 8 * (at - begin), out.bits.begin() + 8 * begin);
    }
    return out;
}
inline fec::RsResult gmd_rs(const fec::DvbtRs::Word &received,
                            const std::array<float, 204> &byte_log_probability,
                            const std::array<float, 204 * 8> &bit_llr) {
    const auto cost = [&](const fec::DvbtRs::Word &word) {
        double score = 0;
        for (unsigned i = 0; i < 204 * 8; ++i)
            if ((word[i / 8] >> (7 - i % 8)) & 1)
                score += bit_llr[i];
        return score;
    };
    fec::RsResult best;
    double best_cost = INFINITY;
    auto candidate = received;
    int changed = fec::DvbtRs::decode(candidate);
    if (changed >= 0 && (candidate[0] == 0x47 || candidate[0] == 0xb8)) {
        best = {candidate, true, unsigned(changed)};
        best_cost = cost(candidate);
    }
    if (changed == 0 && best.valid)
        return best;
    std::array<unsigned, 204> order;
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto a, auto b) { return byte_log_probability[a] < byte_log_probability[b]; });
    std::array<float, 204> reliability;
    for (unsigned i = 0; i < 204; ++i)
        reliability[i] = std::max(0.F, 2 * std::exp(byte_log_probability[i]) - 1);
    for (unsigned erased = 2; erased <= 16; erased += 2) {
        candidate = received;
        changed = rs_erasures(candidate, {order.data(), erased});
        if (changed < 0 || (candidate[0] != 0x47 && candidate[0] != 0xb8))
            continue;
        double distance = 0;
        for (unsigned i = 0; i < 204; ++i)
            distance += 1 - reliability[i] + 2 * reliability[i] * (candidate[i] != received[i]);
        if (distance >= 17)
            continue; // Reject valid words manufactured by 16 erasures.
        const double score = cost(candidate);
        if (score < best_cost) {
            best = {candidate, true, unsigned(changed)};
            best_cost = score;
        }
    }
    return best;
}
inline std::vector<fec::RsResult> soft_outer(std::span<const std::uint8_t> selected,
                                             const SoftInformation &information, bool byte_map = false,
                                             std::span<const fec::RsResult> seed = {}) {
    if (information.byte_log_probability.size() != selected.size() ||
        information.bits.size() / 8 != selected.size())
        throw std::invalid_argument("Outer soft coordinate mismatch");
    std::vector<fec::RsResult> packets(selected.size() / 204 > 11 ? selected.size() / 204 - 11 : 0);
    if (!seed.empty()) {
        if (seed.size() != packets.size())
            throw std::invalid_argument("Soft outer seed coordinates mismatch");
        std::copy(seed.begin(), seed.end(), packets.begin());
    }
    for (std::size_t p = 0; p < packets.size(); ++p) {
        if (packets[p].valid)
            continue;
        fec::DvbtRs::Word word;
        std::array<float, 204> logp;
        std::array<float, 204 * 8> bits;
        for (unsigned j = 0; j < 204; ++j) {
            const auto i = 2244 + p * 204 + j, source = i - (11 - i % 12) * 204;
            word[j] = byte_map ? information.map_bytes[source] : selected[source];
            logp[j] =
                byte_map ? information.map_log_probability[source] : information.byte_log_probability[source];
            std::copy_n(information.bits.data() + 8 * source, 8, bits.data() + 8 * j);
        }
        packets[p] = gmd_rs(word, logp, bits);
    }
    return packets;
}
} // namespace radar::validation
