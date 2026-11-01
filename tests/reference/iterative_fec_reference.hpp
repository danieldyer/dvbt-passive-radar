#pragma once
// Scalar state-pinning test reference for the fixed concatenated DVB-T code.
// Accepted RS words constrain their actual bytes in the convolutional trellis.
// Not a claim of joint ML decoding; an RS miscorrection can poison feedback.
#include "pinned_trellis.hpp"
#include "soft_trellis.hpp"
namespace radar::validation {
inline std::vector<std::uint8_t> constrained_viterbi(std::span<const BranchMetrics> branches,
                                                     std::span<const std::int16_t> known_bytes) {
    if (known_bytes.size() != branches.size() / 8)
        throw std::invalid_argument("Pinned byte coordinates mismatch");
    StateMetrics state{};
    std::vector<std::uint64_t> survivors(branches.size());
    for (std::size_t t = 0; t < branches.size(); ++t) {
        StateMetrics next;
        std::array<std::uint8_t, 32> decisions;
        const auto known = t / 8 < known_bytes.size() ? known_bytes[t / 8] : -1;
        const auto bit = known < 0 ? -1 : (known >> (7 - t % 8)) & 1;
        for (unsigned p = 0; p < 32; ++p) {
            const auto branch = branches[t][fec::DvbtCode::branches[2 * p]];
            const float a = state[p] + branch, b = state[p + 32] - branch;
            const float c = state[p] - branch, d = state[p + 32] + branch;
            next[2 * p] = bit == 1 ? -INFINITY : std::max(a, b);
            next[2 * p + 1] = bit == 0 ? -INFINITY : std::max(c, d);
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
struct IterativeResult {
    std::vector<fec::RsResult> words;
    std::vector<unsigned> accepted_per_pass;
};
template <class Decode>
inline IterativeResult iterative_rs_with(std::vector<std::uint8_t> bytes, Decode decode,
                                         unsigned maximum_passes, std::span<const fec::RsResult> seed = {}) {
    if (!maximum_passes)
        throw std::invalid_argument("No decoding pass");
    std::vector<std::int16_t> known(bytes.size(), -1);
    IterativeResult out;
    out.words.resize(bytes.size() / 204 > 11 ? bytes.size() / 204 - 11 : 0);
    if (!seed.empty()) {
        if (seed.size() != out.words.size())
            throw std::invalid_argument("Seed packet coordinates mismatch");
        out.words.assign(seed.begin(), seed.end());
        for (std::size_t p = 0; p < seed.size(); ++p)
            if (seed[p].valid)
                for (unsigned j = 0; j < 204; ++j) {
                    const auto i = 2244 + p * 204 + j;
                    known[i - (11 - i % 12) * 204] = seed[p].bytes[j];
                }
        bytes = decode(known);
    }
    for (unsigned pass = 0; pass < maximum_passes; ++pass) {
        unsigned accepted = 0, added = 0;
        for (std::size_t p = 0; p < out.words.size(); ++p) {
            auto &packet = out.words[p];
            if (!packet.valid) {
                for (unsigned j = 0; j < 204; ++j) {
                    const auto i = 2244 + p * 204 + j;
                    packet.bytes[j] = bytes[i - (11 - i % 12) * 204];
                }
                const int errors = fec::DvbtRs::decode(packet.bytes);
                packet.valid = errors >= 0 && (packet.bytes[0] == 0x47 || packet.bytes[0] == 0xb8);
                if (packet.valid) {
                    packet.corrected = errors;
                    ++added;
                    for (unsigned j = 0; j < 204; ++j) {
                        const auto i = 2244 + p * 204 + j;
                        known[i - (11 - i % 12) * 204] = packet.bytes[j];
                    }
                }
            }
            accepted += packet.valid;
        }
        out.accepted_per_pass.push_back(accepted);
        if (!added || accepted == out.words.size() || pass + 1 == maximum_passes)
            break;
        bytes = decode(known);
    }
    return out;
}
inline IterativeResult iterative_rs(std::span<const BranchMetrics> branches, unsigned maximum_passes,
                                    std::span<const fec::RsResult> seed = {}) {
    return iterative_rs_with(
        soft_viterbi(branches), [&](auto known) { return constrained_viterbi(branches, known); },
        maximum_passes, seed);
}
template <unsigned Maximum = 7>
inline IterativeResult iterative_rs_quantized(std::span<const ByteCost> costs, unsigned maximum_passes) {
    auto first = [&] {
        if constexpr (Maximum == 7)
            return byte_viterbi(costs);
        else
            return pinned_viterbi<Maximum>(costs, std::vector<std::int16_t>(costs.size() / 8, -1));
    }();
    return iterative_rs_with(
        std::move(first), [&](auto known) { return pinned_viterbi<Maximum>(costs, known); }, maximum_passes);
}
} // namespace radar::validation
