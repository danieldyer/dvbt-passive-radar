#pragma once
// Experimental bounded rescue after the ordinary RS-feedback fixed point.
// List uniqueness is not payload truth; validate miscorrections independently.
#include "chase_rs_reference.hpp"
#include "fec_feedback.hpp"
#include "integer_byte_alternatives.hpp"
#include "iterative_fec_reference.hpp"
#include "soft_rs_reference.hpp"

namespace radar::validation {
// Random access avoids materialising branch costs for already known bytes.
class CompactBranchCosts {
    static constexpr auto costs = []() consteval {
        std::array<ByteCost, 225> table{};
        for (unsigned a = 0; a < 15; ++a)
            for (unsigned b = 0; b < 15; ++b) {
                const auto value = [](unsigned code) {
                    return (code < 7 ? -1 : 1) * int(fec::minimax_llr_levels[code < 7 ? 7 - code : code - 7]);
                };
                const int x = value(a), y = value(b);
                for (unsigned symbol = 0; symbol < 4; ++symbol)
                    table[a * 15 + b][symbol] =
                        std::max(symbol & 1 ? x : -x, 0) + std::max(symbol & 2 ? y : -y, 0);
            }
        return table;
    }();
    std::span<const std::uint8_t> coded_;
    std::span<const unsigned> pattern_;
    std::array<unsigned, 7> offsets_{};
    unsigned period_{}, sent_{};

  public:
    CompactBranchCosts(std::span<const std::uint8_t> coded, unsigned rate)
        : coded_(coded), pattern_(fec::puncture_pattern(rate)), period_(pattern_.size() / 2),
          sent_(std::count(pattern_.begin(), pattern_.end(), 1U)) {
        if (coded.size() % sent_ || !std::ranges::all_of(coded, [](auto x) { return x < 15; }))
            throw std::invalid_argument("Invalid conditional Chase metrics");
        for (unsigned phase = 1; phase < period_; ++phase)
            offsets_[phase] = offsets_[phase - 1] + pattern_[2 * phase - 2] + pattern_[2 * phase - 1];
    }
    std::size_t steps() const {
        return coded_.size() / sent_ * period_;
    }
    ByteCost operator()(std::size_t t) const {
        const auto phase = t % period_;
        auto input = t / period_ * sent_ + offsets_[phase];
        const unsigned a = pattern_[2 * phase] ? coded_[input++] : 7;
        const unsigned b = pattern_[2 * phase + 1] ? coded_[input] : 7;
        return costs[a * 15 + b];
    }
};
struct ConditionalChaseResult {
    std::vector<fec::RsResult> words;
    unsigned seeds{}, rounds{}, feedback_iterations{};
    std::size_t posterior_steps{}, feedback_steps{};
    bool budget_exceeded = false;
};
template <bool Integer = false>
inline ConditionalChaseResult conditional_chase_feedback(
    std::span<const std::uint8_t> coded, unsigned rate, std::span<const fec::RsResult> initial,
    std::size_t first, std::size_t last, unsigned rounds = 2, unsigned positions = 4,
    unsigned maximum_failed = 256, unsigned maximum_unknown_bytes = 65536,
    fec::ConstrainedViterbi *prepared_decoder = nullptr, unsigned maximum_feedback_iterations = 16) {
    if (first > last || last > initial.size() || rounds > 2 || positions > 8 ||
        maximum_feedback_iterations > 16)
        throw std::invalid_argument("Invalid conditional Chase bounds");
    ConditionalChaseResult out;
    out.words.assign(initial.begin(), initial.end());
    const CompactBranchCosts cost_at(coded, rate);
    const auto steps = cost_at.steps();
    if ((steps / 8 / 204 > 11 ? steps / 8 / 204 - 11 : 0) != initial.size())
        throw std::invalid_argument("Conditional Chase packet coordinates mismatch");
    std::vector<std::int16_t> known(steps / 8);
    fec::ConstrainedViterbi owned_decoder;
    auto &decoder = prepared_decoder ? *prepared_decoder : owned_decoder;
    bool prepared = prepared_decoder != nullptr;
    for (unsigned round = 0; round < rounds; ++round) {
        if (out.feedback_iterations == maximum_feedback_iterations)
            break;
        unsigned failed = 0;
        for (auto p = first; p < last; ++p)
            failed += !out.words[p].valid;
        if (!failed)
            break;
        if (failed > maximum_failed) {
            out.budget_exceeded = true;
            break;
        }
        std::fill(known.begin(), known.end(), -1);
        for (std::size_t p = 0; p < out.words.size(); ++p)
            if (out.words[p].valid)
                for (unsigned j = 0; j < 204; ++j) {
                    const auto i = 2244 + p * 204 + j;
                    known[i - (11 - i % 12) * 204] = out.words[p].bytes[j];
                }
        if (std::ranges::count(known, -1) > maximum_unknown_bytes) {
            out.budget_exceeded = true;
            break;
        }
        if (!prepared) {
            decoder.prepare(coded, rate);
            prepared = true;
        }
        const auto &selected = decoder(known);
        ++out.feedback_iterations;
        out.feedback_steps += decoder.decoded_steps();
        const auto observe = [&](std::size_t t) {
            ++out.posterior_steps;
            return cost_at(t);
        };
        const auto alternatives = [&] {
            if constexpr (Integer)
                return conditional_byte_alternatives_with(steps, known, observe, integer_byte_alternatives);
            else
                return conditional_byte_alternatives_at(steps, known, observe);
        }();
        unsigned added = 0;
        for (auto p = first; p < last; ++p)
            if (!out.words[p].valid) {
                fec::DvbtRs::Word received, alternative;
                std::array<float, 204> gaps;
                for (unsigned j = 0; j < 204; ++j) {
                    const auto i = 2244 + p * 204 + j, source = i - (11 - i % 12) * 204;
                    received[j] = selected[source];
                    const bool same = alternatives.map_bytes[source] == received[j];
                    alternative[j] =
                        same ? alternatives.second_bytes[source] : alternatives.map_bytes[source];
                    gaps[j] = same ? alternatives.alternative_gap[source] : 0;
                }
                const auto candidates = chase_candidates(received, alternative, gaps, positions);
                if (candidates.size() != 1)
                    continue;
                auto &packet = out.words[p];
                packet.bytes = candidates.front();
                packet.valid = true;
                packet.corrected = 0;
                for (unsigned j = 0; j < 204; ++j)
                    packet.corrected += packet.bytes[j] != received[j];
                ++added;
            }
        ++out.rounds;
        out.seeds += added;
        if (!added || out.feedback_iterations == maximum_feedback_iterations)
            break;
        auto continued = iterative_rs_with(
            std::vector<std::uint8_t>(known.size()),
            [&](auto constraints) {
                const auto &decoded = decoder(constraints);
                ++out.feedback_iterations;
                out.feedback_steps += decoder.decoded_steps();
                return decoded;
            },
            maximum_feedback_iterations - out.feedback_iterations, out.words);
        out.words = std::move(continued.words);
    }
    return out;
}
} // namespace radar::validation
