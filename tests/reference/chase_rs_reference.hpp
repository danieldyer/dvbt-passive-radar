#pragma once
// Bounded symbol-Chase candidate inventory. Does not use payload truth or
// choose/accept a decoder seed; ranking/feedback require separate validation.
#include "fec.hpp"
#include <cmath>
#include <numeric>
namespace radar::validation {
inline std::vector<fec::DvbtRs::Word> chase_candidates(const fec::DvbtRs::Word &received,
                                                       const fec::DvbtRs::Word &alternative,
                                                       const std::array<float, 204> &gap,
                                                       unsigned positions) {
    if (positions > 12 || !std::ranges::all_of(gap, [](float x) { return std::isfinite(x) && x >= 0; }))
        throw std::invalid_argument("Invalid bounded Chase reliability");
    std::array<unsigned, 204> order;
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(order.begin(), order.begin() + positions, order.end(),
                      [&](auto a, auto b) { return gap[a] != gap[b] ? gap[a] < gap[b] : a < b; });
    std::vector<fec::DvbtRs::Word> result;
    for (unsigned mask = 0; mask < (1U << positions); ++mask) {
        auto candidate = received;
        for (unsigned bit = 0; bit < positions; ++bit)
            if (mask & (1U << bit))
                candidate[order[bit]] = alternative[order[bit]];
        if (fec::DvbtRs::decode(candidate) < 0 || (candidate[0] != 0x47 && candidate[0] != 0xb8))
            continue;
        if (std::ranges::find(result, candidate) == result.end())
            result.push_back(candidate);
    }
    return result;
}
} // namespace radar::validation
