#pragma once
#include "fec.hpp"
#include "soft_quantizer.hpp"
namespace radar::fec {
// One immutable codebook for initial ML, constrained ML and byte rescue.
// A complete observation is the pair (a,b), including puncture erasure code 7.
inline constexpr auto trellis_levels = []() consteval {
    std::array<int, 15> out{};
    for (int code = 0; code < 15; ++code) {
        const int rank = code - 7;
        out[code] = (rank < 0 ? -1 : 1) * minimax_llr_levels[rank < 0 ? -rank : rank];
    }
    return out;
}();
alignas(64) inline constexpr auto trellis_costs = []() consteval {
    std::array<std::array<std::uint8_t, 32>, 225> out{};
    for (unsigned a = 0; a < 15; ++a)
        for (unsigned b = 0; b < 15; ++b)
            for (unsigned p = 0; p < 32; ++p) {
                const auto label = DvbtCode::branches[2 * p];
                const int x = trellis_levels[a], y = trellis_levels[b];
                out[a * 15 + b][p] = std::max(label & 1 ? x : -x, 0) + std::max(label & 2 ? y : -y, 0);
            }
    return out;
}();
inline constexpr auto trellis_weights = []() consteval {
    std::array<std::uint8_t, 225> out{};
    for (unsigned a = 0; a < 15; ++a)
        for (unsigned b = 0; b < 15; ++b) {
            const int x = trellis_levels[a], y = trellis_levels[b];
            out[a * 15 + b] = (x < 0 ? -x : x) + (y < 0 ? -y : y);
        }
    return out;
}();
} // namespace radar::fec
