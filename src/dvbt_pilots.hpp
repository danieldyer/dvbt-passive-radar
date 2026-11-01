// DVB-T pilot definitions from ETSI EN 300 744 / GNU Radio gr-dtv.
// Copyright 2015 Free Software Foundation, Inc.; SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <span>
namespace radar::dvbt {
inline constexpr std::array<unsigned, 45> continual_2k{
    0,   48,   54,   87,   141,  156,  192,  201,  255,  279,  282,  333,  432,  450,  483,
    525, 531,  618,  636,  714,  759,  765,  780,  804,  873,  888,  918,  939,  942,  969,
    984, 1050, 1101, 1107, 1110, 1137, 1140, 1146, 1206, 1269, 1323, 1377, 1491, 1683, 1704};
inline constexpr std::array<unsigned, 17> tps_2k{34,  50,   209,  346,  413,  569,  595,  688, 790,
                                                 901, 1073, 1219, 1262, 1286, 1469, 1594, 1687};
inline constexpr auto continual = []() consteval {
    std::array<std::uint32_t, 177> out{};
    unsigned at = 0;
    for (unsigned block = 0; block < 4; ++block)
        for (auto k : continual_2k)
            if (!block || k)
                out[at++] = 1704 * block + k;
    if (at != out.size())
        throw "Invalid continual pilot count";
    return out;
}();
inline constexpr auto tps = []() consteval {
    std::array<std::uint32_t, 68> out{};
    unsigned at = 0;
    for (unsigned block = 0; block < 4; ++block)
        for (auto k : tps_2k)
            out[at++] = 1704 * block + k;
    return out;
}();
inline constexpr auto polarity = []() consteval {
    std::array<std::uint8_t, 6817> out{};
    unsigned state = 2047;
    for (auto &value : out) {
        value = state & 1;
        state = (state >> 1) | (((state ^ (state >> 2)) & 1) << 10);
    }
    return out;
}();
// Shortened BCH(67,53); the 60 omitted leading zeros leave the LFSR at zero.
constexpr std::uint16_t tps_parity(std::span<const std::uint8_t, 53> data) {
    unsigned state = 0;
    for (auto bit : data)
        state = (state >> 1) ^ (((bit ^ state) & 1) * 0x3bb0U);
    return std::uint16_t(state);
}
} // namespace radar::dvbt
