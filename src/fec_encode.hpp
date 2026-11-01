#pragma once
#include "fec.hpp"
#include <numeric>
namespace radar::fec {
struct Encoded {
    std::vector<std::uint8_t> bits, valid;
};
inline Encoded convolutional_encode(std::span<const std::uint8_t> bytes, std::span<const std::uint8_t> valid,
                                    int rate) {
    const auto pattern = fec::puncture_pattern(rate);
    // The convolutional code is linear: XOR the previous-state contribution
    // with the current byte contribution. Two small tables replace per-bit parity.
    struct Tables {
        std::array<std::uint16_t, 64> state{};
        std::array<std::uint16_t, 256> byte{};
        static constexpr std::uint16_t encode(unsigned reg, unsigned value) {
            std::uint16_t word = 0;
            for (unsigned bit = 0; bit < 8; ++bit) {
                reg = ((reg << 1) | ((value >> (7 - bit)) & 1U)) & 127U;
                word |= static_cast<std::uint16_t>((fec::DvbtCode::symbol(reg) & 1U) << (2 * bit));
                word |= static_cast<std::uint16_t>(((fec::DvbtCode::symbol(reg) >> 1) & 1U) << (2 * bit + 1));
            }
            return word;
        }
        constexpr Tables() {
            for (unsigned i = 0; i < state.size(); ++i)
                state[i] = encode(i, 0);
            for (unsigned i = 0; i < byte.size(); ++i)
                byte[i] = encode(0, i);
        }
    };
    static constexpr Tables table;
    const auto phases = pattern.size() / std::gcd(pattern.size(), std::size_t{16});
    std::array<std::array<unsigned, 16>, 7> positions{};
    std::array<unsigned, 7> counts{};
    for (std::size_t phase = 0; phase < phases; ++phase)
        for (unsigned bit = 0; bit < 16; ++bit)
            if (pattern[(phase * 16 + bit) % pattern.size()])
                positions[phase][counts[phase]++] = bit;
    const auto total = bytes.size() / phases * std::accumulate(counts.begin(), counts.end(), std::size_t{}) +
                       std::accumulate(counts.begin(), counts.begin() + bytes.size() % phases, std::size_t{});
    Encoded result;
    result.bits.resize(total);
    result.valid.resize(total);
    std::size_t at = 0, phase = 0;
    unsigned state = 0;
    bool previous_good = false;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        const auto word = table.state[state] ^ table.byte[bytes[i]];
        const bool good = valid.empty() || valid[i];
        for (unsigned j = 0; j < counts[phase]; ++j) {
            const auto bit = positions[phase][j];
            result.bits[at] = static_cast<std::uint8_t>((word >> bit) & 1U);
            // After seven input bits the previous byte can no longer affect validity.
            result.valid[at++] = good && (previous_good || bit >= 12);
        }
        state = bytes[i] & 63U;
        previous_good = good;
        if (++phase == phases)
            phase = 0;
    }
    return result;
}
} // namespace radar::fec
