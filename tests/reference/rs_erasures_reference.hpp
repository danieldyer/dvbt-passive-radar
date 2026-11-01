#pragma once
// Bounded errors-and-erasures reference for RS(204,188), first root alpha^0.
#include "fec.hpp"
namespace radar::validation {
inline int rs_erasures(fec::DvbtRs::Word &word, std::span<const unsigned> erasures) {
    using F = fec::Gf256<>;
    if (erasures.size() > 16)
        return -1;
    std::array<bool, 204> seen{};
    for (auto p : erasures) {
        if (p >= 204 || seen[p])
            throw std::invalid_argument("Invalid erasure position");
        seen[p] = true;
    }
    if (fec::DvbtRs::valid(word))
        return 0;
    std::array<std::uint8_t, 16> syndrome{};
    for (unsigned j = 0; j < 16; ++j)
        for (auto b : word)
            syndrome[j] = F::mul(syndrome[j], F::alpha(j)) ^ b;
    auto reduced = syndrome;
    unsigned count = 16;
    std::array<std::uint8_t, 17> erased{};
    erased[0] = 1;
    unsigned erased_degree = 0;
    for (auto position : erasures) {
        const auto x = F::alpha(203 - position);
        for (unsigned i = 0; i + 1 < count; ++i)
            reduced[i] = F::mul(reduced[i], x) ^ reduced[i + 1];
        --count;
        for (unsigned i = ++erased_degree; i > 0; --i)
            erased[i] ^= F::mul(erased[i - 1], x);
    }
    std::array<std::uint8_t, 17> locator{}, previous_locator{};
    locator[0] = previous_locator[0] = 1;
    unsigned degree = 0, shift = 1;
    std::uint8_t previous = 1;
    for (unsigned n = 0; n < count; ++n) {
        auto delta = reduced[n];
        for (unsigned j = 1; j <= degree; ++j)
            delta ^= F::mul(locator[j], reduced[n - j]);
        if (!delta) {
            ++shift;
            continue;
        }
        const auto old = locator;
        const auto scale = F::div(delta, previous);
        for (unsigned j = 0; j + shift <= 16; ++j)
            locator[j + shift] ^= F::mul(scale, previous_locator[j]);
        if (2 * degree <= n) {
            degree = n + 1 - degree;
            previous_locator = old;
            previous = delta;
            shift = 1;
        } else
            ++shift;
    }
    if (2 * degree + erasures.size() > 16)
        return -1;
    std::array<std::uint8_t, 17> full{};
    for (unsigned i = 0; i <= degree; ++i)
        for (unsigned j = 0; j <= erased_degree; ++j)
            full[i + j] ^= F::mul(locator[i], erased[j]);
    degree += erased_degree;
    if (!degree)
        return -1;
    std::array<std::uint8_t, 16> omega{};
    for (unsigned i = 0; i < 16; ++i)
        for (unsigned j = 0; j <= degree && j <= i; ++j)
            omega[i] ^= F::mul(syndrome[i - j], full[j]);
    auto corrected = word;
    unsigned roots = 0, changed = 0;
    for (unsigned position = 0; position < 204; ++position) {
        const auto x = F::alpha(203 - position), z = F::div(1, x);
        auto value = full[degree];
        for (unsigned j = degree; j-- > 0;)
            value = F::mul(value, z) ^ full[j];
        if (value)
            continue;
        std::uint8_t derivative = 0, power = 1, numerator = 0;
        for (unsigned j = 1; j <= degree; ++j) {
            if (j & 1)
                derivative ^= F::mul(full[j], power);
            power = F::mul(power, z);
        }
        for (unsigned j = 16; j-- > 0;)
            numerator = F::mul(numerator, z) ^ omega[j];
        if (!derivative || ++roots > degree)
            return -1;
        const auto error = F::div(F::mul(x, numerator), derivative);
        corrected[position] ^= error;
        changed += error != 0;
    }
    if (roots != degree || !fec::DvbtRs::valid(corrected))
        return -1;
    word = corrected;
    return int(changed);
}
} // namespace radar::validation
