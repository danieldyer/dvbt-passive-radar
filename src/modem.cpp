// DVB-T mappings follow ETSI EN 300 744 and GNU Radio gr-dtv (GPL-3.0-or-later).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "modem.hpp"
#include "dvbt_pilots.hpp"
#include "fec_encode.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace radar {
unsigned qam_bits(const Tps &t) {
    if (t.mode != 1 || t.hierarchy != 0 || t.constellation < 0 || t.constellation > 2 || t.rate < 0 ||
        t.rate > 4)
        throw std::runtime_error("Reconstruction currently supports non-hierarchical 8K DVB-T only");
    return 2 + 2 * t.constellation;
}
namespace {
constexpr int bins = 6817, payload = 6048;
constexpr std::array<int, 6> offsets{0, 63, 105, 42, 21, 84};
std::vector<C> constellation(int m) {
    const int size = 1 << m, axis = m / 2, steps = (1 << axis) / 2 - 1;
    const float norm = 1 / std::sqrt(m == 2 ? 2.0F : m == 4 ? 10.0F : 42.0F);
    std::vector<C> points(size);
    for (int i = 0; i < size; ++i) {
        const int q = (i >> (m - 2)) & 3;
        int x = (i >> (axis - 1)) & ((1 << (axis - 1)) - 1), y = i & ((1 << (axis - 1)) - 1);
        const int xv = 1 + 2 * (steps - x), yv = 1 + 2 * (steps - y);
        const int v = (((x >> 1) ^ x) << (axis - 1)) + ((y >> 1) ^ y);
        x = 0;
        y = 0;
        for (int j = 0; j < axis - 1; ++j) {
            x |= ((v >> (1 + 2 * j)) & 1) << j;
            y |= ((v >> (2 * j)) & 1) << j;
        }
        const int label = (q << (m - 2)) + (x << (axis - 1)) + y;
        points[label] =
            norm * C{static_cast<float>((q >> 1) ? -xv : xv), static_cast<float>((q & 1) ? -yv : yv)};
    }
    return points;
}
consteval std::array<int, 6048> symbol_permutation() {
    constexpr int p[]{7, 1, 4, 2, 9, 6, 8, 10, 0, 3, 11, 5};
    std::array<int, 6048> h{};
    unsigned at = 0;
    int reg = 0;
    for (int i = 0; i < 8192; ++i) {
        if (i == 2)
            reg = 1;
        else if (i > 2)
            reg = (reg >> 1) | (((reg ^ (reg >> 1) ^ (reg >> 4) ^ (reg >> 6)) & 1) << 11);
        int r = (i & 1) << 12;
        for (int k = 0; k < 12; ++k)
            r |= ((reg >> k) & 1) << p[k];
        if (r < payload)
            h[at++] = r;
    }
    if (at != payload)
        throw "DVB-T permutation generation failed";
    return h;
}
inline constexpr auto permutation = symbol_permutation();
inline constexpr auto inverse_permutation = []() consteval {
    std::array<unsigned, 6048> out{};
    for (unsigned i = 0; i < out.size(); ++i)
        out[permutation[i]] = i;
    return out;
}();
} // namespace

std::vector<C> qam_constellation(int bits) {
    if (bits != 2 && bits != 4 && bits != 6)
        throw std::invalid_argument("Invalid QAM bit count");
    return constellation(bits);
}

Modem::Modem(int guard) {
    if (guard != 4 && guard != 8 && guard != 16 && guard != 32)
        throw std::invalid_argument("Invalid DVB-T guard interval");
}
PilotPattern Modem::pilot_pattern() const {
    PilotPattern result;
    for (unsigned k = 0; k < 6817; ++k)
        result.values[k] = dvbt::polarity[k] ? -4.0F / 3 : 4.0F / 3;
    result.continual.assign(dvbt::continual.begin(), dvbt::continual.end());
    result.tps.assign(dvbt::tps.begin(), dvbt::tps.end());
    return result;
}
std::vector<int> Modem::data_carriers(int modulo4) const {
    std::array<bool, bins> excluded{};
    if (modulo4 < 0 || modulo4 > 3)
        throw std::invalid_argument("Invalid DVB-T symbol phase");
    for (auto k : dvbt::continual)
        excluded[k] = true;
    for (auto k : dvbt::tps)
        excluded[k] = true;
    for (int i = 3 * modulo4; i < bins; i += 12)
        excluded[i] = true;
    std::vector<int> result;
    for (int i = 0; i < bins; ++i)
        if (!excluded[i])
            result.push_back(i);
    if (result.size() != payload)
        throw std::runtime_error("Invalid pilot partition");
    return result;
}
bool Modem::tps_bits(std::span<const std::uint8_t> decoded, Tps &result) const {
    if (decoded.size() != 68 || std::any_of(decoded.begin(), decoded.end(), [](auto v) { return v > 1; }))
        return false;
    std::copy(decoded.begin(), decoded.end(), result.bits.begin());
    auto bits = [&](int begin, int end) {
        int value = 0;
        for (int i = begin; i <= end; ++i)
            value = value * 2 + result.bits[i];
        return value;
    };
    const auto sync = bits(1, 16);
    if (sync != 0x35ee && sync != 0xca11)
        return false;
    const auto parity = dvbt::tps_parity(std::span(result.bits).subspan<1, 53>());
    for (unsigned bit = 0; bit < 14; ++bit)
        if (result.bits[54 + bit] != ((parity >> bit) & 1))
            return false;
    result.frame = bits(23, 24);
    result.constellation = bits(25, 26);
    result.hierarchy = bits(27, 29);
    result.rate = bits(30, 32);
    result.guard = bits(36, 37);
    result.mode = bits(38, 39);
    return (result.frame % 2 ? 0xca11 : 0x35ee) == sync;
}

std::vector<std::uint32_t> Modem::payload_bit_indices(const Tps &mode) const {
    const unsigned m = qam_bits(mode);
    std::vector<std::uint32_t> result(4 * payload * m);
    for (unsigned mod = 0; mod < 4; ++mod) {
        const auto carriers = data_carriers(mod);

        for (unsigned i = 0; i < payload; ++i)
            for (unsigned bit = 0; bit < m; ++bit) {
                const unsigned e = bit / (m / 2) + 2 * (bit % (m / 2));
                const unsigned w = i / 126 * 126 + (i % 126 + 126 - offsets[e]) % 126;
                const unsigned source = mod & 1 ? inverse_permutation[w] : permutation[w];
                result[(mod * payload + i) * m + bit] = 8 * carriers[source] + e;
            }
    }
    return result;
}

std::vector<C> Modem::synthesize(std::span<const std::uint8_t> packets, const Tps &mode,
                                 std::size_t count) const {
    const int m = qam_bits(mode);
    if (packets.size() % 188)
        throw std::runtime_error("Randomized packet input must be a multiple of 188 bytes");
    std::vector<std::uint8_t> rs(packets.size() / 188 * 204);
    for (std::size_t p = 0; p < packets.size() / 188; ++p) {
        fec::DvbtRs::Word code{};
        std::copy_n(packets.begin() + static_cast<std::ptrdiff_t>(p * 188), 188, code.begin());
        fec::DvbtRs::encode(code);
        std::copy(code.begin(), code.end(), rs.begin() + static_cast<std::ptrdiff_t>(p * 204));
    }
    std::vector<std::uint8_t> outer(rs.size());
    for (std::size_t i = 0; i < rs.size(); ++i)
        if (i >= (i % 12) * 204)
            outer[i] = rs[i - (i % 12) * 204];
    const auto encoded = fec::convolutional_encode(outer, {}, mode.rate);
    if (encoded.bits.size() < count * payload * static_cast<std::size_t>(m))
        throw std::runtime_error("Insufficient packets for requested symbols");
    const auto points = constellation(m);
    std::vector<C> out(count * bins);
    std::vector<std::uint8_t> labels(payload);
    std::array<std::uint8_t, 68> tps_bits{};
    int tps_sign = 1;
    for (std::size_t s = 0; s < count; ++s) {
        if (s % 68 == 0) {
            tps_bits.fill(0);
            tps_sign = 1;
            auto set = [&](int first, int last, int value) {
                for (int bit = last; bit >= first; --bit) {
                    tps_bits[bit] = value & 1;
                    value >>= 1;
                }
            };
            const int frame = (mode.frame + static_cast<int>(s / 68)) % 4;
            set(1, 16, frame % 2 ? 0xca11 : 0x35ee);
            set(17, 22, 0x17);
            set(23, 24, frame);
            set(25, 26, mode.constellation);
            set(27, 29, mode.hierarchy);
            set(30, 32, mode.rate);
            set(33, 35, mode.rate);
            set(36, 37, mode.guard);
            set(38, 39, mode.mode);
            const auto parity = dvbt::tps_parity(std::span(tps_bits).subspan<1, 53>());
            for (int bit = 0; bit < 14; ++bit)
                tps_bits[54 + bit] = (parity >> bit) & 1;
        } else if (tps_bits[s % 68])
            tps_sign = -tps_sign;
        for (int b = 0; b < payload; b += 126) {
            std::uint8_t streams[6][126]{};
            for (int i = 0; i < 126; ++i)
                for (int bit = 0; bit < m; ++bit)
                    streams[bit / (m / 2) + 2 * (bit % (m / 2))][i] =
                        encoded.bits[(s * payload + b + i) * static_cast<std::size_t>(m) + bit];
            for (int w = 0; w < 126; ++w) {
                labels[b + w] = 0;
                for (int e = 0; e < m; ++e)
                    labels[b + w] =
                        static_cast<std::uint8_t>((labels[b + w] << 1) | streams[e][(w + offsets[e]) % 126]);
            }
        }
        const auto carriers = data_carriers(static_cast<int>(s % 4));
        for (int q = 0; q < payload; ++q) {
            const int src = (s & 1) ? permutation[q] : q, dst = (s & 1) ? q : permutation[q];
            out[s * bins + carriers[dst]] = points[labels[src]];
        }
        for (int k = 0; k < bins; ++k)
            if (out[s * bins + k] == C{})
                out[s * bins + k] = C{(dvbt::polarity[k] ? -1.0F : 1.0F) * 4 / 3, 0};
        for (auto k : dvbt::tps) {
            out[s * bins + k] = C{static_cast<float>(tps_sign * (dvbt::polarity[k] ? -1 : 1)), 0};
        }
    }
    return out;
}
} // namespace radar
