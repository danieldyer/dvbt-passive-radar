// SPDX-License-Identifier: GPL-3.0-or-later
#include "waveform.hpp"
#include "fec_decoder.hpp"
#include "fec_encode.hpp"
#include <chrono>
#include <numeric>
namespace radar::validation {
namespace {
constexpr int bins = 6817, payload = 6048;
constexpr std::array<int, 6> offsets{0, 63, 105, 42, 21, 84};
std::vector<int> symbol_permutation() {
    constexpr int p[]{7, 1, 4, 2, 9, 6, 8, 10, 0, 3, 11, 5};
    std::vector<int> h;
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
            h.push_back(r);
    }
    if (h.size() != payload)
        throw std::runtime_error("DVB-T permutation generation failed");
    return h;
}

} // namespace
ReferenceModem::ReferenceModem(int guard)
    : radar::Modem(guard), pattern_(pilot_pattern()), permutation_(symbol_permutation()) {}
bool ReferenceModem::tps(std::span<const Symbol> frame, Tps &result) const {
    if (frame.size() != 68)
        return false;
    std::array<std::uint8_t, 68> decoded{};
    for (std::size_t s = 1; s < 68; ++s) {
        double sum = 0;
        for (int j = 0; j < int(pattern_.tps.size()); ++j) {
            const auto k = pattern_.tps[j];
            sum += (frame[s].equalized[k] * std::conj(frame[s - 1].equalized[k])).real();
        }
        decoded[s] = sum < 0;
    }
    return tps_bits(decoded, result);
}
Reconstruction ReferenceModem::reconstruct(std::span<const Symbol> symbols, const Tps &mode,
                                           ReconstructionTimings *timings, FecDecoder *decoder) const {
    using Clock = std::chrono::steady_clock;
    auto mark = timings ? Clock::now() : Clock::time_point{};
    auto measure = [&](double ReconstructionTimings::*stage) {
        if (timings) {
            const auto now = Clock::now();
            timings->*stage += std::chrono::duration<double>(now - mark).count();
            mark = now;
        }
    };
    const int m = qam_bits(mode);
    const auto points = qam_constellation(m);
    std::vector<std::uint8_t> corrected, good;
    Reconstruction result;
    FecDecoder local;
    FecTimings fec_timing;
    const auto decoded =
        (decoder ? *decoder : local).decode(symbols, mode, *this, timings ? &fec_timing : nullptr);
    corrected.resize(decoded.trellis_bits / 8);
    good.resize(corrected.size());
    for (std::size_t p = 0; p < decoded.count; ++p) {
        const auto &packet = decoded.packets[p];
        ++result.packets;
        result.bad_packets += !packet.valid;
        result.corrected_bytes += packet.corrected;
        std::array<std::uint8_t, 188> data{};
        std::copy_n(packet.bytes.begin(), 188, data.begin());
        result.corrected_packets.push_back(data);
        result.packet_valid.push_back(packet.valid);
        for (std::size_t j = 0; j < 204; ++j) {
            const auto i = 2244 + p * 204 + j, destination = i - (11 - i % 12) * 204;
            corrected[destination] = packet.bytes[j];
            good[destination] = packet.valid;
        }
    }
    if (timings) {
        timings->demap += fec_timing.prepare;
        timings->viterbi += fec_timing.viterbi;
        timings->reed_solomon += fec_timing.reed_solomon;
        mark = Clock::now();
    }
    const auto recoded = fec::convolutional_encode(corrected, good, mode.rate);
    measure(&ReconstructionTimings::reencode);
    result.carriers.resize(symbols.size() * bins);
    result.valid.resize(symbols.size() * bins);
    std::vector<std::uint8_t> labels(payload), labels_valid(payload);
    Tps frame_tps{};
    int tps_sign = 1;
    for (std::size_t s = 0; s < symbols.size(); ++s) {
        if (s % 68 == 0) {
            tps_sign = 1;
            if (s + 68 <= symbols.size() && !tps(symbols.subspan(s, 68), frame_tps))
                throw std::runtime_error("Cannot regenerate TPS from a frame without BCH validation");
        } else if (frame_tps.bits[s % 68])
            tps_sign = -tps_sign;
        for (int b = 0; b < payload; b += 126) {
            std::uint8_t streams[6][126]{}, validity[6][126]{};
            for (int i = 0; i < 126; ++i)
                for (int bit = 0; bit < m; ++bit) {
                    const auto at = (s * payload + b + i) * static_cast<std::size_t>(m) + bit;
                    const int e = bit / (m / 2) + 2 * (bit % (m / 2));
                    if (at < recoded.bits.size()) {
                        streams[e][i] = recoded.bits[at];
                        validity[e][i] = recoded.valid[at];
                    }
                }
            for (int w = 0; w < 126; ++w) {
                labels[b + w] = 0;
                labels_valid[b + w] = 1;
                for (int e = 0; e < m; ++e) {
                    labels[b + w] =
                        static_cast<std::uint8_t>((labels[b + w] << 1) | streams[e][(w + offsets[e]) % 126]);
                    labels_valid[b + w] &= validity[e][(w + offsets[e]) % 126];
                }
            }
        }
        const auto carriers = data_carriers(symbols[s].modulo4);
        for (int q = 0; q < payload; ++q) {
            const int src = (symbols[s].modulo4 & 1) ? permutation_[q] : q;
            const int dst = (symbols[s].modulo4 & 1) ? q : permutation_[q];
            result.carriers[s * bins + carriers[dst]] = points[labels[src]];
            result.valid[s * bins + carriers[dst]] = labels_valid[src];
        }
        // Pilots are known; regenerate TPS DBPSK from BCH-checked bits and each carrier's PRBS seed.
        for (int k = 0; k < bins; ++k)
            if (result.carriers[s * bins + k] == C{}) {
                result.carriers[s * bins + k] = C{((pattern_.values[k] < 0) ? -1.0F : 1.0F) * 4 / 3, 0};
                result.valid[s * bins + k] = 1;
            }
        for (int j = 0; j < int(pattern_.tps.size()); ++j) {
            const auto k = pattern_.tps[j];
            result.carriers[s * bins + k] =
                C{static_cast<float>(tps_sign * ((pattern_.values[k] < 0) ? -1 : 1)), 0};
            if ((s / 68) * 68 + 68 > symbols.size())
                result.valid[s * bins + k] = 0;
        }
    }
    measure(&ReconstructionTimings::remap);
    return result;
}

} // namespace radar::validation
