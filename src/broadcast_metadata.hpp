#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <stdexcept>
#include <vector>

namespace radar {
// Optional SI inspection after FEC. This does not drive the coherent sample-clock loop.
class BroadcastMetadata {
    std::ofstream output_;
    std::map<unsigned, std::vector<std::uint8_t>> sections_;
    std::map<unsigned, unsigned> continuity_;
    unsigned prbs_ = 0xa9;
    bool randomized_lock_ = false;
    unsigned random_byte() {
        unsigned value = 0;
        for (int i = 0; i < 8; ++i) {
            const auto feedback = ((prbs_ >> 13) ^ (prbs_ >> 14)) & 1U;
            prbs_ = ((prbs_ << 1) | feedback) & 0x7fff;
            value = value * 2 + feedback;
        }
        return value;
    }
    static unsigned u16(const std::uint8_t *p) {
        return unsigned(p[0]) * 256 + p[1];
    }
    void section(unsigned pid, std::span<const std::uint8_t> data) {
        if (data.size() < 12)
            return;
        std::uint32_t crc = 0xffffffffU;
        for (auto byte : data) {
            crc ^= std::uint32_t(byte) << 24;
            for (int i = 0; i < 8; ++i)
                crc = (crc << 1) ^ ((crc & 0x80000000U) ? 0x04c11db7U : 0);
        }
        if (crc)
            return;
        output_ << "section pid=" << pid << " table_id=" << unsigned(data[0])
                << " extension=" << u16(data.data() + 3) << " crc=pass\n";
        if (data[0] != 0x42)
            return;
        // SDT service descriptor (0x48): enough to identify the received multiplex.
        std::size_t at = 11;
        while (at + 5 <= data.size() - 4) {
            const unsigned service = u16(data.data() + at);
            const auto length = ((data[at + 3] & 15U) << 8) | data[at + 4];
            const auto end = at + 5 + length;
            if (end > data.size() - 4)
                return;
            for (auto d = at + 5; d + 2 <= end;) {
                const auto next = d + 2 + data[d + 1];
                if (next > end)
                    return;
                if (data[d] == 0x48 && next >= d + 5) {
                    const auto provider_size = data[d + 3];
                    const auto name_size_at = d + 4 + provider_size;
                    if (name_size_at < next && name_size_at + 1 + data[name_size_at] <= next) {
                        auto printable = [&](std::size_t first, std::size_t n) {
                            for (std::size_t i = first; i < first + n; ++i)
                                output_
                                    << ((data[i] >= 32 && data[i] < 127) ? static_cast<char>(data[i]) : '?');
                        };
                        output_ << "service id=" << service << " provider=\"";
                        printable(d + 4, provider_size);
                        output_ << "\" name=\"";
                        printable(name_size_at + 1, data[name_size_at]);
                        output_ << "\"\n";
                    }
                }
                d = next;
            }
            at = end;
        }
    }
    void feed(unsigned pid, std::span<const std::uint8_t> bytes) {
        auto &buffer = sections_[pid];
        for (auto byte : bytes) {
            if (buffer.empty() && byte == 0xff)
                return;
            buffer.push_back(byte);
            if (buffer.size() >= 3) {
                const auto length = 3 + ((buffer[1] & 15U) << 8) + buffer[2];
                if (length > 4096 || length < 4) {
                    buffer.clear();
                    return;
                }
                if (buffer.size() == length) {
                    section(pid, buffer);
                    buffer.clear();
                }
            }
        }
    }

  public:
    explicit BroadcastMetadata(const std::filesystem::path &path) : output_(path) {}
    void batch(std::span<const std::array<std::uint8_t, 188>> packets, std::span<const std::uint8_t> valid) {
        randomized_lock_ = false;
        sections_.clear();
        continuity_.clear();
        for (std::size_t i = 0; i < packets.size(); ++i) {
            auto packet = packets[i];
            if (!valid[i]) {
                randomized_lock_ = false;
                sections_.clear();
                continue;
            }
            if (packet[0] == 0xb8) {
                prbs_ = 0xa9;
                randomized_lock_ = true;
            }
            if (!randomized_lock_)
                continue;
            for (std::size_t j = 1; j < 188; ++j)
                packet[j] ^= random_byte();
            (void)random_byte();
            packet[0] = 0x47;
            const unsigned pid = ((packet[1] & 31U) << 8) | packet[2];
            if (packet[1] & 0x80 || packet[3] & 0xc0)
                continue;
            const unsigned afc = (packet[3] >> 4) & 3U;
            std::size_t payload = 4;
            if (afc & 2) {
                const auto length = packet[4];
                if (length > 183)
                    continue;
                if (length >= 7 && (packet[5] & 0x10)) {
                    const auto *p = packet.data() + 6;
                    const std::uint64_t base = (std::uint64_t(p[0]) << 25) | (std::uint64_t(p[1]) << 17) |
                                               (std::uint64_t(p[2]) << 9) | (std::uint64_t(p[3]) << 1) |
                                               (p[4] >> 7);
                    const unsigned ext = ((p[4] & 1U) << 8) | p[5];
                    output_ << "pcr pid=" << pid << " ticks_27MHz=" << base * 300 + ext
                            << " clock=program_not_RF\n";
                }
                payload = 5 + length;
            }
            if (!(afc & 1) || payload >= 188)
                continue;
            if (pid == 0x15)
                output_ << "mip_pid_candidate present=1 validated=0\n";
            if (pid != 0x10 && pid != 0x11 && pid != 0x14)
                continue;
            const unsigned cc = packet[3] & 15U;
            if (continuity_.contains(pid) && cc != ((continuity_[pid] + 1) & 15U))
                sections_[pid].clear();
            continuity_[pid] = cc;
            if (packet[1] & 0x40) {
                const auto pointer = packet[payload++];
                if (payload + pointer > 188)
                    continue;
                if (!sections_[pid].empty())
                    feed(pid, std::span{packet}.subspan(payload, pointer));
                sections_[pid].clear();
                payload += pointer;
            } else if (sections_[pid].empty())
                continue;
            feed(pid, std::span{packet}.subspan(payload));
        }
        if (!output_)
            throw std::runtime_error("Broadcast metadata write failed");
    }
};
} // namespace radar
