#include "dvbt_configure.h"
#include "dvbt_pilot_gen.h"
#include "dvbt_pilots.hpp"
#include "modem.hpp"
#include <algorithm>
#include <deque>
#include <iostream>
#include <random>

namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
} // namespace
int main() {
    using namespace radar;
    std::mt19937 random(791623);
    for (unsigned guard : {4U, 8U, 16U, 32U}) {
        const auto interval = guard == 4    ? gr::dtv::GI_1_4
                              : guard == 8  ? gr::dtv::GI_1_8
                              : guard == 16 ? gr::dtv::GI_1_16
                                            : gr::dtv::GI_1_32;
        gr::dl5eu::dvbt_configure config(gr::dtv::MOD_64QAM, gr::dtv::NH, gr::dtv::C3_4, gr::dtv::C3_4,
                                         interval, gr::dtv::T8k);
        gr::dl5eu::dvbt_pilot_gen reference(config);
        Modem modem(guard);
        const auto actual = modem.pilot_pattern();
        const auto polarity = reference.pilot_values();
        require(std::equal(actual.continual.begin(), actual.continual.end(), reference.cpilot_carriers(),
                           reference.cpilot_carriers() + reference.cpilot_carriers_size()),
                "Continual pilot positions changed");
        require(std::equal(actual.tps.begin(), actual.tps.end(), reference.tps_carriers(),
                           reference.tps_carriers() + reference.tps_carriers_size()),
                "TPS pilot positions changed");
        for (unsigned k = 0; k < actual.values.size(); ++k)
            require(actual.values[k] == (polarity[k] ? -4.F / 3 : 4.F / 3), "Pilot PRBS changed");
        for (unsigned phase = 0; phase < 4; ++phase) {
            std::array<bool, 6817> excluded{};
            for (int i = 0; i < reference.cpilot_carriers_size(); ++i)
                excluded[reference.cpilot_carriers()[i]] = true;
            for (int i = 0; i < reference.tps_carriers_size(); ++i)
                excluded[reference.tps_carriers()[i]] = true;
            for (unsigned k = 3 * phase; k < excluded.size(); k += 12)
                excluded[k] = true;
            std::vector<int> expected;
            for (unsigned k = 0; k < excluded.size(); ++k)
                if (!excluded[k])
                    expected.push_back(k);
            require(modem.data_carriers(phase) == expected, "Data carrier partition changed");
        }
        for (unsigned trial = 0; trial < 1000; ++trial) {
            std::array<std::uint8_t, 68> bits{};
            for (auto &bit : bits)
                bit = random() & 1;
            const unsigned frame = trial % 4, sync = frame % 2 ? 0xca11 : 0x35ee;
            for (unsigned i = 0; i < 16; ++i)
                bits[1 + i] = (sync >> (15 - i)) & 1;
            bits[23] = frame >> 1;
            bits[24] = frame & 1;
            const auto parity = dvbt::tps_parity(std::span(bits).subspan<1, 53>());
            for (unsigned i = 0; i < 14; ++i)
                bits[54 + i] = (parity >> i) & 1;
            const std::deque<char> word(bits.begin(), bits.end());
            require(reference.verify_bch_code(word) == 0, "TPS parity differs from upstream");
            Tps decoded;
            require(modem.tps_bits(bits, decoded), "Valid TPS rejected");
            require(decoded.frame == int(frame) && decoded.bits == bits, "TPS metadata changed");
            for (unsigned position = 1; position < bits.size(); ++position) {
                bits[position] ^= 1;
                require(!modem.tps_bits(bits, decoded), "Corrupt TPS accepted");
                bits[position] ^= 1;
            }
        }
    }
    std::cout << "All guards: pilot positions, PRBS, carrier partitions and TPS BCH match upstream.\n";
}
