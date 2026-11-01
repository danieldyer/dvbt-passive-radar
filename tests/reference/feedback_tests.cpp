#include "fec_feedback.hpp"
#include "pinned_trellis.hpp"
#include <iostream>
#include <random>
int main() {
    using namespace radar;
    std::mt19937 random(89734);
    for (unsigned rate = 0; rate < 5; ++rate)
        for (unsigned cycles : {0U, 1U, 2U, 3U, 31U, 1024U, 4001U})
            for (unsigned fixture = 0; fixture < 4; ++fixture) {
                const auto pattern = fec::puncture_pattern(rate);
                const unsigned sent = std::count(pattern.begin(), pattern.end(), 1U);
                std::vector<std::uint8_t> coded(cycles * sent);
                for (auto &x : coded)
                    x = fixture == 0 ? 7 : fixture == 1 ? (random() & 1 ? 0 : 14) : random() % 15;
                fec::ConstrainedViterbi decoder;
                decoder.prepare(coded, rate);
                std::vector<validation::ByteCost> costs(cycles * pattern.size() / 2);
                std::size_t at = 0;
                auto value = [&] {
                    const int x = coded[at++];
                    return (x < 7 ? -1 : 1) * fec::minimax_llr_levels[std::abs(x - 7)];
                };
                for (std::size_t t = 0; t < costs.size(); ++t) {
                    const auto phase = 2 * t % pattern.size();
                    const int a = pattern[phase] ? value() : 0, b = pattern[phase + 1] ? value() : 0;
                    for (unsigned k = 0; k < 4; ++k)
                        costs[t][k] = std::max(k & 1 ? a : -a, 0) + std::max(k & 2 ? b : -b, 0);
                }
                std::vector<std::int16_t> known(costs.size() / 8, -1);
                const auto initial = decoder(known);
                fec::Viterbi full;
                if (full.decode(coded, rate) != initial)
                    throw std::runtime_error("Initial Viterbi differs from feedback path certificate");
                for (unsigned pass = 0; pass < 30; ++pass) {
                    const auto selected = decoder(known);
                    if (pass)
                        for (auto &x : known)
                            if (random() % 13 == 0 && (x < 0 || pass == 28)) {
                                const auto i = std::size_t(&x - known.data());
                                // Most RS-pinned bytes already agree with the previous path;
                                // corrected bytes force recomputation at their boundaries.
                                x = fixture == 3 && random() % 16 ? selected[i] : random() % 256;
                            }
                    if (pass == 29)
                        std::fill(known.begin(), known.end(), -1);
                    const auto actual = decoder(known),
                               expected = validation::pinned_viterbi<14>(costs, known);
                    if (actual != expected) {
                        std::cerr << "Mismatch " << rate << ' ' << cycles << ' ' << fixture << ' ' << pass
                                  << '\n';
                        return 1;
                    }
                }
                std::fill(known.begin(), known.end(), -1);
                const auto selected = decoder(known);
                for (std::size_t i = 0; i < known.size(); i += 13)
                    known[i] = selected[i];
                if (decoder(known) != selected || decoder.decoded_steps() != 0)
                    throw std::runtime_error("Compatible ML subpaths were needlessly recomputed");
                decoder.prepare(coded, rate);
                if (decoder(known) != validation::pinned_viterbi<14>(costs, known))
                    throw std::runtime_error("Preparation retained a stale path certificate");
            }
    {
        std::vector<std::uint8_t> coded(256, 7);
        std::vector<std::int16_t> known(16, -1);
        known[7] = 91;
        fec::ConstrainedViterbi decoder;
        decoder.prepare(coded, 0);
        const auto expected = decoder(known);
        for (auto invalid : {-32768, -2, 256, 32767})
            for (unsigned at : {0U, 7U, 15U}) {
                auto bad = known;
                bad[at] = invalid;
                bool rejected = false;
                try {
                    decoder(bad);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                if (!rejected || decoder(known) != expected)
                    throw std::runtime_error("Invalid constraint accepted or changed decoder state");
            }
        for (unsigned invalid : {15U, 255U})
            for (unsigned at : {0U, 127U, 255U}) {
                auto bad = coded;
                bad[at] = invalid;
                bool rejected = false;
                try {
                    decoder.prepare(bad, 0);
                } catch (const std::invalid_argument &) {
                    rejected = true;
                }
                if (!rejected || decoder(known) != expected)
                    throw std::runtime_error("Invalid compact metric accepted or changed decoder state");
            }
    }
    std::cout
        << "All-rate certified constrained trellis matches independent reference, including ties, partial "
           "tails, new/revised/removed constraints.\n";
}
