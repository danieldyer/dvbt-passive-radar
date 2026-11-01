#include "conditional_chase_reference.hpp"
#include "fec_rescue.hpp"
#include <iostream>
#include <random>

using namespace radar;
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
int main() {
    try {
        std::mt19937 random(81724);
        fec::RsRescue rescue;
        for (unsigned rate = 0; rate < 5; ++rate)
            for (unsigned cycles : {1U, 7U, 1024U, 2051U})
                for (unsigned fixture = 0; fixture < 3; ++fixture) {
                    const auto pattern = fec::puncture_pattern(rate);
                    const auto sent = std::count(pattern.begin(), pattern.end(), 1U);
                    std::vector<std::uint8_t> coded(cycles * sent);
                    for (auto &x : coded)
                        x = fixture == 0 ? 7 : fixture == 1 ? (random() & 1 ? 0 : 14) : random() % 15;
                    validation::CompactBranchCosts costs(coded, rate);
                    std::vector<std::int16_t> known(costs.steps() / 8, -1);
                    if (fixture)
                        for (auto &x : known)
                            if (random() % 5 == 0)
                                x = random() % 256;
                    rescue.prepare(coded, rate);
                    rescue.estimate(known);
                    const auto expected = validation::conditional_byte_alternatives_at(
                        costs.steps(), known, [&](auto t) { return costs(t); });
                    const auto actual = rescue.alternatives();
                    require(actual.size() == expected.map_bytes.size(), "Posterior length changed");
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        require(actual[i].best == expected.map_bytes[i] &&
                                    actual[i].second == expected.second_bytes[i] &&
                                    float(actual[i].gap) == expected.alternative_gap[i],
                                "Production posterior differs from independent float reference");
                }
        // The unnormalised production posterior must remain exact at its work cap,
        // including forced endpoints and worst-magnitude rate-1/2 observations.
        for (unsigned fixture = 0; fixture < 3; ++fixture) {
            constexpr unsigned bytes = fec::RsRescue::maximum_unknown_bytes;
            std::vector<std::int16_t> known(bytes + 2, -1);
            if (fixture) {
                known.front() = random() % 256;
                known.back() = random() % 256;
            } else
                known.resize(bytes);
            std::vector<std::uint8_t> coded(16 * known.size());
            for (auto &x : coded)
                x = fixture == 0 ? 7 : fixture == 1 ? (random() & 1 ? 0 : 14) : random() % 15;
            validation::CompactBranchCosts costs(coded, 0);
            rescue.prepare(coded, 0);
            rescue.estimate(known);
            const auto expected = validation::conditional_byte_alternatives_with(
                costs.steps(), known, [&](auto t) { return costs(t); },
                validation::integer_byte_alternatives);
            for (std::size_t i = 0; i < known.size(); ++i) {
                const auto &a = rescue.alternatives()[i];
                require(a.best == expected.map_bytes[i] && a.second == expected.second_bytes[i] &&
                            float(a.gap) == expected.alternative_gap[i],
                        "Maximum-length posterior changed after removing normalisation");
            }
        }
        std::vector<std::int16_t> excessive(fec::RsRescue::maximum_unknown_bytes + 1, -1);
        require(!fec::RsRescue::eligible(excessive, 1), "Unknown-byte work bound ignored");
        require(!fec::RsRescue::eligible({}, 257) && !fec::RsRescue::eligible({}, 0),
                "Failed-word work bound ignored");
        const auto rejected = [](auto operation) {
            try {
                operation();
            } catch (const std::invalid_argument &) {
                return true;
            }
            return false;
        };
        std::array<std::uint8_t, 16> metrics;
        metrics.fill(7);
        for (unsigned bad : {15U, 255U}) {
            metrics[7] = bad;
            require(rejected([&] { rescue.prepare(metrics, 0); }), "Invalid compact metric was accepted");
        }
        metrics.fill(7);
        rescue.prepare(metrics, 0);
        for (std::int16_t bad : {std::int16_t(-2), std::int16_t(256)}) {
            const std::array known{bad};
            require(rejected([&] { rescue.estimate(known); }), "Invalid byte constraint was accepted");
        }
        require(rejected([&] { rescue.estimate({}); }), "Wrong posterior length was accepted");

        constexpr unsigned count = 24, bytes = (count + 11) * 204;
        const auto index = [](unsigned p, unsigned j) {
            const auto i = 2244 + p * 204 + j;
            return i - (11 - i % 12) * 204;
        };
        std::vector<fec::RsResult> words(count);
        std::vector<std::uint8_t> selected(bytes), coded(bytes * 8 / 3 * 4, 7);
        for (unsigned p = 0; p < count; ++p) {
            auto &word = words[p].bytes;
            for (auto &x : word)
                x = random();
            word[0] = 0x47;
            fec::DvbtRs::encode(word);
            for (unsigned j = 0; j < 204; ++j)
                selected[index(p, j)] = word[j];
        }
        const auto truth = words;
        std::vector<std::int16_t> known(selected.begin(), selected.end());
        rescue.prepare(coded, 2);
        rescue.estimate(known);
        for (unsigned p = 0; p < count; ++p) {
            const unsigned errors = 8 + p % 6;
            for (unsigned j = 0; j < errors; ++j)
                selected[index(p, 20 + j)] ^= std::uint8_t(j + 1);
            words[p].valid = p % 7 == 0;
            words[p].corrected = words[p].valid ? 17 : 0;
        }
        auto expected = words;
        const auto alternatives = rescue.alternatives();
        for (unsigned p = 0; p < count; ++p)
            if (!expected[p].valid) {
                fec::DvbtRs::Word received, alternate;
                std::array<float, 204> gaps;
                for (unsigned j = 0; j < 204; ++j) {
                    const auto source = index(p, j);
                    received[j] = selected[source];
                    const auto &a = alternatives[source];
                    const bool same = a.best == received[j];
                    alternate[j] = same ? a.second : a.best;
                    gaps[j] = same ? a.gap : 0;
                }
                const auto candidates = validation::chase_candidates(received, alternate, gaps, 4);
                if (candidates.size() == 1) {
                    expected[p].bytes = candidates.front();
                    expected[p].valid = true;
                    expected[p].corrected = 0;
                    for (unsigned j = 0; j < 204; ++j)
                        expected[p].corrected += received[j] != expected[p].bytes[j];
                }
            }
        rescue.correct(words, selected);
        for (unsigned p = 0; p < count; ++p) {
            require(words[p].valid == expected[p].valid && words[p].bytes == expected[p].bytes &&
                        words[p].corrected == expected[p].corrected,
                    "Production Chase changed reference candidate acceptance");
            if (p % 6 <= 4 && p % 7)
                require(words[p].valid && words[p].bytes == truth[p].bytes,
                        "Lost known-payload rescue beyond eight errors");
        }
        // Adjacent decoding spheres: the list contains two different valid words.
        // Reject the word, even though each individual bounded-distance RS decode succeeds.
        auto a = truth[0].bytes, b = a;
        b[20] ^= 1;
        fec::DvbtRs::encode(b);
        std::vector<unsigned> differences;
        for (unsigned j = 0; j < 204; ++j)
            if (a[j] != b[j])
                differences.push_back(j);
        require(differences.size() == 17, "Expected minimum-distance RS pair");
        auto received = a;
        for (unsigned i = 0; i < 8; ++i)
            received[differences[i]] = b[differences[i]];
        auto second = received;
        second[differences[8]] = b[differences[8]];
        auto decode_a = received, decode_b = second;
        require(fec::DvbtRs::decode(decode_a) >= 0 && decode_a == a && fec::DvbtRs::decode(decode_b) >= 0 &&
                    decode_b == b,
                "Ambiguity fixture invalid");
        for (unsigned p = 0; p < count; ++p) {
            words[p] = truth[p];
            words[p].valid = p != 0;
            for (unsigned j = 0; j < 204; ++j) {
                selected[index(p, j)] = p ? truth[p].bytes[j] : received[j];
                known[index(p, j)] = p ? truth[p].bytes[j] : second[j];
            }
        }
        rescue.estimate(known);
        require(rescue.correct(words, selected) == 0 && !words[0].valid, "Ambiguous RS list was accepted");
        std::cout << "Production rescue matches all-rate posteriors and Chase reference; known payload, "
                     "ambiguity, preservation and work gates pass.\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
