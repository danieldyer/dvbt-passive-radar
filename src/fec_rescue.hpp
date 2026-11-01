#pragma once
#include "fec.hpp"

namespace radar::fec {
// Bounded symbol-Chase rescue. Likelihoods retain the compact demapper's scale;
// accepted words are constraints, not independent evidence of payload truth.
class RsRescue {
  public:
    static constexpr unsigned maximum_rounds = 2, positions = 4, maximum_failed = 256;
    static constexpr unsigned maximum_unknown_bytes = 65536, maximum_feedback_passes = 16;
    struct Alternative {
        std::uint8_t best{}, second{};
        std::int32_t gap{};
        bool operator==(const Alternative &) const = default;
    };
    static bool eligible(std::span<const std::int16_t> known, std::size_t failed);
    void prepare(std::span<const std::uint8_t> coded, unsigned rate);
    void estimate(std::span<const std::int16_t> known);
    // Only a unique RS-valid, transport-sync-valid word among all 16 trials wins.
    unsigned correct(std::span<RsResult> packets, std::span<const std::uint8_t> selected) const;
    std::span<const Alternative> alternatives() const {
        return alternatives_;
    }

  private:
    using Cost = std::array<std::uint8_t, 4>;
    using State = std::array<std::int32_t, 64>;
    std::span<const std::uint8_t> coded_;
    unsigned rate_{};
    std::size_t steps_{};
    std::vector<Alternative> alternatives_;
    std::vector<Cost> costs_;
    std::vector<State> checkpoints_, scratch_;
    void interval(std::size_t byte_begin, int initial, int final);
    template <unsigned Rate> void estimate_rate(std::span<const std::int16_t> known);
};
} // namespace radar::fec
