#pragma once
#include "fec.hpp"
namespace radar::validation {
// A bit is 0 or 1; 2 means an erased observation (including a non-finite carrier).
// Full survivor traceback, unknown initial/final state, no guessed termination.
enum class SoftMetric { uniform, minimax };
class ReferenceViterbi {
    std::vector<std::uint64_t> survivors_;
    std::vector<std::uint8_t> cached_bits_;
    std::vector<std::array<std::uint8_t, 64>> checkpoints_, next_checkpoints_;
    unsigned cached_rate_ = 15;
    std::size_t reused_steps_{};
    template <unsigned Rate, bool Erasures, bool Soft = false, bool Companded = false, bool Cache = true>
    std::vector<std::uint8_t> decode_rate(std::span<const std::uint8_t>, std::size_t overlap);

  public:
    // Optional repeated suffix/prefix, in coded bits. Reuse requires identical
    // observations AND exact equality of all normalized state metrics. A miss
    // falls back to the complete trellis; traceback is never shortened.
    std::vector<std::uint8_t> decode(std::span<const std::uint8_t> bits, unsigned rate,
                                     std::size_t overlap = 0);
    // Quantised LLR+7: 0..14, positive LLR favours zero, 7 is an erasure.
    // Both encodings use 0..14 with erasure 7. Minimax entries index the fixed
    // nonuniform half-LLR codebook; they are not linear numeric likelihoods.
    std::vector<std::uint8_t> decode_soft(std::span<const std::uint8_t>, unsigned rate,
                                          std::size_t overlap = 0, SoftMetric = SoftMetric::uniform);
    std::size_t reused_steps() const {
        return reused_steps_;
    }
};

} // namespace radar::validation
