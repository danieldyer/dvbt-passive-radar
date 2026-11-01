#pragma once
#include "fec.hpp"
#include "fec_feedback.hpp"
#include "fec_rescue.hpp"
namespace radar {
// Thread CPU seconds, summed across workers; excludes scheduling/wait time.
struct FecTimings {
    double prepare{}, viterbi{}, reed_solomon{};
    double initial_viterbi{}; // Subset of viterbi; the remainder is constrained feedback.
};
struct FecBatch {
    // Borrowed until the next decode; corrected RS words retain their coordinates.
    std::span<const fec::RsResult> packets;
    std::size_t count{}, trellis_bits{}, reused_trellis_bits{};
    unsigned passes{}, rescue_rounds{}, rescued_words{};
};
inline constexpr unsigned maximum_fec_passes = 32;
inline constexpr unsigned default_fec_workers = 5;
inline constexpr std::size_t maximum_coded_bits = 1020U * 6048 * 6;
class CpuFec {
    fec::Viterbi viterbi_;
    fec::ConstrainedViterbi constrained_;
    fec::RsRescue rescue_;
    std::vector<std::int16_t> known_;
    std::vector<unsigned> unresolved_;
    static constexpr unsigned passes_ = maximum_fec_passes;
    unsigned passes_used_{}, rescue_rounds_{}, rescued_words_{};
    std::vector<fec::RsResult> packets_;
    void pin(std::size_t);
    unsigned refine(FecTimings *);
    bool rescue(std::span<const std::uint8_t>, unsigned, FecTimings *);

  public:
    // Fixed minimax half-LLR codebook, full traceback and bounded RS feedback.
    // The caller owns these immutable CPU bytes through this synchronous call.
    FecBatch decode(std::span<const std::uint8_t>, unsigned rate, FecTimings * = nullptr);
};
} // namespace radar
