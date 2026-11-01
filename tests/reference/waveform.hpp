#pragma once
#include "modem.hpp"
namespace radar::validation {
class FecDecoder;
struct ReconstructionTimings {
    double demap{}, viterbi{}, reed_solomon{}, reencode{}, remap{};
};
// Independent scalar transmitter/reconstructor for numerical qualification.
class ReferenceModem : public radar::Modem {
    PilotPattern pattern_;
    std::vector<int> permutation_;

  public:
    explicit ReferenceModem(int guard = 16);
    bool tps(std::span<const Symbol>, Tps &) const;
    Reconstruction reconstruct(std::span<const Symbol>, const Tps &, ReconstructionTimings * = nullptr,
                               FecDecoder * = nullptr) const;
};
} // namespace radar::validation
