#pragma once
#include "gpu_receiver.hpp"
namespace radar::validation {
// Independent rough pilot interpolation for explicit host qualification only.
std::vector<Symbol> ofdm_symbols(const OfdmBatch &, std::span<const OfdmWindow>);
std::vector<Symbol> receiver_symbols(const ReceiverBatch &);
} // namespace radar::validation
