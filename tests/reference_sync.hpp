#pragma once
#include "modem.hpp"
#include <functional>
#include <memory>
namespace radar {
class SampleClock;
void synchronize_samples(std::span<const C>, int, const std::function<void(Symbol)> &,
                         std::shared_ptr<SampleClock> clock = {}, std::uint64_t origin = 0,
                         unsigned decimation = 1);
} // namespace radar
