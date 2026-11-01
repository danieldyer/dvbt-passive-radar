#include "cpu_fec.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <ctime>
namespace radar {
namespace {
// Count CPU consumed by this worker, excluding preemption by other workers or
// unrelated processes. Summing these measurements gives core-seconds of work.
struct Clock {
    using duration = std::chrono::nanoseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<Clock>;
    [[maybe_unused]] static constexpr bool is_steady = true;
    static time_point now() {
        timespec value{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
        return time_point{std::chrono::seconds{value.tv_sec} + duration{value.tv_nsec}};
    }
};
double seconds(Clock::time_point from) {
    return std::chrono::duration<double>(Clock::now() - from).count();
}
} // namespace
void CpuFec::pin(std::size_t p) {
    for (unsigned j = 0; j < 204; ++j) {
        const auto i = 2244 + p * 204 + j;
        known_[i - (11 - i % 12) * 204] = packets_[p].bytes[j];
    }
}
unsigned CpuFec::refine(FecTimings *timing) {
    auto start = timing ? Clock::now() : Clock::time_point{};
    const auto &refined = constrained_(known_);
    ++passes_used_;
    if (timing) {
        timing->viterbi += seconds(start);
        start = Clock::now();
    }
    const auto added = std::erase_if(unresolved_, [&](unsigned p) {
        auto &packet = packets_[p];
        fec::DvbtRs::Word received;
        for (unsigned j = 0; j < 204; ++j) {
            const auto i = 2244 + p * 204 + j;
            received[j] = refined[i - (11 - i % 12) * 204];
        }
        // An unresolved word already failed on these exact bytes. Pure RS
        // decoding cannot change its result until a trellis byte changes.
        if (received == packet.bytes)
            return false;
        packet.bytes = received;
        const int errors = fec::DvbtRs::decode(packet.bytes);
        packet.valid = errors >= 0 && (packet.bytes[0] == 0x47 || packet.bytes[0] == 0xb8);
        packet.corrected = packet.valid ? unsigned(errors) : 0;
        if (packet.valid)
            pin(p);
        return packet.valid;
    });
    if (timing)
        timing->reed_solomon += seconds(start);
    return added;
}
bool CpuFec::rescue(std::span<const std::uint8_t> coded, unsigned rate, FecTimings *timing) {
    if (!fec::RsRescue::eligible(known_, unresolved_.size()))
        return false;
    auto start = timing ? Clock::now() : Clock::time_point{};
    rescue_.prepare(coded, rate);
    if (timing)
        timing->prepare += seconds(start);
    const auto end = std::min(passes_, passes_used_ + fec::RsRescue::maximum_feedback_passes);
    const auto before = rescued_words_;
    for (unsigned round = 0; round < fec::RsRescue::maximum_rounds && passes_used_ < end; ++round) {
        if (!fec::RsRescue::eligible(known_, unresolved_.size()))
            break;
        start = timing ? Clock::now() : Clock::time_point{};
        const auto &selected = constrained_(known_);
        ++passes_used_;
        rescue_.estimate(known_);
        if (timing) {
            timing->viterbi += seconds(start);
            start = Clock::now();
        }
        const auto added = rescue_.correct(packets_, selected);
        ++rescue_rounds_;
        rescued_words_ += added;
        std::erase_if(unresolved_, [&](unsigned p) {
            if (!packets_[p].valid)
                return false;
            pin(p);
            return true;
        });
        if (timing)
            timing->reed_solomon += seconds(start);
        if (!added)
            break;
        while (passes_used_ < end && !unresolved_.empty())
            if (!refine(timing))
                break;
    }
    return rescued_words_ != before;
}
FecBatch CpuFec::decode(std::span<const std::uint8_t> bits, unsigned rate, FecTimings *timing) {
    if (bits.size() > maximum_coded_bits)
        throw std::invalid_argument("FEC batch exceeds the supported 1020-symbol bound");
    passes_used_ = 1;
    rescue_rounds_ = rescued_words_ = 0;
    auto start = timing ? Clock::now() : Clock::time_point{};
    const auto bytes = viterbi_.decode(bits, rate);
    if (timing) {
        const auto elapsed = seconds(start);
        timing->viterbi += elapsed;
        timing->initial_viterbi += elapsed;
        start = Clock::now();
    }
    packets_.resize(bytes.size() / 204 > 11 ? bytes.size() / 204 - 11 : 0);
    for (std::size_t p = 0; p < packets_.size(); ++p) {
        auto &packet = packets_[p];
        for (unsigned j = 0; j < 204; ++j) {
            const auto i = 2244 + p * 204 + j;
            packet.bytes[j] = bytes[i - (11 - i % 12) * 204];
        }
        const int errors = fec::DvbtRs::decode(packet.bytes);
        packet.valid = errors >= 0 && (packet.bytes[0] == 0x47 || packet.bytes[0] == 0xb8);
        packet.corrected = packet.valid ? static_cast<unsigned>(errors) : 0;
    }
    if (timing)
        timing->reed_solomon += seconds(start);
    {
        const auto accepted = std::ranges::count_if(packets_, [](const auto &p) { return p.valid; });
        if (accepted && accepted < std::ssize(packets_)) {
            start = timing ? Clock::now() : Clock::time_point{};
            constrained_.prepare(bits, rate);
            constrained_.seed(bytes);
            known_.assign(bytes.size(), -1);
            if (timing) {
                timing->prepare += seconds(start);
                start = Clock::now();
            }
            unresolved_.clear();
            for (unsigned p = 0; p < packets_.size(); ++p) {
                if (packets_[p].valid)
                    pin(p);
                else
                    unresolved_.push_back(p);
            }
            if (timing)
                timing->reed_solomon += seconds(start);
            bool rescue_attempted = false;
            while (passes_used_ < passes_) {
                const auto added = refine(timing);
                bool rescued = false;
                // Reserve up to half the 32-pass budget for the bounded rescue;
                // an earlier fixed point may use the same bounded stage.
                if (!rescue_attempted && (passes_used_ == 16 || !added) && !unresolved_.empty() &&
                    passes_used_ < passes_) {
                    rescue_attempted = true;
                    rescued = rescue(bits, rate, timing);
                }
                if ((!added && !rescued) || unresolved_.empty())
                    break;
            }
        }
    }
    const auto pattern = fec::puncture_pattern(rate);
    return {packets_,
            packets_.size(),
            bits.size() / std::count(pattern.begin(), pattern.end(), 1U) * (pattern.size() / 2),
            0,
            passes_used_,
            rescue_rounds_,
            rescued_words_};
}
} // namespace radar
