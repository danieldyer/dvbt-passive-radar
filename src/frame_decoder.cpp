#include "frame_decoder.hpp"
#include "gpu_commands.hpp"
#include "thread_name.hpp"
#include <condition_variable>
#include <deque>
#include <future>
#include <thread>

namespace radar {
struct FrameDecoder::Impl {
    struct Slot {
        GpuFrameStorage storage;
        DecodedFrame output;
        gpu::Completion ready;
        std::promise<void> completion;
        std::shared_future<void> finished;
        explicit Slot(std::shared_ptr<gpu::Context> context) : storage(std::move(context)) {}
    };
    std::vector<std::unique_ptr<Slot>> storage;
    std::deque<Slot *> free, pending, work;
    Slot *selected = nullptr;
    std::mutex mutex;
    std::condition_variable condition;
    bool stopping = false;
    std::vector<std::jthread> workers;
    Impl(unsigned count, unsigned slots, std::shared_ptr<gpu::Context> context) {
        if (!count || count > 64 || slots < count || slots > 128)
            throw std::invalid_argument("Invalid bounded FEC worker configuration");
        for (unsigned i = 0; i < slots; ++i) {
            storage.push_back(std::make_unique<Slot>(context));
            free.push_back(storage.back().get());
        }
        try {
            for (unsigned i = 0; i < count; ++i)
                workers.emplace_back([this] { decode(); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Impl() {
        stop();
    }
    Slot *ready() const {
        if (selected)
            return selected;
        std::uint64_t blocked = 0;
        for (auto *slot : pending) {
            const auto channel = std::uint64_t{1} << slot->output.frame.channel;
            const bool first = !(blocked & channel);
            blocked |= channel;
            if (first && slot->finished.wait_for(std::chrono::seconds{0}) == std::future_status::ready)
                return slot;
        }
        return nullptr;
    }
    void stop() {
        {
            const std::lock_guard lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        workers.clear(); // Join before snapshots or borrowed tables are destroyed.
    }
    void decode() {
        name_thread("dvbt-fec");
        // Feedback batches have a complete trellis and overlapping context.
        // Worker-owned scratch permits independent batches of one channel to
        // decode concurrently; ready() still publishes each channel in order.
        CpuFec decoder;
        for (;;) {
            Slot *slot;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return stopping || !work.empty(); });
                if (work.empty())
                    return;
                slot = work.front();
                work.pop_front();
            }
            try {
                slot->ready.wait();
                auto &out = slot->output;
                out.timing = {};
                const auto &carriers = out.frame.carriers;
                if (!carriers.count || carriers.count > 1020 ||
                    carriers.coded_metric != CodedMetric::minimax_likelihood || !carriers.coded_bits.mapped)
                    throw std::invalid_argument("Invalid compact FEC buffer");
                const auto count = carriers.count * 6048 * qam_bits(out.frame.frames.front());
                if (carriers.coded_bits.bytes < count)
                    throw std::invalid_argument("Incomplete compact FEC buffer");
                static_assert(std::endian::native == std::endian::little);
                const auto decoded =
                    decoder.decode({static_cast<const std::uint8_t *>(carriers.coded_bits.mapped), count},
                                   out.frame.frames.front().rate, &out.timing);
                out.packets.assign(decoded.packets.begin(), decoded.packets.end());
                out.trellis_bits = decoded.trellis_bits;
                out.reused_trellis_bits = decoded.reused_trellis_bits;
                out.passes = decoded.passes;
                out.rescue_rounds = decoded.rescue_rounds;
                out.rescued_words = decoded.rescued_words;
                slot->completion.set_value();
            } catch (...) {
                slot->completion.set_exception(std::current_exception());
            }
            condition.notify_all();
        }
    }
};
FrameDecoder::FrameDecoder(unsigned workers, unsigned slots, std::shared_ptr<gpu::Context> context)
    : impl_(std::make_unique<Impl>(workers, slots, std::move(context))) {}
FrameDecoder::~FrameDecoder() = default;
bool FrameDecoder::full() const {
    return impl_->free.empty();
}
bool FrameDecoder::empty() const {
    return impl_->pending.empty();
}
bool FrameDecoder::ready() const {
    return impl_->ready() != nullptr;
}
void FrameDecoder::submit(const FrameSelection &selection, gpu::Commands &commands, ComputeTimings *timings) {
    auto &p = *impl_;
    if (full())
        throw std::logic_error("FEC frame queue is full");
    const auto &frame = selection.frame;
    if (frame.channel >= 64 || frame.frames.empty() || frame.frames.size() > 15 ||
        selection.rows.size() != 68 * frame.frames.size() || frame.symbols.size() != selection.rows.size() ||
        frame.first >= frame.last || frame.last > selection.rows.size())
        throw std::invalid_argument("Invalid queued FEC frame coordinates");
    auto *slot = p.free.front();
    p.free.pop_front();
    slot->output.frame = slot->storage.capture(selection, timings, &commands);
    slot->ready = commands.submit();
    slot->completion = std::promise<void>{};
    slot->finished = slot->completion.get_future().share();
    p.pending.push_back(slot);
    {
        const std::lock_guard lock(p.mutex);
        p.work.push_back(slot);
    }
    p.condition.notify_one();
}
const DecodedFrame &FrameDecoder::front() {
    if (empty())
        throw std::logic_error("FEC frame queue is empty");
    auto &p = *impl_;
    if (!p.selected)
        p.selected = p.ready();
    if (!p.selected)
        p.selected = p.pending.front(); // Backpressure: wait for the oldest queued head.
    auto &slot = *p.selected;
    slot.finished.get();
    slot.storage.wait();
    return slot.output;
}
void FrameDecoder::pop() {
    if (empty())
        throw std::logic_error("FEC frame queue is empty");
    auto &p = *impl_;
    if (!p.selected)
        throw std::logic_error("No decoded frame selected for release");
    p.free.push_back(p.selected);
    std::erase(p.pending, p.selected);
    p.selected = nullptr;
}
} // namespace radar
