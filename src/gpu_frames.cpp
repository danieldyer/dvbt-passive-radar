#include "gpu_frames.hpp"
#include "gpu_compute.hpp"
#include "pilot_frequency_filter.hpp"
#include <cmath>
#include <deque>
#include <numbers>

namespace radar {
namespace {
constexpr std::uint32_t copy_code[] =
#include "carrier_copy.inc"
    ;
constexpr std::uint32_t equalize_code[] =
#include "frame_equalize.inc"
    ;
constexpr std::uint32_t demap_code[] =
#include "frame_demap.inc"
    ;
constexpr std::uint32_t soft_stats_code[] =
#include "frame_soft_stats.inc"
    ;
constexpr std::uint32_t soft_noise_code[] =
#include "frame_soft_noise.inc"
    ;
constexpr std::uint32_t phase_code[] =
#include "frame_phase_scan.inc"
    ;
constexpr std::uint32_t align_code[] =
#include "frame_pilot_align.inc"
    ;
constexpr std::uint32_t risk_code[] =
#include "frame_pilot_risk.inc"
    ;
constexpr std::uint32_t select_code[] =
#include "frame_pilot_select.inc"
    ;
constexpr std::uint32_t tiled_pilot_code[] =
#include "frame_pilots_tiled.inc"
    ;
struct FramePasses {
    vk::Pipeline equalize, pilots, demap, stats, noise, frequency, phase, align, risk, select, continual;
    unsigned bits{};
};
constexpr std::uint32_t frequency_code[] =
#include "frame_frequency.inc"
    ;
OfdmBatch copy_rows(gpu::Kernel &kernel, const OfdmBatch &source, std::span<const std::uint32_t> rows,
                    unsigned destination, unsigned size, ComputeTimings *timings,
                    gpu::Commands *commands = nullptr, const FramePasses &passes = {}) {
    const auto &[pipeline, pilots, demap, stats, noise, frequency, phase, align, risk, select, continual,
                 bits] = passes;
    if (rows.empty() || rows.size() > size || source.pilot_stride < 2 || source.pilot_stride > 6817 ||
        source.measured.bytes < source.count * source.carrier_stride * sizeof(C) ||
        source.pilots.bytes < source.count * source.pilot_stride * sizeof(C))
        throw std::invalid_argument("Invalid carrier gather dimensions");
    for (auto row : rows)
        if (row >= source.count)
            throw std::invalid_argument("Carrier gather row outside source");
    if (source.carrier_stride != 6817 && source.carrier_stride != 8192)
        throw std::invalid_argument("Unsupported carrier storage geometry");
    if (source.carrier_stride == 8192) {
        if (pipeline || source.phase_corrections.context != source.measured.context ||
            !source.phase_corrections.mapped || source.phase_corrections.bytes < source.count * sizeof(C))
            throw std::invalid_argument(
                "Native FFTs require complete phase coordinates before the frame ring");
        // A frame callback can split the command group. Own the small host-
        // written phase vector until THIS gather completes; producer FFT data
        // stays on the queue, but its host phase buffer may already be reused.
        kernel.upload<C>(1, {static_cast<const C *>(source.phase_corrections.mapped), source.count});
    }
    kernel.bind(0, source.measured);
    kernel.bind(2, source.pilots);
    kernel.storage(3, size * 6817 * sizeof(C));
    if (pipeline)
        kernel.storage(4, size * 6817 * sizeof(C));
    kernel.storage(5, size * source.pilot_stride * sizeof(C));
    kernel.upload(6, rows);
    const auto cmd = kernel.begin(timings, timings && timings->detailed ? 6 : 2, commands);
    const std::array push{static_cast<unsigned>(rows.size()), static_cast<unsigned>(source.pilot_stride),
                          demap ? bits : destination, size};
    if (stats) {
        kernel.dispatch(cmd, stats, push, rows.size() * 64);
        kernel.barrier(cmd);
    }
    if (phase) {
        kernel.groups(cmd, phase, push, 1);
        kernel.barrier(cmd);
    }
    if (timings && timings->detailed)
        kernel.stamp(cmd, 1);
    if (noise) {
        kernel.dispatch(cmd, noise, push, 128 * 64);
        kernel.barrier(cmd);
    }
    if (timings && timings->detailed)
        kernel.stamp(cmd, 2);
    if (align) {
        kernel.dispatch(cmd, align, push, rows.size() * 2273);
        kernel.barrier(cmd);
    }
    if (risk) {
        kernel.groups(cmd, risk, push, 2273);
        kernel.barrier(cmd);
    }
    if (select) {
        kernel.groups(cmd, select, push, 1);
        kernel.barrier(cmd);
    }
    if (pilots) {
        kernel.groups(cmd, pilots, push, (2273 + 15) / 16, (unsigned(rows.size()) + 15) / 16);
        kernel.barrier(cmd);
    }
    if (continual) {
        kernel.groups(cmd, continual, push, (177 + 15) / 16, (unsigned(rows.size()) + 15) / 16);
        kernel.barrier(cmd);
    }
    if (frequency) {
        kernel.groups(cmd, frequency, push, (2273 + 63) / 64, unsigned(rows.size()));
        kernel.barrier(cmd);
    }
    if (timings && timings->detailed)
        kernel.stamp(cmd, 3);
    kernel.dispatch(cmd, pipeline ? pipeline : kernel.primary(), push, rows.size() * 6817);
    if (timings && timings->detailed)
        kernel.stamp(cmd, 4);
    if (demap) {
        kernel.barrier(cmd);
        kernel.groups(cmd, demap, push, unsigned(rows.size()));
    }
    kernel.finish(cmd);
    return {kernel.view(3),      pipeline ? kernel.view(4) : gpu::BufferView{}, kernel.view(5), rows.size(),
            source.pilot_stride, demap ? kernel.view(11) : gpu::BufferView{}};
}
} // namespace
struct GpuFrameStorage::Impl {
    gpu::Kernel kernel;
    vk::UniquePipeline pilots, stats, noise, frequency, phase, align, risk, select, continual;
    static constexpr DecoderPolicy options{};
    std::array<vk::UniquePipeline, 3> equalize, demap;
    int guard = -1, constellation = -1;
    explicit Impl(std::shared_ptr<gpu::Context> context) : kernel(copy_code, 16, {}, std::move(context)) {
        const auto limits = kernel.context()->gpu.physical.getProperties().limits;
        if (limits.maxComputeSharedMemorySize < 32768 || limits.maxComputeWorkGroupInvocations < 512 ||
            limits.maxComputeWorkGroupSize[0] < 512 || limits.maxComputeWorkGroupSize[1] < 16)
            throw std::runtime_error("Capture requires 32 KiB workgroup storage and 512 invocations");
        pilots = kernel.pipeline(tiled_pilot_code, std::array{options.pilot_radius, 1U, 0U});
        continual = kernel.pipeline(tiled_pilot_code, std::array{options.pilot_radius, 1U, 1U});
        risk = kernel.pipeline(risk_code, std::array{32U});
        select = kernel.pipeline(select_code, std::array{32U});
        stats = kernel.pipeline(soft_stats_code);
        phase = kernel.pipeline(phase_code);
        align = kernel.pipeline(align_code);
        noise = kernel.pipeline(soft_noise_code);
        frequency = kernel.pipeline(frequency_code);
        // Compile the three fixed constellation shapes before live capture.
        for (unsigned constellation = 0; constellation < 3; ++constellation) {
            const std::array bits{2 + 2 * constellation, 1U};
            equalize[constellation] = kernel.pipeline(equalize_code, std::array{bits[0], bits[1], 1U, 1U});
            demap[constellation] = kernel.pipeline(demap_code, std::array{bits[0], bits[1], 1U, 512U});
        }
    }
    void tables(const Tps &mode) {
        if (constellation != mode.constellation) {
            const auto indices = Modem{}.payload_bit_indices(mode);
            kernel.upload(10, std::span<const std::uint32_t>{indices});
            constellation = mode.constellation;
        }
        const int value = mode.guard;
        if (guard == value)
            return;
        if (value < 0 || value > 3)
            throw std::invalid_argument("Invalid frame equaliser guard interval");
        const auto pattern = Modem{}.pilot_pattern();
        std::vector<C> table(3 * 6817 + 192 + 49 + 177);
        // Shift the channel's causal GI support about zero before frequency
        // interpolation. This is a decoder coordinate, not a receive retiming
        // or an alteration of the measured carrier phase.
        const double centre = (8192 / (32U >> value)) * .5;
        for (unsigned k = 0; k < 6817; ++k) {
            const double angle = 2 * std::numbers::pi * (int(k) - 3408) * (fft_window_margin + centre) / 8192;
            table[k] = static_cast<C>(std::polar(1., angle)) / pattern.values[k];
            table[6817 + k] = static_cast<C>(std::polar(1., -angle));
        }
        // Eighth-order accuracy in carrier spacing: seven-degree Lagrange
        // interpolation on the dense, temporally reconstructed pilot grid.
        for (unsigned phase = 0; phase < 24; ++phase)
            for (unsigned j = 0; j < 8; ++j) {
                double weight = 1;
                for (unsigned l = 0; l < 8; ++l)
                    if (l != j)
                        weight *= (double(phase) / 3 - l) / (int(j) - int(l));
                table[2 * 6817 + phase * 8 + j] = {static_cast<float>(weight), 0};
            }
        if (pattern.continual.size() != 177)
            throw std::logic_error("Unexpected 8K continual-pilot table");
        for (unsigned i = 0; i < 177; ++i) {
            const auto k = pattern.continual[i];
            table[2 * 6817 + 192 + k] = {1, 0};
            table[3 * 6817 + 192 + 49 + i] = {std::bit_cast<float>(k), 0};
        }
        for (unsigned j = 0; j < 49; ++j)
            table[3 * 6817 + 192 + j] = {pilot_frequency_filters[value][j], 0};
        kernel.upload(7, std::span<const C>{table});
        guard = value;
    }
};
GpuFrameStorage::GpuFrameStorage(std::shared_ptr<gpu::Context> context)
    : impl_(std::make_unique<Impl>(std::move(context))) {}
GpuFrameStorage::~GpuFrameStorage() = default;
void GpuFrameStorage::wait() {
    impl_->kernel.wait();
}
FrameBatch GpuFrameStorage::capture(const FrameSelection &selection, ComputeTimings *timings,
                                    gpu::Commands *commands) {
    auto frame = selection.frame;
    if (frame.frames.empty() || selection.rows.size() < 8 || selection.rows.size() > 1020)
        throw std::invalid_argument("Incomplete frame equalisation context");
    if (!frame.symbols.empty() && frame.symbols.size() != selection.rows.size())
        throw std::invalid_argument("Incomplete source timing coordinates");
    impl_->tables(frame.frames.front());
    std::vector<float> offsets(selection.rows.size());
    for (std::size_t row = 0; row < frame.symbols.size(); ++row) {
        const auto offset = frame.symbols[row].source_offset_samples;
        if (!std::isfinite(offset))
            throw std::invalid_argument("Nonfinite source timing coordinate");
        offsets[row] = static_cast<float>(offset);
    }
    impl_->kernel.upload(8, std::span<const float>{offsets});
    impl_->kernel.storage(9, (selection.rows.size() * (3 * 2273 + 2) + 129) * sizeof(C));
    const auto bits = qam_bits(frame.frames.front());
    impl_->kernel.storage(1, selection.rows.size() * 6817 * sizeof(std::uint32_t));
    impl_->kernel.storage(11, selection.rows.size() * 6048 * bits);
    frame.carriers = copy_rows(impl_->kernel, frame.carriers, selection.rows, 0,
                               unsigned(selection.rows.size()), timings, commands,
                               {.equalize = *impl_->equalize.at((bits - 2) / 2),
                                .pilots = *impl_->pilots,
                                .demap = *impl_->demap.at((bits - 2) / 2),
                                .stats = *impl_->stats,
                                .noise = *impl_->noise,
                                .frequency = *impl_->frequency,
                                .phase = *impl_->phase,
                                .align = *impl_->align,
                                .risk = *impl_->risk,
                                .select = *impl_->select,
                                .continual = *impl_->continual,
                                .bits = bits});
    frame.carriers.coded_metric = CodedMetric::minimax_likelihood;
    return frame;
}
struct GpuFrames::Impl {
    struct Entry {
        std::uint32_t slot;
        TrackedSymbol symbol;
    };
    struct Channel {
        std::unique_ptr<gpu::Kernel> ring;
        OfdmBatch carriers;
        std::deque<Entry> pending, aligned;
        std::vector<Tps> frames;
        std::uint64_t written = 0;
        unsigned guard, segment = 0;
        bool history = false;
    };
    Modem tables;
    std::shared_ptr<gpu::Context> context;
    GpuFrameStorage staging;
    std::vector<Channel> channels;
    unsigned frames_per_batch, capacity;
    Impl(std::span<const unsigned> guards, std::shared_ptr<gpu::Context> context, unsigned batch)
        : context(std::move(context)), staging(this->context), frames_per_batch(batch),
          capacity((batch + 2) * 68) {
        if (guards.empty() || guards.size() > 64)
            throw std::invalid_argument("Invalid frame-buffer channel count");
        if (batch < 3 || batch > 15)
            throw std::invalid_argument("FEC batch requires 3..15 TPS frames");
        for (unsigned guard : guards) {
            if (guard != 4 && guard != 8 && guard != 16 && guard != 32)
                throw std::invalid_argument("Invalid frame-buffer guard interval");
            Channel c;
            c.guard = guard;
            c.ring = std::make_unique<gpu::Kernel>(copy_code, 16, std::array<std::uint32_t, 1>{8192},
                                                   this->context);
            channels.push_back(std::move(c));
        }
    }
    void emit(unsigned channel, bool final, const Select &consume) {
        auto &c = channels[channel];
        if (c.frames.empty())
            return;
        FrameSelection selection;
        auto &output = selection.frame;
        output.frames = c.frames;
        output.channel = channel;
        output.segment = c.segment;
        auto &rows = selection.rows;
        rows.reserve(c.aligned.size());
        output.symbols.reserve(c.aligned.size());
        for (const auto &entry : c.aligned) {
            rows.push_back(entry.slot);
            output.symbols.push_back(entry.symbol);
        }
        output.carriers = c.carriers;
        // Share one TPS frame, split equally across the emitted boundary.
        // 34 symbols exceed the 16-symbol maximum pilot radius and the
        // convolutional byte-interleaver span, while preserving TPS alignment.
        output.first = c.history ? 34 : 0;
        output.last = final ? rows.size() : rows.size() - 34;
        consume(selection);
        if (final) {
            c.aligned.clear();
            c.frames.clear();
            c.history = false;
        } else {
            c.aligned.erase(c.aligned.begin(), c.aligned.end() - 68);
            c.frames.erase(c.frames.begin(), c.frames.end() - 1);
            c.history = true;
        }
    }
    void accept(unsigned channel, Entry entry, const Select &consume) {
        auto &c = channels[channel];
        if (entry.symbol.discontinuity) {
            emit(channel, true, consume);
            c.pending.clear();
            ++c.segment;
        }
        c.pending.push_back(std::move(entry));
        if (c.pending.size() < 68)
            return;
        std::array<std::uint8_t, 68> bits{};
        bool indices = true;
        for (unsigned j = 0; j < 68; ++j) {
            indices = indices && c.pending[j].symbol.modulo4 == j % 4;
            if (j)
                bits[j] = (c.pending[j].symbol.tps * std::conj(c.pending[j - 1].symbol.tps)).real() < 0;
        }
        Tps frame;
        if (!indices || !tables.tps_bits(bits, frame)) {
            if (!c.frames.empty()) {
                emit(channel, true, consume);
                ++c.segment;
            }
            c.pending.pop_front();
            return;
        }
        const int guard = c.guard == 32 ? 0 : c.guard == 16 ? 1 : c.guard == 8 ? 2 : 3;
        if (frame.guard != guard)
            throw std::runtime_error("BCH-valid TPS disagrees with receiver guard interval");
        if (!c.frames.empty()) {
            const auto &last = c.frames.back();
            if (frame.frame != (last.frame + 1) % 4 || frame.constellation != last.constellation ||
                frame.rate != last.rate || frame.mode != last.mode || frame.hierarchy != last.hierarchy) {
                emit(channel, true, consume);
                ++c.segment;
            }
        }
        if (c.frames.empty())
            c.pending.front().symbol.discontinuity = true;
        for (auto &row : c.pending) {
            row.symbol.segment = c.segment;
            c.aligned.push_back(std::move(row));
        }
        c.pending.clear();
        c.frames.push_back(frame);
        if (c.frames.size() == frames_per_batch)
            emit(channel, false, consume);
    }
    void append(const ReceiverBatch &input, const Select &consume, ComputeTimings *timings,
                gpu::Commands *commands) {
        if (input.symbols.empty())
            return;
        if (input.symbols.size() != input.carriers.count)
            throw std::invalid_argument("Receiver carrier/metadata count mismatch");
        std::vector<std::vector<std::uint32_t>> selected(channels.size());
        for (unsigned row = 0; row < input.symbols.size(); ++row) {
            const auto &meta = input.symbols[row];
            if (meta.channel >= channels.size())
                throw std::invalid_argument("Receiver channel outside frame buffer");
            if (meta.ready)
                selected[meta.channel].push_back(row);
        }
        for (unsigned channel = 0; channel < channels.size(); ++channel) {
            auto &c = channels[channel];
            for (std::size_t at = 0; at < selected[channel].size(); at += 68) {
                const auto rows = std::span<const std::uint32_t>{selected[channel]}.subspan(
                    at, std::min<std::size_t>(68, selected[channel].size() - at));
                // At most 67 queued symbols can follow a complete batch. This
                // bounded append cannot overwrite its history before emission.
                c.carriers = copy_rows(*c.ring, input.carriers, rows, c.written % capacity, capacity, timings,
                                       commands);
                c.carriers.count = capacity;
                for (unsigned j = 0; j < rows.size(); ++j)
                    accept(channel,
                           {static_cast<unsigned>((c.written + j) % capacity), input.symbols[rows[j]]},
                           consume);
                c.written += rows.size();
            }
        }
    }
};
GpuFrames::GpuFrames(std::span<const unsigned> guards, std::shared_ptr<gpu::Context> context,
                     unsigned frames_per_batch)
    : impl_(std::make_unique<Impl>(guards, std::move(context), frames_per_batch)) {}
GpuFrames::~GpuFrames() = default;
void GpuFrames::wait() {
    for (auto &channel : impl_->channels)
        channel.ring->wait();
    impl_->staging.wait();
}
void GpuFrames::reset() {
    wait();
    for (auto &c : impl_->channels) {
        c.pending.clear();
        c.aligned.clear();
        c.frames.clear();
        c.written = 0;
        c.segment = 0;
        c.history = false;
    }
}
void GpuFrames::select(const ReceiverBatch &input, const Select &consume, ComputeTimings *timings,
                       gpu::Commands *commands) {
    impl_->append(input, consume, timings, commands);
}
void GpuFrames::finish_select(const Select &consume) {
    for (unsigned c = 0; c < impl_->channels.size(); ++c) {
        impl_->emit(c, true, consume);
        impl_->channels[c].pending.clear();
    }
}
void GpuFrames::append(const ReceiverBatch &input, const Consume &consume, ComputeTimings *timings) {
    select(
        input, [&](const FrameSelection &selected) { consume(impl_->staging.capture(selected, timings)); },
        timings);
}
void GpuFrames::finish(const Consume &consume, ComputeTimings *timings) {
    finish_select(
        [&](const FrameSelection &selected) { consume(impl_->staging.capture(selected, timings)); });
}
} // namespace radar
