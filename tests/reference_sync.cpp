#include "reference_sync.hpp"
#include "dvbt_ofdm_synchronization_impl.h"
#include <gnuradio/dl5eu/dvbt_ofdm_synchronization.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/sync_block.h>
#include <gnuradio/top_block.h>
namespace radar {
namespace {
constexpr std::size_t bins = 6817;
class Output : public gr::sync_block {
    std::function<void(Symbol)> receive_;

  public:
    std::exception_ptr failure;
    explicit Output(std::function<void(Symbol)> receive)
        : gr::sync_block("reconstruct_residual", gr::io_signature::make(2, 2, bins * sizeof(C)),
                         gr::io_signature::make(0, 0, 0)),
          receive_(std::move(receive)) {}
    int work(int n, gr_vector_const_void_star &input, gr_vector_void_star &) override {
        try {
            const auto *eq = static_cast<const C *>(input[0]);
            const auto *raw = static_cast<const C *>(input[1]);
            for (int i = 0; i < n; ++i) {
                Symbol s;
                s.equalized.assign(eq + i * bins, eq + (i + 1) * bins);
                s.measured.assign(raw + i * bins, raw + (i + 1) * bins);
                std::vector<gr::tag_t> tags;
                get_tags_in_range(tags, 0, nitems_read(0) + i, nitems_read(0) + i + 1);
                bool have_time = false, have_symbol = false;
                for (const auto &tag : tags) {
                    const auto key = pmt::symbol_to_string(tag.key);
                    if (key == "radar_timing") {
                        s.sample = pmt::to_double(pmt::tuple_ref(tag.value, 0));
                        s.sample_step = pmt::to_double(pmt::tuple_ref(tag.value, 1));
                        s.phase = pmt::to_double(pmt::tuple_ref(tag.value, 2));
                        s.frequency_hz = pmt::to_double(pmt::tuple_ref(tag.value, 3));
                        have_time = true;
                    } else if (key == "relative_symbol_index") {
                        s.modulo4 = static_cast<int>(pmt::to_long(tag.value));
                        have_symbol = true;
                    } else if (key == "resync" || key == "sync_start")
                        s.discontinuity = true;
                }
                if (!have_time || !have_symbol)
                    throw std::runtime_error("Synchronizer omitted sample coordinates");
                receive_(std::move(s));
            }
            return n;
        } catch (...) {
            failure = std::current_exception();
            return WORK_DONE;
        }
    }
};
class MemoryInput : public gr::sync_block {
    std::span<const C> input_;

  public:
    explicit MemoryInput(std::span<const C> input)
        : gr::sync_block("test_samples", gr::io_signature::make(0, 0, 0),
                         gr::io_signature::make(1, 1, sizeof(C))),
          input_(input) {}
    int work(int n, gr_vector_const_void_star &, gr_vector_void_star &out) override {
        if (input_.empty())
            return WORK_DONE;
        const auto count = std::min<std::size_t>(n, input_.size());
        std::copy_n(input_.begin(), count, static_cast<C *>(out[0]));
        input_ = input_.subspan(count);
        return static_cast<int>(count);
    }
};
} // namespace
void synchronize_samples(std::span<const C> input, int guard, const std::function<void(Symbol)> &receive,
                         std::shared_ptr<SampleClock> clock, std::uint64_t origin, unsigned decimation) {
    auto source = gnuradio::make_block_sptr<MemoryInput>(input);
    auto sink = gnuradio::make_block_sptr<Output>(receive);
    const auto gi = guard == 4    ? gr::dtv::GI_1_4
                    : guard == 8  ? gr::dtv::GI_1_8
                    : guard == 16 ? gr::dtv::GI_1_16
                                  : gr::dtv::GI_1_32;
    auto sync = gr::dl5eu::dvbt_ofdm_synchronization::make(gr::dtv::T8k, gi, 30, true);
    auto instrumented = std::static_pointer_cast<gr::dl5eu::dvbt_ofdm_synchronization_impl>(sync);
    if (clock)
        instrumented->radar_use_common_clock(clock, 0, origin, decimation);
    auto graph = gr::make_top_block("phase_coordinate_test");
    graph->connect(source, 0, sync, 0);
    graph->connect(sync, 0, sink, 0);
    graph->connect(sync, 1, sink, 1);
    sync->set_max_noutput_items(8);
    sink->set_max_noutput_items(8);
    graph->run(65536);
    if (sink->failure)
        std::rethrow_exception(sink->failure);
}
} // namespace radar
