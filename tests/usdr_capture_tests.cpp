#include "capture_source.hpp"
#include <array>
#include <cstring>
#include <dm_dev.h>
#include <dm_stream.h>
#include <iostream>
#include <vector>
using namespace radar;
namespace {
void check(bool ok, const char *text) {
    if (!ok)
        throw std::runtime_error(text);
}
std::array<Iq16, 8> dma;
unsigned acquired, released, stops, destroyed, closed, syncs;
bool borrowed = false, release_failure = false, sync_failure = false;
int acquire_error = 0;
std::uint64_t timestamp = 100, lost = 0;
std::vector<std::string> events;
void reset() {
    acquired = released = stops = destroyed = closed = syncs = 0;
    borrowed = release_failure = sync_failure = false;
    acquire_error = 0;
    timestamp = 100;
    lost = 0;
    events.clear();
}
SdrCaptureSettings settings() {
    SdrCaptureSettings s;
    s.driver = "usdr";
    s.device_args = "bus=pcie";
    s.antenna = "EXT";
    s.sample_rate_hz = 64e6;
    s.centre_hz = 202e6;
    s.bandwidth_hz = 56e6;
    s.gain_db = 30;
    return s;
}
} // namespace
extern "C" {
int usdr_dmd_create_string(const char *, pdm_dev_t *out) {
    *out = reinterpret_cast<pdm_dev_t>(1);
    return 0;
}
int usdr_dmd_close(pdm_dev_t) {
    check(!borrowed, "device closed with a DMA loan");
    ++closed;
    return 0;
}
int usdr_dme_set_uint(pdm_dev_t, const char *, std::uint64_t) {
    return 0;
}
int usdr_dme_set_string(pdm_dev_t, const char *, const char *) {
    return 0;
}
int usdr_dme_get_string(pdm_dev_t, const char *, const char **out) {
    *out = "EXT";
    return 0;
}
int usdr_dme_get_uint(pdm_dev_t, const char *path, std::uint64_t *out) {
    const std::string_view p = path;
    if (p == "/dm/rate/master")
        *out = 64000001;
    else if (p == "/dm/rate/adc")
        *out = 64000001;
    else if (p.ends_with("frequency"))
        *out = 201999998;
    else if (p.ends_with("bandwidth"))
        *out = 0; // Analog LPF bypass, not 56 MHz.
    else if (p.ends_with("gain/auto"))
        *out = 29;
    else
        return -EINVAL;
    return 0;
}
int usdr_dms_create(pdm_dev_t, const char *, const char *format, logical_ch_msk_t channels, unsigned size,
                    pusdr_dms_t *out) {
    check(std::strcmp(format, "ci16") == 0 && channels == 1 && size == 16384, "Wrong native stream layout");
    *out = reinterpret_cast<pusdr_dms_t>(2);
    return 0;
}
int usdr_dms_destroy(pusdr_dms_t) {
    check(!borrowed, "destroyed with DMA loan");
    ++destroyed;
    return 0;
}
int usdr_dms_sync(pdm_dev_t, const char *, unsigned, pusdr_dms_t *) {
    ++syncs;
    return sync_failure && syncs == 2 ? -EIO : 0;
}
int usdr_dms_op(pusdr_dms_t, unsigned op, dm_time_t) {
    if (op == USDR_DMS_STOP) {
        check(!borrowed, "stop preceded release");
        ++stops;
        events.push_back("stop");
    }
    return 0;
}
int usdr_dms_acquire(pusdr_dms_t, unsigned timeout, usdr_dms_rx_block_t *b) {
    check(!borrowed && timeout == 250, "Invalid acquisition ownership/timeout");
    if (acquire_error)
        return acquire_error;
    ++acquired;
    borrowed = true;
    for (unsigned i = 0; i < dma.size(); ++i)
        dma[i] = {std::int16_t(100 + i), std::int16_t(-100 - int(i))};
    *b = {dma.data(),
          sizeof(dma),
          unsigned(dma.size()),
          USDR_DMS_NATIVE_CI16_LE,
          timestamp,
          lost,
          acquired,
          lost ? USDR_DMS_BLOCK_DISCONTINUITY : 0u};
    return 0;
}
int usdr_dms_release(pusdr_dms_t, const usdr_dms_rx_block_t *b) {
    check(borrowed && b->data == dma.data() && b->token == acquired, "Wrong DMA release");
    if (release_failure) {
        release_failure = false;
        return -EIO;
    }
    borrowed = false;
    ++released;
    events.push_back("release");
    for (auto &x : dma)
        x = {-1, -1}; // Returned pages may immediately be overwritten.
    return 0;
}
}
int main() {
    try {
        reset();
        {
            auto source = make_usdr_source(settings());
            const auto a = source->applied();
            check(a.driver == "usdr" && a.rate_hz == 64000001 && a.centre_hz == 201999998 &&
                      a.bandwidth_hz == 0 && a.gain_db == 29,
                  "Requested values substituted for readback");
            check(!source->time_anchor(), "Invented wall-clock anchor");
            source->start();
            std::array<Iq16, 3> small;
            std::array<Iq16, 16> big;
            auto x = source->receive(small);
            check(x.count == 3 && x.tick == 100 && acquired == 1 && released == 0,
                  "Partial loan consumed incorrectly");
            check(small[2].i == 102 && small[2].q == -102, "I/Q copy changed samples");
            x = source->receive(big);
            check(x.count == 5 && x.tick == 103 && acquired == 1 && released == 1,
                  "Remainder timestamp/release incorrect");
            check(big[0].i == 103 && big[4].q == -107, "Copy happened after DMA release");
            timestamp = 120;
            lost = 12;
            x = source->receive(big);
            check(x.count == 0 && x.error == CaptureRead::Error::overflow && borrowed,
                  "Loss event not before affected block");
            x = source->receive(big);
            check(x.count == 8 && x.tick == 120 && x.error == CaptureRead::Error::none && acquired == 2,
                  "Loss event repeated or block discarded");
            acquire_error = -ETIMEDOUT;
            x = source->receive(big);
            check(x.error == CaptureRead::Error::timeout && !borrowed, "Timeout acquired a loan");
            acquire_error = -EOVERFLOW;
            x = source->receive(big);
            check(x.error == CaptureRead::Error::fatal && !x.count, "Unknown ADC loss silently continued");
            source->stop();
            source->stop();
            check(stops == 1, "Stop was not idempotent");
        }
        check(closed == 1 && destroyed == 1, "Source leaked resources");
        reset();
        {
            auto source = make_usdr_source(settings());
            source->start();
            std::array<Iq16, 3> output;
            source->receive(output);
            source->stop();
            check(events == std::vector<std::string>{"release", "stop"},
                  "Partial loan not released before stop");
        }
        reset();
        {
            auto source = make_usdr_source(settings());
            source->start();
            std::array<Iq16, 8> output;
            release_failure = true;
            auto x = source->receive(output);
            check(x.error == CaptureRead::Error::fatal && !x.count && borrowed,
                  "Failed release lost ownership or published samples");
            source->stop();
            check(!borrowed && stops == 1 && released == 1, "Failed release could not be retried");
        }
        reset();
        {
            auto source = make_usdr_source(settings());
            sync_failure = true;
            bool failed = false;
            try {
                source->start();
            } catch (const std::exception &) {
                failed = true;
            }
            check(failed, "Injected start failure ignored");
        }
        check(stops == 1 && destroyed == 1 && closed == 1, "Failed start was not cleaned up");
        std::cout << "PASS uSDR partial loans, exact CI16, timestamps, gaps, errors, readbacks and cleanup\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
