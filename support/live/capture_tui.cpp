#include "capture_tui.hpp"
#include <ftxui/component/app.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/loop.hpp>
#include <mutex>
#include <thread>

namespace radar::live {
namespace {
class Session final : public CaptureUiBackend {
    ftxui::App &app_;
    mutable std::mutex mutex_;
    CaptureTuiView view_;
    CaptureUiHooks hooks_;
    std::atomic<bool> stop_ = false, exiting_ = false, done_ = false;
    void event(double seconds, std::string message, bool warning = false) {
        view_.events.push_back({seconds, std::move(message), warning});
        if (view_.events.size() > 128)
            view_.events.pop_front();
    }

  public:
    Session(ftxui::App &app, CaptureUiHooks hooks) : app_(app), hooks_(std::move(hooks)) {}
    bool display_available() const override {
        return bool(hooks_.open_channel);
    }
    void open_channel(std::size_t channel) override {
        try {
            if (hooks_.open_channel)
                hooks_.open_channel(channel);
        } catch (const std::exception &e) {
            display_error(e.what());
        }
    }
    void display_error(std::string message) {
        std::lock_guard lock(mutex_);
        event(view_.status.elapsed, "Display: " + message, true);
    }
    bool stop_requested() const override {
        return stop_;
    }
    void request_stop() override {
        stop_ = true;
        app_.PostEvent(ftxui::Event::Custom);
    }
    void quit() {
        exiting_ = true;
        request_stop();
        if (done_)
            app_.Exit();
    }
    CaptureTuiView view() const override {
        std::lock_guard lock(mutex_);
        auto value = view_;
        value.stop_requested = stop_;
        return value;
    }
    void publish(const CaptureStatus &s) override {
        {
            std::lock_guard lock(mutex_);
            const auto &old = view_.status;
            if (old.message != s.message || old.phase != s.phase)
                event(s.elapsed, std::string(capture_phase_name(s.phase)) + ": " + s.message,
                      s.phase == CapturePhase::failed);
            if (old.replay_loops != s.replay_loops)
                event(s.elapsed, "Replay loop " + std::to_string(s.replay_loops + 1) +
                                     ": new acquisition segment; no sample loss");
            if (old.missing != s.missing || old.host_discarded != s.host_discarded ||
                old.unknown_gaps != s.unknown_gaps || old.overflows != s.overflows ||
                old.sequence_errors != s.sequence_errors)
                event(s.elapsed,
                      "Capture loss: missing " + std::to_string(s.missing) + ", discarded " +
                          std::to_string(s.host_discarded) + ", unquantified " +
                          std::to_string(s.unknown_gaps) + ", overflows " + std::to_string(s.overflows),
                      true);
            view_.history.resize(s.channels.size());
            for (std::size_t c = 0; c < s.channels.size(); ++c) {
                const auto &channel = s.channels[c];
                auto &h = view_.history[c];
                if (c < old.channels.size()) {
                    const auto &previous = old.channels[c].receiver;
                    if (previous.ready != channel.receiver.ready)
                        event(s.elapsed,
                              channel.name + (channel.receiver.ready ? ": pilot timing tracking"
                                                                     : ": acquiring pilot timing"));
                    if (previous.segment != channel.receiver.segment) {
                        h.clear();
                        event(s.elapsed,
                              channel.name + ": timing segment " + std::to_string(channel.receiver.segment));
                    }
                }
                if (channel.receiver.ready && (h.empty() || h.back().seconds < s.processed_seconds)) {
                    h.push_back({s.processed_seconds, channel.receiver.residual_rate_ppm,
                                 channel.receiver.timing_samples * 125, channel.receiver.phase_rms_degrees,
                                 channel.receiver.frequency_hz});
                    if (h.size() > 512)
                        h.pop_front();
                }
            }
            view_.status = s;
        }
        app_.PostEvent(ftxui::Event::Custom);
    }
    void fail(std::string message) {
        auto s = view().status;
        s.phase = CapturePhase::failed;
        s.message = std::move(message);
        publish(s);
    }
    void complete(int code) {
        done_ = true;
        app_.PostEvent(ftxui::Event::Custom);
        if (exiting_ || (!stop_ && !code))
            app_.Exit();
    }
};
} // namespace
int run_capture_tui(const CaptureWork &work, CaptureUiHooks hooks) {
    auto app = ftxui::App::Fullscreen();
    app.ForceHandleCtrlC(false);
    Session session(app, hooks);
    auto component = capture_screen(session, [&] { session.quit(); });
    int result = 0;
    std::jthread worker([&] {
        try {
            result = work(session);
        } catch (const std::exception &e) {
            result = 1;
            session.fail(e.what());
        }
        session.complete(result);
    });
    {
        ftxui::Loop loop(&app, component);
        while (!loop.HasQuitted()) {
            loop.RunOnce();
            if (hooks.poll_display) {
                try {
                    hooks.poll_display();
                } catch (const std::exception &e) {
                    session.display_error(e.what());
                    hooks.poll_display = {};
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }
    session.request_stop();
    worker.join();
    return result;
}
} // namespace radar::live
