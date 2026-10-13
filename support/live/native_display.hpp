#pragma once
#include "live_capture.hpp"
#include <array>

namespace radar::live {
// Main-thread window owner. The factory's GPU projector exchanges only owned,
// bounded preview frames with it; no DSP device handles reach the renderer.
class NativeDisplay {
    struct Impl;
    std::shared_ptr<Impl> impl_;
public:
    struct Status {bool open=false;std::size_t channel{};std::uint64_t rendered_frames{},texture_uploads{};double map_time{};std::array<double,4> plot_pixels{};
        double integration{},effective_integration{},data_age{};};
    NativeDisplay();
    ~NativeDisplay();
    LiveDisplayFactory factory() const;
    void open(std::size_t channel);
    void close();
    void poll();
    Status status() const;
    // Diagnostic export of the next owned framebuffer; ordinary display never
    // performs a graphics readback. Call/poll on the window's main thread.
    void save_snapshot(const std::filesystem::path&);
};
}
