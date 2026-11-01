#pragma once
namespace radar {
// Shared by acquisition window selection and regenerated-reference phase compensation.
// Covers the centred channelizer (~74 output samples of lookahead) plus the
// fractional clock interpolator (48 samples), including the shortest 8K guard.
inline constexpr int fft_window_margin = 128;
} // namespace radar
