#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace radar {
// Fractional-frequency Allan deviations at one second. These are explicit
// stochastic model parameters, not oscillator frequency-accuracy limits.
// AVAR(tau) = white_fm^2/tau + random_walk_fm^2*tau (SI seconds),
// excluding deterministic drift. Optional drift_diffusion is the diffusion
// of fractional-frequency drift (units s^-3), not an Allan deviation.
struct ClockNoise {
    double white_fm = 1e-11, random_walk_fm = 1e-11, drift_diffusion = 0;
};
// Optional coloured observation error, distinct from either oscillator. A
// stationary second-order process: b'' + 2*zeta*omega*b' + omega^2*b = w.
// sigma is its stationary timing deviation in seconds. This model represents
// path/estimation fluctuations, not an independently measured hardware clock.
// Empty spans retain the original estimator and allocate no extra states.
struct ClockPathNoise {
    double sigma_seconds{}, frequency_hz = 1, damping = 1;
};
struct ClockEstimate {
    double offset{}, velocity{}, acceleration{}; // ADC samples; seconds.
    double offset_variance{}, velocity_variance{}, acceleration_variance{};
};
struct ClockInnovation {
    double value{}, variance{}, normalized_square{}, log_likelihood{};
    bool gaussian_inlier = true; // Gaussian inlier; a false value has bounded influence.
    double robust_weight = 1;
};

// Linear-Gaussian clock ensemble, with a fixed arithmetic reference. Common
// phase/frequency/drift and N-1 differential transmitter phases/frequencies
// are estimated jointly. Their sums are zero by construction. Precision
// changes therefore cannot select a different clock epoch or frequency.
// The estimate is minimum mean-square error for this declared model. It
// does not certify the model for unknown propagation/oscillator impairments.
// A finite innovation bound enables Huberized corrections; while active, the
// covariance is nominal gain-conditioned bookkeeping, not a robust CRLB claim.
class ClockEstimator {
  public:
    ClockEstimator(std::size_t channels, double sample_rate, ClockNoise noise = {},
                   std::span<const ClockNoise> transmitter_noise = {},
                   std::span<const ClockPathNoise> path_noise = {})
        : channels_(channels), size_(dimension(channels) + (path_noise.empty() ? 0 : 2 * channels)),
          fs_(sample_rate), noise_(noise), paths_(path_noise.begin(), path_noise.end()), state_(size_),
          covariance_(size_ * size_), transition_(size_ * size_), work_(size_ * size_),
          temporary_(size_ * size_), phase_diffusion_(size_ * size_), frequency_diffusion_(size_ * size_),
          cross_diffusion_(size_ * size_), integrated_diffusion_(size_ * size_) {
        if (!std::isfinite(sample_rate) || sample_rate <= 0)
            throw std::invalid_argument("Invalid clock ensemble dimensions");
        for (double value : {noise.white_fm, noise.random_walk_fm, noise.drift_diffusion})
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("Invalid clock Allan model");
        if (!transmitter_noise.empty() && transmitter_noise.size() != channels)
            throw std::invalid_argument("Incomplete transmitter clock noise model");
        if (!paths_.empty() && paths_.size() != channels)
            throw std::invalid_argument("Incomplete path timing noise model");
        for (std::size_t c = 0; c < paths_.size(); ++c) {
            const auto p = paths_[c];
            if (!std::isfinite(p.sigma_seconds) || p.sigma_seconds < 0 || !std::isfinite(p.frequency_hz) ||
                p.frequency_hz <= 0 || !std::isfinite(p.damping) || p.damping <= 0)
                throw std::invalid_argument("Invalid path timing noise model");
            const auto at = dimension(channels_) + 2 * c;
            const double variance = std::pow(fs_ * p.sigma_seconds, 2),
                         omega = 2 * std::acos(-1.) * p.frequency_hz;
            covariance_[at * size_ + at] = variance;
            covariance_[(at + 1) * size_ + at + 1] = variance * omega * omega;
        }
        // Broad acquisition priors. The common absolute time origin is a
        // coordinate gauge; no UTC calibration is inferred from it.
        covariance_[0] = 1;
        covariance_[size_ + 1] = std::pow(fs_ * 1e-4, 2);
        covariance_[2 * size_ + 2] = std::pow(fs_ * 1e-6, 2);
        for (std::size_t c = 0; c + 1 < channels_; ++c)
            for (std::size_t j = 0; j + 1 < channels_; ++j) {
                const double contrast = (c == j ? 1. : 0.) - 1. / channels_;
                covariance_[(3 + c) * size_ + 3 + j] = 4096 * 4096. * contrast;
                const auto r = 3 + channels_ - 1;
                covariance_[(r + c) * size_ + r + j] = std::pow(fs_ * 1e-6, 2) * contrast;
            }
        // Transform independent transmitter-clock fluctuations into their
        // arithmetic mean and zero-sum contrasts, including cross covariance.
        // This preserves the reference even for unequal transmitter stability.
        for (std::size_t c = 0; c < transmitter_noise.size(); ++c) {
            const auto noise = transmitter_noise[c];
            if (!std::isfinite(noise.white_fm) || !std::isfinite(noise.random_walk_fm) ||
                noise.white_fm < 0 || noise.random_walk_fm < 0 || noise.drift_diffusion != 0)
                throw std::invalid_argument("Invalid differential transmitter clock model");
            std::vector<double> phase(size_), rate(size_);
            phase[0] = -1. / channels_;
            rate[1] = -1. / channels_;
            for (std::size_t j = 0; j + 1 < channels_; ++j) {
                phase[3 + j] = 1. / channels_ - (c == j ? 1. : 0.);
                rate[3 + channels_ - 1 + j] = phase[3 + j];
            }
            const double qp = std::pow(fs_ * noise.white_fm, 2),
                         qf = 3 * std::pow(fs_ * noise.random_walk_fm, 2);
            for (std::size_t i = 0; i < size_; ++i)
                for (std::size_t j = 0; j < size_; ++j) {
                    const auto at = i * size_ + j;
                    phase_diffusion_[at] += qp * phase[i] * phase[j];
                    frequency_diffusion_[at] += qf * rate[i] * rate[j];
                    cross_diffusion_[at] += qf * (phase[i] * rate[j] + rate[i] * phase[j]);
                    integrated_diffusion_[at] += qf * phase[i] * phase[j];
                }
        }
    }
    bool ready() const {
        return ready_;
    }
    double epoch() const {
        return epoch_;
    }
    std::span<const double> state() const {
        return state_;
    }
    std::span<const double> covariance() const {
        return covariance_;
    }
    void initialize(double epoch, double offset, double velocity = 0) {
        if (ready_ || !std::isfinite(epoch) || !std::isfinite(offset))
            throw std::invalid_argument("Invalid clock initialization");
        if (!std::isfinite(velocity))
            throw std::invalid_argument("Invalid initial clock rate");
        epoch_ = epoch;
        state_[0] = offset;
        state_[1] = velocity;
        ready_ = true;
    }
    ClockInnovation observe(std::size_t channel, double epoch, double offset, double variance,
                            double maximum_normalized_square = std::numeric_limits<double>::infinity()) {
        if (channel >= channels_ || !std::isfinite(offset) || !std::isfinite(variance) || variance <= 0)
            throw std::invalid_argument("Invalid clock measurement/covariance");
        if (!ready_)
            initialize(epoch, offset);
        predict(epoch);
        std::vector<double> h(size_), ph(size_), gain(size_);
        h[0] = 1;
        if (channel + 1 < channels_)
            h[3 + channel] = 1;
        else
            for (std::size_t c = 0; c + 1 < channels_; ++c)
                h[3 + c] = -1;
        if (!paths_.empty())
            h[dimension(channels_) + 2 * channel] = 1;
        double expected = 0, innovation_variance = variance;
        for (std::size_t i = 0; i < size_; ++i) {
            expected += h[i] * state_[i];
            for (std::size_t j = 0; j < size_; ++j)
                ph[i] += covariance_[i * size_ + j] * h[j];
            innovation_variance += h[i] * ph[i];
        }
        if (!(innovation_variance > 0) || !std::isfinite(innovation_variance))
            throw std::runtime_error("Invalid clock innovation covariance");
        const double innovation = offset - expected;
        const double normalized_square = innovation * innovation / innovation_variance;
        const double weight = normalized_square > maximum_normalized_square
                                  ? std::sqrt(maximum_normalized_square / normalized_square)
                                  : 1.;
        const ClockInnovation result{
            innovation,        innovation_variance,
            normalized_square, -.5 * (std::log(2 * std::acos(-1.) * innovation_variance) + normalized_square),
            weight == 1,       weight};
        // Huberize the standardized innovation. Hard deletion can permanently
        // lock a filter out after a mismodelled acquisition transient. A
        // bounded, nonzero influence allows recovery without epoch resets.
        // Joseph's update uses the actual bounded gain and original R.
        for (std::size_t i = 0; i < size_; ++i) {
            gain[i] = weight * ph[i] / innovation_variance;
            state_[i] += gain[i] * innovation;
        }
        // Joseph covariance update avoids cancellation when an acquisition
        // prior is many orders larger than a precise pilot observation.
        for (std::size_t i = 0; i < size_; ++i)
            for (std::size_t j = 0; j < size_; ++j)
                transition_[i * size_ + j] = (i == j ? 1. : 0.) - gain[i] * h[j];
        congruence(transition_);
        for (std::size_t i = 0; i < size_; ++i)
            for (std::size_t j = 0; j < size_; ++j)
                covariance_[i * size_ + j] += gain[i] * variance * gain[j];
        symmetrize();
        return result;
    }
    double channel_offset(std::size_t channel, double epoch) const {
        if (channel >= channels_)
            throw std::invalid_argument("Invalid clock channel");
        const auto common = estimate(epoch);
        const double t = epoch - epoch_;
        const auto rate = 3 + channels_ - 1;
        if (channel + 1 < channels_)
            return common.offset + state_[3 + channel] + t * state_[rate + channel];
        double value = common.offset;
        for (std::size_t c = 0; c + 1 < channels_; ++c)
            value -= state_[3 + c] + t * state_[rate + c];
        return value;
    }
    ClockEstimate estimate(double epoch) const {
        if (!ready_ || !std::isfinite(epoch) || epoch < epoch_)
            throw std::invalid_argument("Clock estimate precedes its observations");
        const double t = epoch - epoch_;
        const double h[]{1, t, .5 * t * t}, v[]{0, 1, t};
        auto quadratic = [&](const double *a) {
            double q = 0;
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned j = 0; j < 3; ++j)
                    q += a[i] * covariance_[i * size_ + j] * a[j];
            return q;
        };
        const double qx = std::pow(fs_ * noise_.white_fm, 2),
                     qv = 3 * std::pow(fs_ * noise_.random_walk_fm, 2),
                     qa = fs_ * fs_ * noise_.drift_diffusion;
        return {state_[0] + t * state_[1] + .5 * t * t * state_[2],
                state_[1] + t * state_[2],
                state_[2],
                quadratic(h) + (qx + phase_diffusion_[0]) * t +
                    (qv + integrated_diffusion_[0]) * t * t * t / 3 + qa * std::pow(t, 5) / 20,
                quadratic(v) + (qv + frequency_diffusion_[size_ + 1]) * t + qa * t * t * t / 3,
                covariance_[2 * size_ + 2] + qa * t};
    }

  private:
    static std::size_t dimension(std::size_t channels) {
        if (!channels || channels > 32)
            throw std::invalid_argument("Invalid clock ensemble dimensions");
        return 3 + 2 * (channels - 1);
    }
    void congruence(std::span<const double> f) {
        std::fill(work_.begin(), work_.end(), 0.);
        std::fill(temporary_.begin(), temporary_.end(), 0.);
        for (std::size_t i = 0; i < size_; ++i)
            for (std::size_t k = 0; k < size_; ++k) {
                const double a = f[i * size_ + k];
                for (std::size_t j = 0; j < size_; ++j)
                    work_[i * size_ + j] += a * covariance_[k * size_ + j];
            }
        for (std::size_t i = 0; i < size_; ++i)
            for (std::size_t j = 0; j < size_; ++j)
                for (std::size_t k = 0; k < size_; ++k)
                    temporary_[i * size_ + j] += work_[i * size_ + k] * f[j * size_ + k];
        covariance_.swap(temporary_);
    }
    void predict(double epoch) {
        if (!std::isfinite(epoch) || epoch < epoch_)
            throw std::invalid_argument("Out-of-order clock observation");
        const double t = epoch - epoch_;
        if (t == 0)
            return;
        state_[0] += t * state_[1] + .5 * t * t * state_[2];
        state_[1] += t * state_[2];
        std::fill(transition_.begin(), transition_.end(), 0.);
        for (std::size_t i = 0; i < size_; ++i)
            transition_[i * size_ + i] = 1;
        transition_[1] = t;
        transition_[2] = .5 * t * t;
        transition_[size_ + 2] = t;
        for (std::size_t c = 0; c + 1 < channels_; ++c) {
            const auto p = 3 + c, r = 3 + channels_ - 1 + c;
            state_[p] += t * state_[r];
            transition_[p * size_ + r] = t;
        }
        for (std::size_t c = 0; c < paths_.size(); ++c) {
            const auto p = paths_[c];
            const auto i = dimension(channels_) + 2 * c;
            const double omega = 2 * std::acos(-1.) * p.frequency_hz, a = p.damping * omega;
            double cosine, sine;
            if (p.damping < 1) {
                const double b = omega * std::sqrt(1 - p.damping * p.damping), decay = std::exp(-a * t);
                cosine = decay * std::cos(b * t);
                sine = decay * std::sin(b * t) / b;
            } else if (p.damping > 1) {
                const double b = omega * std::sqrt(p.damping * p.damping - 1);
                const double slow = std::exp(-omega / (p.damping + std::sqrt(p.damping * p.damping - 1)) * t),
                             fast = std::exp(-(a + b) * t);
                cosine = .5 * (slow + fast);
                sine = -.5 * slow * std::expm1(-2 * b * t) / b;
            } else {
                cosine = std::exp(-a * t);
                sine = t * cosine;
            }
            const double f00 = cosine + a * sine, f01 = sine, f10 = -omega * omega * sine,
                         f11 = cosine - a * sine;
            const double position = state_[i], rate = state_[i + 1];
            state_[i] = f00 * position + f01 * rate;
            state_[i + 1] = f10 * position + f11 * rate;
            transition_[i * size_ + i] = f00;
            transition_[i * size_ + i + 1] = f01;
            transition_[(i + 1) * size_ + i] = f10;
            transition_[(i + 1) * size_ + i + 1] = f11;
        }
        congruence(transition_);
        // Exact discrete diffusion from the stationary covariance: Q=P-FPF'.
        // Path states are independent in this optional model. A disturbance
        // common to every path is not identifiable as clock versus RF delay.
        for (std::size_t c = 0; c < paths_.size(); ++c) {
            const auto i = dimension(channels_) + 2 * c;
            const auto p = paths_[c];
            const double variance = std::pow(fs_ * p.sigma_seconds, 2),
                         omega = 2 * std::acos(-1.) * p.frequency_hz;
            const double f00 = transition_[i * size_ + i], f01 = transition_[i * size_ + i + 1];
            const double f10 = transition_[(i + 1) * size_ + i], f11 = transition_[(i + 1) * size_ + i + 1];
            const double a = p.damping * omega, q = 4 * a * omega * omega * variance;
            double position_noise, rate_noise;
            if (std::max(a, omega) * t < 1e-3) {
                // Avoid subtracting stationary variances to recover a tiny
                // O(dt^3) term. Integrated impulse-response series; omitted
                // relative terms are O(max(a,omega)^4*dt^4) in this branch.
                const double at = a * t, wt = omega * t;
                position_noise =
                    q * t * t * t *
                    (1. / 3 - at / 2 + (7 * at * at - wt * wt) / 15 + at * (wt * wt - 3 * at * at) / 9);
                rate_noise =
                    q * t *
                    (1 - 2 * at + (8 * at * at - wt * wt) / 3 + at * (5 * wt * wt - 16 * at * at) / 6);
            } else {
                position_noise = variance * (1 - f00 * f00 - omega * omega * f01 * f01);
                rate_noise = variance * (omega * omega - f10 * f10 - omega * omega * f11 * f11);
            }
            covariance_[i * size_ + i] += position_noise;
            covariance_[(i + 1) * size_ + i + 1] += rate_noise;
            const double cross = .5 * q * f01 * f01;
            covariance_[i * size_ + i + 1] += cross;
            covariance_[(i + 1) * size_ + i] += cross;
        }
        const double qx = std::pow(fs_ * noise_.white_fm, 2),
                     qv = 3 * std::pow(fs_ * noise_.random_walk_fm, 2),
                     qa = fs_ * fs_ * noise_.drift_diffusion;
        const double q[3][3] = {
            {qx * t + qv * t * t * t / 3 + qa * std::pow(t, 5) / 20, qv * t * t / 2 + qa * std::pow(t, 4) / 8,
             qa * t * t * t / 6},
            {qv * t * t / 2 + qa * std::pow(t, 4) / 8, qv * t + qa * t * t * t / 3, qa * t * t / 2},
            {qa * t * t * t / 6, qa * t * t / 2, qa * t}};
        for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j)
                covariance_[i * size_ + j] += q[i][j];
        for (std::size_t j = 0; j < covariance_.size(); ++j)
            covariance_[j] += (phase_diffusion_[j] + frequency_diffusion_[j]) * t +
                              cross_diffusion_[j] * t * t / 2 + integrated_diffusion_[j] * t * t * t / 3;
        epoch_ = epoch;
        symmetrize();
    }
    void symmetrize() {
        for (std::size_t i = 0; i < size_; ++i) {
            if (!std::isfinite(covariance_[i * size_ + i]) || covariance_[i * size_ + i] < 0)
                throw std::runtime_error("Invalid clock posterior covariance");
            for (std::size_t j = 0; j < i; ++j) {
                const double p = .5 * (covariance_[i * size_ + j] + covariance_[j * size_ + i]);
                covariance_[i * size_ + j] = covariance_[j * size_ + i] = p;
            }
        }
    }
    std::size_t channels_, size_;
    double fs_, epoch_ = 0;
    ClockNoise noise_;
    bool ready_ = false;
    std::vector<ClockPathNoise> paths_;
    std::vector<double> state_, covariance_, transition_, work_, temporary_;
    std::vector<double> phase_diffusion_, frequency_diffusion_, cross_diffusion_, integrated_diffusion_;
};
} // namespace radar
