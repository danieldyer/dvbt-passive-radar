#include "dsp.hpp"
#include "gpu_compute.hpp"
#include "gpu_focus.hpp"
#include "gpu_noise.hpp"
#include "matched_detector.hpp"
#include "matched_profiles.hpp"
#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <numbers>
#include <random>
#include <thread>

using namespace radar;
namespace {
constexpr unsigned bins = 6817, nfft = 8192;
constexpr double tau = 2 * std::numbers::pi;
void check(bool b, const char *message) {
    if (!b)
        throw std::runtime_error(message);
}
struct Plane {
    std::shared_ptr<gpu::Context> context;
    gpu::Buffer storage;
    gpu::BufferView view;
    Plane(std::shared_ptr<gpu::Context> c, std::size_t bytes)
        : context(std::move(c)), storage(gpu::make_buffer(context->gpu, bytes)),
          view{context, *storage.handle, bytes, context->gpu.device->mapMemory(*storage.memory, 0, bytes)} {}
    ~Plane() {
        context->gpu.device->unmapMemory(*storage.memory);
    }
    template <class T> void set(std::span<const T> values) {
        std::copy(values.begin(), values.end(), static_cast<T *>(view.mapped));
    }
};
void noise(std::shared_ptr<gpu::Context> context, float scale = 1, unsigned bits = 2,
           bool fixed_pilot = false, float tone = 4) {
    constexpr unsigned rows = 512;
    const float variance = .01F * scale * scale;
    std::mt19937 rng(9854);
    std::normal_distribution<float> normal(0, std::sqrt(variance / 2));
    std::vector<C> x(rows * bins), y(x.size()), h(bins, scale * C{.8F, .3F});
    std::vector<std::uint32_t> v(x.size(), 1);
    const auto points = qam_constellation(bits);
    for (unsigned i = 0; i < x.size(); ++i) {
        x[i] = points[rng() % points.size()];
        y[i] = h[i % bins] * x[i] + C{normal(rng), normal(rng)};
    }
    for (unsigned r = 0; r < rows; ++r) {
        const auto at = r * bins + 5840;
        if (fixed_pilot) {
            x[at] = C{4.F / 3, 0};
            y[at] = h[5840] * x[at] + C{normal(rng), normal(rng)};
        }
        y[at] += scale * std::polar(tone, fixed_pilot ? float(tau * 1.17 * r * .001088) : 0.F);
    }
    Plane xp(context, x.size() * sizeof(C)), yp(context, y.size() * sizeof(C)),
        hp(context, h.size() * sizeof(C)), vp(context, v.size() * sizeof(v[0]));
    xp.set<C>(x);
    yp.set<C>(y);
    hp.set<C>(h);
    vp.set<std::uint32_t>(v);
    FrameBatch frame;
    frame.first = 0;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    SpectralBatch batch{frame,
                        {},
                        {xp.view, vp.view, rows},
                        {hp.view, {}, rows, 0, 0},
                        {xp.view, {}, {}, hp.view, vp.view, rows, 0}};
    GpuNoise estimator(context);
    const auto result = estimator.estimate(batch);
    std::vector<float> measured(static_cast<const float *>(result.mapped),
                                static_cast<const float *>(result.mapped) + bins);
    auto ordered = measured;
    std::ranges::sort(ordered);
    check(std::abs(ordered[bins / 2] / variance - 1) < .04, "White carrier variance is biased");
    const float relative = tone / 4;
    check(measured[5840] > 8 * scale * scale * relative * relative &&
              measured[5840] < 24 * scale * scale * relative * relative,
          "Coherent narrowband interference was lost by noise differencing");
    std::cout << "Noise: median=" << ordered[bins / 2] << " coherent carrier=" << measured[5840]
              << " fixed_pilot=" << fixed_pilot << '\n';
}
void finite(std::shared_ptr<gpu::Context> context, unsigned guard = 16, bool pilots = false,
            bool drifting = false, unsigned rows = 259, unsigned profile_size = 1024, unsigned echo_delay = 0,
            double origin = 128.375) {
    const unsigned delay = echo_delay ? echo_delay : profile_size == 1024 ? 700 : 2668;
    const unsigned cp = 8192 / guard;
    constexpr double frequency = 200;
    const C channel{.8F, .3F}, amplitude = std::polar(.02F, .73F);
    std::mt19937 rng(63214);
    const auto points = qam_constellation(2);
    std::vector<C> s(rows * bins), aligned(s.size()), y(s.size()), h(bins, channel), useful(rows * nfft),
        previous(rows * nfft);
    std::vector<float> offsets(rows);
    std::vector<double> local(rows, origin);
    for (unsigned row = 0; row < rows; ++row) {
        offsets[row] = drifting ? std::remainder(.37 + .002 * row, 1.) : 0;
        local[row] -= offsets[row];
    }
    std::vector<std::uint32_t> v(s.size(), 1);
    std::vector<float> d(bins, 1);
    Fft fft(nfft);
    std::vector<C> work(nfft);
    for (unsigned r = 0; r < rows; ++r) {
        std::fill(work.begin(), work.end(), C{});
        for (unsigned k = 0; k < bins; ++k) {
            s[r * bins + k] = points[rng() % 4];
            if (pilots && k % 12 == r % 4 * 3)
                s[r * bins + k] = {rng() % 2 ? 4.F / 3 : -4.F / 3, 0};
            aligned[r * bins + k] =
                s[r * bins + k] * static_cast<C>(std::polar(1.0, -tau * (int(k) - 3408) * local[r] / nfft));
            work[(k + nfft - 3408) % nfft] =
                s[r * bins + k] *
                static_cast<C>(
                    std::polar(1.0, -tau * (int(k) - 3408) * (local[r] - std::floor(local[r])) / nfft));
        }
        fft.transform(work, true);
        std::copy(work.begin(), work.end(), useful.begin() + r * nfft);
    }
    for (unsigned r = 1; r < rows; ++r) {
        std::fill(work.begin(), work.end(), C{});
        for (unsigned k = 0; k < bins; ++k)
            work[(k + nfft - 3408) % nfft] =
                s[(r - 1) * bins + k] *
                static_cast<C>(
                    std::polar(1., -tau * (int(k) - 3408) * (local[r] - std::floor(local[r])) / nfft));
        fft.transform(work, true);
        std::copy(work.begin(), work.end(), previous.begin() + r * nfft);
    }
    auto source = [&](unsigned r, int coordinate) {
        if (coordinate < -int(cp))
            return previous[r * nfft + (coordinate + int(nfft + cp)) % nfft];
        return useful[r * nfft + (coordinate + int(nfft)) % nfft];
    };
    for (unsigned r = 1; r < rows; ++r) {
        for (unsigned j = 0; j < nfft; ++j)
            work[j] = amplitude * source(r, int(j) - int(std::floor(local[r])) - delay) *
                      static_cast<C>(std::polar(1.0, tau * frequency * (double(j) - 4095.5) / 8e6));
        fft.transform(work);
        for (unsigned k = 0; k < bins; ++k)
            y[r * bins + k] = channel * (aligned[r * bins + k] + work[(k + nfft - 3408) % nfft]);
    }
    std::copy_n(aligned.begin(), bins, y.begin());
    for (unsigned k = 0; k < bins; ++k)
        y[k] *= channel;
    Plane sp(context, s.size() * sizeof(C)), xp(context, s.size() * sizeof(C)),
        yp(context, s.size() * sizeof(C)), hp(context, h.size() * sizeof(C)),
        vp(context, v.size() * sizeof(v[0])), dp(context, d.size() * sizeof(float)),
        op(context, rows * sizeof(float));
    sp.set<C>(s);
    xp.set<C>(aligned);
    yp.set<C>(y);
    hp.set<C>(h);
    vp.set<std::uint32_t>(v);
    dp.set<float>(d);
    op.set<float>(offsets);
    FrameBatch frame;
    frame.first = 1;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    SpectralBatch batch{frame,
                        {},
                        {sp.view, vp.view, rows},
                        {hp.view, {}, rows, origin, 0, op.view},
                        {xp.view, {}, {}, hp.view, vp.view, rows, origin}};
    GpuMatchedProfiles processor(context, profile_size);
    ComputeTimings timing;
    const auto result = processor.calculate(batch, dp.view, guard, &timing);
    if (rows == 19 && guard == 16 && profile_size > 1024) {
        MatchedProfiles reused;
        processor.calculate_into(reused, batch, dp.view, guard, nullptr);
        check(reused.values == result.values, "Caller-owned output differs from owned profiles");
        const auto capacity = reused.values.capacity();
        frame.last = frame.first;
        processor.calculate_into(reused, batch, dp.view, guard, nullptr);
        check(reused.values.empty() && reused.count == 0 && reused.values.capacity() == capacity,
              "Empty batch retained data or discarded reusable capacity");
        frame.last = rows;
        processor.calculate_into(reused, batch, dp.view, guard, nullptr);
        check(reused.values == result.values, "Reused output has stale complex samples after an empty batch");
    }
    const double gain = std::norm(channel) * bins / nfft;
    double worst = 0, phase = 0;
    for (unsigned r = 1; r < rows; ++r) {
        std::fill(work.begin(), work.end(), C{});
        for (unsigned k = 0; k < bins; ++k)
            work[(k + nfft - 3408) % nfft] =
                (y[r * bins + k] - channel * aligned[r * bins + k]) * std::conj(channel);
        fft.transform(work, true);
        std::vector<int> lags{-64, -32, 0, 16, 128, 384, 600, 700, 800, 928};
        if (profile_size > 1024)
            lags.insert(lags.end(),
                        {1024, 2000, 2668, 2669, 2734, int(matched_context_for_size(profile_size))});
        for (int lag : lags)
            for (unsigned moment = 0; moment < 2; ++moment) {
                std::complex<double> expected{};
                for (unsigned j = 0; j < nfft; ++j)
                    expected +=
                        std::complex<double>(work[j]) *
                        std::conj(std::complex<double>(source(r, int(j) - int(std::floor(local[r])) - lag))) *
                        (moment ? (double(j) - 4095.5) / nfft : 1);
                expected /= gain;
                const auto index = lag < 0 ? profile_size + lag : lag;
                const auto actual = result.values[(moment * result.count + r - 1) * profile_size + index];
                worst = std::max(worst, std::abs(std::complex<double>(actual) - expected));
                if (lag == delay && moment == 0)
                    phase = std::max(phase, std::abs(std::arg(std::complex<double>(actual) / expected)) *
                                                180 / std::numbers::pi);
                check(worst < (drifting ? 2e-5 : 2e-6),
                      "Finite correlation, fractional reference phase or Doppler moment wrong");
            }
    }
    check(phase < .37, "Source-clock bookkeeping exceeded the phase budget");
    std::cout << "Finite CP context, source drift=" << drifting << ", maximum complex error=" << worst
              << ", phase error=" << phase << " degrees; GPU " << timing.gpu * 1000 << " ms\n";
}
void focus(std::shared_ptr<gpu::Context> context, double rf = 557625000, bool full_cpi = false,
           bool remove_dc = false, bool weighted = false, bool erasures = false, unsigned profile_size = 1024,
           bool odd_rows = false) {
    const unsigned rows = (full_cpi ? 2758 : 512) - unsigned(odd_rows), size = full_cpi ? 11520 : 2048,
                   tone = full_cpi ? 2305 : 401;
    constexpr double dt = .001088;
    const double delay = profile_size == 1024 ? 167.375 : 2667.375;
    const double fd = tone / (size * dt), mid = (rows - 1) * dt * .5, x = fd * .001024;
    const double sinc = std::sin(std::numbers::pi * x) / (std::numbers::pi * x);
    const double derivative =
        (std::numbers::pi * x * std::cos(std::numbers::pi * x) - std::sin(std::numbers::pi * x)) /
        (std::numbers::pi * x * x);
    const C amplitude = std::polar(.17F, 1.23F);
    std::vector<double> information(rows, 1);
    double total = 0;
    if (weighted)
        for (unsigned n = 0; n < rows; ++n)
            information[n] = n < rows / 3 ? 1 : n < 2 * rows / 3 ? .1 : 10;
    if (erasures)
        for (unsigned n = 0; n < rows; ++n)
            if (n % 43 == 0 || (n >= rows / 2 && n < rows / 2 + 16))
                information[n] = 0;
    for (double value : information)
        total += value;
    std::vector<double> weights(rows);
    for (unsigned n = 0; n < rows; ++n)
        weights[n] = information[n] / total * rows;
    const int band = 400 * profile_size / 1024;
    const unsigned carriers = 2 * band + 1;
    std::vector<C> moments(2 * rows * profile_size), work(profile_size);
    Fft fft(profile_size);
    for (unsigned r = 0; r < rows; ++r) {
        const double t = r * dt - mid, d = delay - fd * 8e6 / rf * t;
        std::fill(work.begin(), work.end(), C{});
        for (int k = -band; k <= band; ++k)
            work[(k + int(profile_size)) % int(profile_size)] =
                amplitude * static_cast<C>(std::polar(double(profile_size) / carriers,
                                                      tau * fd * t - tau * k * d / profile_size));
        fft.transform(work, true);
        for (unsigned k = 0; k < profile_size; ++k) {
            moments[r * profile_size + k] = work[k] * float(sinc);
            moments[(rows + r) * profile_size + k] = work[k] * C{0, float(-derivative / tau)};
            if (remove_dc) {
                const float static_shape = std::exp(-std::pow((float(k) - 20) / 10, 2));
                moments[r * profile_size + k] += C{2, -3} * static_shape;
                moments[(rows + r) * profile_size + k] += C{.2F, .3F} * static_shape;
            }
        }
    }
    GpuFocus processor(context);
    ComputeTimings timing;
    auto result =
        processor.calculate(moments, rows, dt, rf, 5,
                            {profile_size == 1024 ? 8000.0 : 100000.0, 200, remove_dc}, &timing, information);
    check(result.ranges == std::size_t((profile_size == 1024 ? 8000.0 : 100000.0) / result.range_step) + 1,
          "Wrong map range extent");
    const int half = int((result.frequencies - 1) / 2);
    double worst = 0;
    const unsigned range_offset = profile_size == 1024 ? 0 : 10000;
    for (int bin : {-400, -37, int(tone), int(tone + 29)})
        for (unsigned range :
             {660 + range_offset, 669 + range_offset, 670 + range_offset, 674 + range_offset}) {
            std::complex<double> expected{};
            const double f = bin / (size * dt), rr = range * .25;
            for (int k = -band; k <= band; ++k) {
                std::complex<double> temporal{}, source_mean{}, template_sum{};
                for (unsigned n = 0; n < rows; ++n) {
                    const double t = n * dt - mid;
                    temporal +=
                        weights[n] *
                        std::polar(1.0, tau * ((fd - f) + k * 8e6 / (profile_size * rf) * (fd - f)) * t);
                    if (remove_dc) {
                        const double scale = 1 + k * 8e6 / (profile_size * rf);
                        source_mean += weights[n] * std::polar(1.0, tau * fd * scale * t);
                        template_sum += weights[n] * std::polar(1.0, -tau * f * scale * t);
                    }
                }
                if (remove_dc)
                    temporal -= source_mean * template_sum / double(rows);
                expected += temporal * std::polar(1.0, tau * k * (rr - delay) / profile_size);
            }
            expected *= std::complex<double>(amplitude) / double(rows * carriers);
            const double fx = f * .001024, a = std::numbers::pi * fx;
            const double gain = std::abs(a) < 1e-8 ? 1 : (2 * std::sin(a) - a * std::cos(a)) / a;
            expected *= (sinc - f * .001024 * derivative) / gain;
            const auto actual = result.values[range * result.frequencies + bin + half];
            worst = std::max(worst, std::abs(std::complex<double>(actual) - expected));
        }
    std::cout << "Constant-velocity focus complex error=" << worst << " weighted=" << weighted
              << " GPU=" << timing.gpu * 1000 << " ms\n";
    check(worst < 8e-5, "Range migration interpolation or centred phase epoch is wrong");
    check(std::abs(result.phase_epoch - (5 + mid + 4095.5 / 8e6)) < 1e-12,
          "Focus epoch omits within-symbol timing");
    check(result.amplitude_units == "complex_reflectivity" && result.noise_range.empty() &&
              result.noise_doppler.empty(),
          "Focus applied an empirical map normalization");
}
void noise_steps(std::shared_ptr<gpu::Context> context) {
    constexpr unsigned rows = 512;
    std::mt19937 random(7854);
    const auto points = qam_constellation(2);
    std::vector<C> source(rows * bins), measured(source.size()), response(bins, C{.8F, .3F});
    std::vector<std::uint32_t> valid(source.size(), 1);
    for (auto &value : source)
        value = points[random() % 4];
    Plane sp(context, source.size() * sizeof(C)), yp(context, source.size() * sizeof(C)),
        hp(context, response.size() * sizeof(C)), vp(context, valid.size() * sizeof(valid[0]));
    sp.set<C>(source);
    hp.set<C>(response);
    vp.set<std::uint32_t>(valid);
    FrameBatch frame;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    SpectralBatch batch{frame,
                        {},
                        {sp.view, vp.view, rows},
                        {hp.view, {}, rows, 0, 0},
                        {sp.view, {}, {}, hp.view, vp.view, rows, 0}};
    GpuNoise estimator(context);
    float previous = 0;
    for (float variance : {.01F, .1F, .001F}) {
        std::normal_distribution<float> normal(0, std::sqrt(variance / 2));
        for (unsigned i = 0; i < source.size(); ++i)
            measured[i] = response[i % bins] * source[i] + C{normal(random), normal(random)};
        for (unsigned r = 0; r < rows; ++r)
            measured[r * bins + 5840] += C{4, 0};
        yp.set<C>(measured);
        auto result = estimator.estimate(batch);
        const auto *values = static_cast<const float *>(result.mapped);
        std::vector<float> ordered(values, values + bins);
        std::ranges::sort(ordered);
        if (previous)
            check(std::abs(ordered[bins / 2] / previous - 1) < .12, "Primary noise changed its own weights");
        result = estimator.estimate(batch);
        values = static_cast<const float *>(result.mapped);
        ordered.assign(values, values + bins);
        std::ranges::sort(ordered);
        previous = variance;
        check(std::abs(ordered[bins / 2] / variance - 1) < .12,
              "Carrier weights failed to track a common noise-level step");
        check(values[5840] > 8, "Noise-level adaptation underestimated a stable narrow interferer");
        std::cout << "Noise step true/median=" << variance << " / " << ordered[bins / 2] << '\n';
    }
}
void covariance_efficiency(std::shared_ptr<gpu::Context> context) {
    constexpr unsigned rows = 512;
    std::mt19937 random(13657);
    std::normal_distribution<float> normal(0, 1);
    const auto points = qam_constellation(6);
    std::vector<C> source(rows * bins), measured(source.size()), response(bins);
    std::vector<float> truth(bins);
    std::vector<std::uint32_t> valid(source.size(), 1);
    for (auto &z : source)
        z = points[random() % points.size()];
    for (unsigned k = 0; k < bins; ++k)
        response[k] = static_cast<C>(std::exp(.3 * std::polar(1.0, -tau * (int(k) - 3408) / 8192)));
    Plane sp(context, source.size() * sizeof(C)), yp(context, source.size() * sizeof(C)),
        hp(context, response.size() * sizeof(C)), vp(context, valid.size() * sizeof(valid[0]));
    sp.set<C>(source);
    hp.set<C>(response);
    vp.set<std::uint32_t>(valid);
    FrameBatch frame;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    SpectralBatch batch{frame,
                        {},
                        {sp.view, vp.view, rows},
                        {hp.view, {}, rows, 0, 0},
                        {sp.view, {}, {}, hp.view, vp.view, rows, 0}};
    for (unsigned model = 0; model < 3; ++model) {
        for (unsigned k = 0; k < bins; ++k)
            truth[k] =
                model == 1 ? float(.0001 * std::pow(10., 2 * std::pow((int(k) - 3408) / 3408., 2))) : .0001F;
        if (model == 2)
            truth[5840] = 10000;
        for (unsigned i = 0; i < source.size(); ++i) {
            measured[i] = source[i] * response[i % bins] +
                          std::sqrt(truth[i % bins] / 2) * C{normal(random), normal(random)};
        }
        yp.set<C>(measured);
        GpuNoise estimator(context);
        const auto trained = estimator.estimate(batch);
        std::vector<float> prediction(static_cast<const float *>(trained.mapped),
                                      static_cast<const float *>(trained.mapped) + bins);
        // A new, strong moving echo must not determine its own carrier weights.
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned k = 0; k < bins; ++k)
                measured[r * bins + k] +=
                    source[r * bins + k] *
                    static_cast<C>(std::polar(.7, tau * (37 * r * .001088 - (int(k) - 3408) * 180. / 8192)));
        yp.set<C>(measured);
        const auto primary = estimator.estimate(batch);
        check(std::equal(prediction.begin(), prediction.end(), static_cast<const float *>(primary.mapped)),
              "Primary target altered its own covariance weights");
        double cross = 0, variance = 0, optimum = 0;
        for (unsigned k = 0; k < bins; ++k) {
            const double energy = std::norm(response[k]), d = prediction[k];
            cross += energy / d;
            variance += energy * truth[k] / (d * d);
            optimum += energy / truth[k];
        }
        const double loss = -10 * std::log10(cross * cross / (variance * optimum));
        std::cout << "Independent diagonal covariance model " << model << ": matched-filter SNR loss=" << loss
                  << " dB\n";
        check(loss < .03,
              "Covariance estimation left significant SNR on the table in the stated Gaussian model");
    }
}
void owned_focus(std::shared_ptr<gpu::Context> context) {
    constexpr unsigned rows = 512, delay = 100;
    std::mt19937 random(46059);
    std::normal_distribution<float> normal(0, .001F);
    const auto points = qam_constellation(2);
    std::vector<C> source(rows * bins), thermal(source.size()), measured(source.size()),
        response(bins, C{1, 0});
    std::vector<std::uint32_t> valid(source.size(), 1);
    for (unsigned i = 0; i < source.size(); ++i) {
        source[i] = points[random() % 4];
        thermal[i] = {normal(random), normal(random)};
    }
    Plane sp(context, source.size() * sizeof(C)), yp(context, source.size() * sizeof(C)),
        hp(context, response.size() * sizeof(C)), vp(context, valid.size() * sizeof(valid[0]));
    FrameBatch frame;
    frame.first = 1;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    frame.carriers.count = rows;
    frame.symbols.resize(rows);
    SpectralBatch batch{frame,
                        {},
                        {sp.view, vp.view, rows},
                        {hp.view, {}, rows, 0, 0},
                        {sp.view, {}, {}, hp.view, vp.view, rows, 0}};
    SarCapture capture{56'000'000, 7, 0, 553000000, 3, {{557625000, 4625000, 16}, {564625000, 11625000, 16}}};
    auto fill = [&](unsigned channel) {
        frame.channel = channel;
        for (unsigned r = 0; r < rows; ++r) {
            frame.symbols[r].channel = channel;
            frame.symbols[r].segment = frame.segment;
            frame.symbols[r].window.sample = r * 8704;
        }
        for (unsigned i = 0; i < source.size(); ++i) {
            const auto echo =
                std::polar(.02F, float(.73 + 1.12 * channel - tau * (int(i % bins) - 3408) * delay / nfft));
            measured[i] = source[i] * (C{1, 0} + echo) + thermal[i];
        }
        sp.set<C>(source);
        yp.set<C>(measured);
        hp.set<C>(response);
        vp.set<std::uint32_t>(valid);
    };
    auto poison = [&] {
        std::fill_n(static_cast<C *>(sp.view.mapped), source.size(), C{});
        std::fill_n(static_cast<C *>(yp.view.mapped), source.size(), C{});
        std::fill_n(static_cast<C *>(hp.view.mapped), response.size(), C{});
        std::fill_n(static_cast<std::uint32_t *>(vp.view.mapped), valid.size(), 0U);
    };
    DetectionOptions options;
    options.integration_seconds = .5;
    options.focus = {8000, 200, false};
    MatchedDetector detector(context, 2, options);
    DetectionTimings timing;
    const auto owner = std::this_thread::get_id();
    std::vector<DopplerMap> maps(2);
    unsigned count = 0;
    auto consume = [&](std::size_t channel, DopplerMap &&map) {
        check(std::this_thread::get_id() == owner, "Async focus invoked a consumer on its worker thread");
        check(map.sequence == 1 && map.channel == channel, "Async focus reordered map metadata");
        maps[channel] = std::move(map);
        ++count;
    };
    auto append = [&](MatchedDetector &target, unsigned channel, const MatchedDetector::Consumer &sink,
                      DetectionTimings *t = nullptr) {
        fill(channel);
        target.append(SarBatch(capture, batch, {{0, 0, 1}}), sink, t);
        poison();
    };
    // Live scientific profiles precede stationary suppression, from the very
    // first decoded batch. Display DC settings must not alter their samples,
    // phase, weights, or admission times (including stationary target energy).
    std::vector<C> preserved;
    double preserved_information = 0;
    unsigned preserved_count = 0;
    for (bool remove_dc : {false, true}) {
        DetectionOptions live_options = options;
        live_options.publish_maps = false;
        live_options.focus.remove_dc = remove_dc;
        unsigned calls = 0;
        MatchedDetector scientific(
            context, 2, live_options, {}, [&](const auto &, const auto &profiles, auto) {
                ++calls;
                check(profiles.count == rows - 1, "First-batch scientific profile rows were omitted");
                if (!remove_dc) {
                    preserved = profiles.values;
                    preserved_information = profiles.information;
                    preserved_count = profiles.count;
                } else
                    check(profiles.values == preserved && profiles.information == preserved_information &&
                              profiles.count == preserved_count,
                          "Display clutter suppression changed archived complex profiles");
                check(std::abs(profiles.values[delay]) > .01,
                      "Stationary target was removed before archival");
            });
        fill(0);
        scientific.append(SarBatch(capture, batch, {{0, 0, 1}}), [](auto, auto &&) {});
        check(calls == 1, "Live archive discarded the first noise-training batch");
    }
    for (unsigned channel = 0; channel < 2; ++channel)
        append(detector, channel, consume, &timing);
    detector.finish(consume, &timing);
    detector.finish(consume, &timing);
    check(count == 2 && timing.focus.batches == 2, "Async finish lost, duplicated or mistimed a map");
    const auto zero = maps[0].frequencies / 2;
    const auto a = maps[0].values[4 * delay * maps[0].frequencies + zero],
               b = maps[1].values[4 * delay * maps[1].frequencies + zero];
    check(std::abs(a) > .019 && std::abs(b) > .019,
          "Async focus retained borrowed buffers instead of owned profiles");
    check(std::abs(maps[0].phase_epoch - maps[1].phase_epoch) < 1e-12,
          "Async channels acquired different phase epochs");
    check(std::abs(std::arg(b * std::conj(a)) - 1.12) < .003, "Async scheduling changed interchannel phase");
    ++frame.segment;
    append(detector, 0, consume);
    bool failed = false;
    try {
        detector.finish([](std::size_t, DopplerMap &&) { throw std::runtime_error("sink failed"); });
    } catch (const std::runtime_error &error) {
        failed = std::string_view(error.what()) == "sink failed";
    }
    check(failed, "Async map consumer failure was swallowed");
    capture.channels[0].rf_hz = 50e6;
    ++frame.segment;
    append(detector, 0, consume);
    failed = false;
    try {
        detector.finish(consume);
    } catch (const std::invalid_argument &) {
        failed = true;
    }
    check(failed, "Async numerical failure was swallowed");
    capture.channels[0].rf_hz = 557625000;
    for (const auto &map : maps)
        check(map.amplitude_units == "complex_reflectivity" && map.noise_range.empty() &&
                  map.noise_doppler.empty(),
              "Async detector applied an empirical map normalization");
    // Exercise admission through the production detector, not just GpuFocus.
    // One failed oracle row used to flush the CPI despite continuous ADC data.
    std::fill_n(valid.begin() + 256 * bins, bins, 0U);
    ++frame.segment;
    MatchedDetector erased(context, 2, options);
    unsigned erased_count = 0;
    auto erased_sink = [&](std::size_t, DopplerMap &&map) {
        check(map.samples == maps[0].samples && map.phase_epoch == maps[0].phase_epoch,
              "Oracle erasure changed the coherent interval or phase epoch");
        const auto actual = map.values[4 * delay * map.frequencies + map.frequencies / 2];
        check(std::abs(actual - a) < 4e-5 && map.enbw > maps[0].enbw,
              "Erased oracle row was invented or changed the known target phase");
        ++erased_count;
    };
    append(erased, 0, erased_sink);
    erased.finish(erased_sink);
    check(erased_count == 1, "A decoder erasure discarded an otherwise complete coherent interval");
    // A stalled optional projector must not hold up scientific profiles.
    // The watchdog releases it even on failure so this regression cannot hang.
    std::fill(valid.begin(), valid.end(), 1U);
    ++frame.segment;
    options.best_effort_maps = true;
    options.update_seconds = .001;
    options.focus.range_metres = 100000;
    std::promise<void> started, returned, release;
    auto entered = started.get_future(), completed = returned.get_future();
    auto resume = release.get_future().share();
    bool blocked = false;
    unsigned retained = 0;
    MatchedDetector live(
        context, 2, options,
        [&](auto &, const auto &) {
            started.set_value();
            resume.wait();
        },
        [&](const auto &, const auto &profiles, auto) {
            check(profiles.size == matched_size_for_range(100000),
                  "Live detector truncated the saved range history");
            ++retained;
        });
    std::jthread watchdog([&] {
        entered.wait();
        blocked = completed.wait_for(std::chrono::seconds(5)) != std::future_status::ready;
        release.set_value();
    });
    append(live, 0, [](std::size_t, DopplerMap &&) {});
    returned.set_value();
    watchdog.join();
    unsigned live_maps = 0;
    live.finish([&](std::size_t, DopplerMap &&map) {
        ++live_maps;
        check(map.ranges * map.range_step >= 100000, "Live detector truncated its map extent");
        check(map.samples == maps[0].samples, "Busy display shortened its coherent aperture");
    });
    check(!blocked && retained == 1 && live_maps == 1,
          "Busy live display blocked capture, lost profiles or queued stale maps");
    std::cout << "Owned async profiles, interchannel phase, physical units, caller-thread delivery, drain "
                 "and failures: PASS\n";
}
void compact_validation(std::shared_ptr<gpu::Context> context) {
    for (unsigned guard : {4U, 8U, 16U, 32U}) {
        finite(context, guard, true, true, 19, 3072);
        finite(context, guard, true, false, 19, 3072, 2976, 256);
    }
    finite(context, 16, true, true, 259, 3072);
    finite(context, 32, true, true, 19, 3072, 3300);
    finite(context, 32, true, true, 19, 3072, 4300);
    focus(context, 177500000, true, true, true, true, 3072);
}
void profile_benchmark(std::shared_ptr<gpu::Context> context, bool compact = false) {
    constexpr unsigned rows = 273;
    std::vector<C> x(rows * bins), y(x.size()), h(bins, C{.8F, .3F});
    std::vector<std::uint32_t> valid(x.size(), 1);
    std::vector<float> variance(bins, 1), offsets(rows);
    std::mt19937 rng(76252);
    std::normal_distribution<float> noise(0, .01F);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = {rng() % 2 ? 1.F : -1.F, rng() % 2 ? 1.F : -1.F};
        y[i] = h[i % bins] * x[i] + C{noise(rng), noise(rng)};
    }
    for (unsigned r = 0; r < rows; ++r)
        offsets[r] = std::remainder(.37 + .002 * r, 1.);
    Plane xp(context, x.size() * sizeof(C)), yp(context, y.size() * sizeof(C)),
        hp(context, h.size() * sizeof(C)), vp(context, valid.size() * 4), dp(context, variance.size() * 4),
        op(context, offsets.size() * 4);
    xp.set<C>(x);
    yp.set<C>(y);
    hp.set<C>(h);
    vp.set<std::uint32_t>(valid);
    dp.set<float>(variance);
    op.set<float>(offsets);
    FrameBatch frame;
    frame.first = 1;
    frame.last = rows;
    frame.carriers.measured = yp.view;
    SpectralBatch batch{frame, {}, {xp.view, vp.view, rows}, {hp.view, {}, rows, 128.375, 0, op.view}, {}};
    const unsigned size = compact ? 3072 : 4096;
    GpuMatchedProfiles tiled(context, size, 64), previous(context, 4096, 128);
    const auto a = tiled.calculate(batch, dp.view, 32), b = previous.calculate(batch, dp.view, 32);
    double error = 0, energy = 0;
    for (unsigned m = 0; m < 2; ++m)
        for (unsigned r = 0; r < rows - 1; ++r)
            for (int lag = -64; lag <= int(matched_context_for_size(size)); ++lag) {
                auto u = a.values[(m * (rows - 1) + r) * size + (lag < 0 ? lag + size : lag)];
                auto v = b.values[(m * (rows - 1) + r) * 4096 + (lag < 0 ? lag + 4096 : lag)];
                error += std::norm(u - v);
                energy += std::norm(v);
            }
    std::cout << "Common-lag relative complex RMS=" << std::sqrt(error / energy) << '\n';
    check(error < 1e-10 * energy, "FFT tile changed the finite correlation");
    std::vector<std::unique_ptr<GpuMatchedProfiles>> processors;
    for (unsigned tile : {16U, 32U, 64U, 128U}) {
        processors.push_back(std::make_unique<GpuMatchedProfiles>(context, size, tile));
        processors.back()->calculate(batch, dp.view, 32);
    }
    // Profile the same operations on one tile: prepare/IFFT, finite-prefix
    // correction, time moment/FFT, products/IFFT, extraction.
    frame.last = 65;
    for (unsigned guard : {4U, 8U, 16U, 32U}) {
        ComputeTimings timing;
        timing.detailed = true;
        MatchedProfiles output;
        processors[2]->calculate_into(output, batch, dp.view, guard);
        for (unsigned n = 0; n < 25; ++n)
            processors[2]->calculate_into(output, batch, dp.view, guard, &timing);
        std::cout << "PROFILE_STAGES guard=" << guard << " gpu_ms=" << timing.gpu * 1000 / 25;
        for (auto stage : timing.stages)
            std::cout << ' ' << stage * 1000 / 25;
        ComputeTimings control;
        for (unsigned n = 0; n < 25; ++n)
            processors[2]->calculate_into(output, batch, dp.view, guard, &control);
        std::cout << " control_gpu_ms=" << control.gpu * 1000 / 25 << std::endl;
    }
    frame.last = rows;
    for (unsigned trial = 0; trial < 5; ++trial)
        for (unsigned j = 0; j < processors.size(); ++j) {
            const auto index = trial % 2 ? processors.size() - 1 - j : j;
            auto &processor = *processors[index];
            ComputeTimings timing;
            MatchedProfiles output;
            processor.calculate_into(output, batch, dp.view, 32);
            const auto start = std::chrono::steady_clock::now();
            double check_sum = 0;
            for (unsigned n = 0; n < 25; ++n) {
                processor.calculate_into(output, batch, dp.view, 32, &timing);
                check_sum += std::abs(output.values[2668 + n]);
            }
            std::cout << "PROFILE_BENCH trial=" << trial << " size=" << size << " tile=" << (16U << index)
                      << " gpu_ms=" << timing.gpu * 1000 / 25 << " wall_ms="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() *
                             1000 / 25
                      << " checksum=" << check_sum << '\n';
        }
}

} // namespace
int main(int argc, char **argv) {
    try {
        auto context = std::make_shared<gpu::Context>();
        if (argc == 2 && std::string_view(argv[1]) == "--profile-benchmark") {
            profile_benchmark(context);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--compact-benchmark") {
            profile_benchmark(context, true);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--compact-validation") {
            compact_validation(context);
            return 0;
        }
        std::cout << "Storage buffer limit: "
                  << context->gpu.physical.getProperties().limits.maxStorageBufferRange << '\n';
        noise(context);
        noise(context, .01F);
        noise(context, 100);
        noise(context, 1, 4);
        noise(context, 1, 6);
        noise(context, 1, 2, true);
        noise(context, 1, 2, false, 10000);
        finite(context);
        finite(context, 32);
        finite(context, 4);
        // The normal 259-row case covers 128- and 16-row FFT tiles. Exercise
        // both intermediate cached shapes against the same finite-waveform oracle.
        finite(context, 16, false, false, 19);
        finite(context, 16, false, false, 34);
        finite(context, 16, true);
        finite(context, 16, true, true);
        finite(context, 32, true, true, 19, 4096);
        finite(context, 4, false, false, 19, 4096);
        for (unsigned guard : {8U, 16U})
            finite(context, guard, true, true, 19, 4096);
        finite(context, 32, true, true, 19, 4096, 4300);
        focus(context);
        focus(context, 191625000, false, true, true, true, 1024, true);
        focus(context, 177500000, true, true, true, true, 3072, true);
        focus(context, 191625000);
        focus(context, 191625000, true);
        focus(context, 191625000, true, true);
        focus(context, 191625000, true, true, true);
        noise_steps(context);
        focus(context, 191625000, true, true, true, true);
        focus(context, 177500000, true, true, true, true, 4096);
        compact_validation(context);
        covariance_efficiency(context);
        owned_focus(context);
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
