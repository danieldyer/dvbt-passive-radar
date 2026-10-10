"""Apply audited instrumentation to pinned gr-dl5eu sources in the build tree."""
import pathlib
import shutil
import sys

source, destination = map(pathlib.Path, sys.argv[1:])
shutil.copytree(source / 'lib', destination, dirs_exist_ok=True)

def replace(path, old, new):
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f'Pinned upstream changed: {path}: {old!r}')
    path.write_text(text.replace(old, new))

header = destination / 'dvbt_ofdm_synchronization_impl.h'
code = destination / 'dvbt_ofdm_synchronization_impl.cc'
replace(header, '#include <gnuradio/filter/mmse_fir_interpolator_cc.h>', '#include "fractional_resampler.hpp"')
replace(header, 'gr::filter::mmse_fir_interpolator_cc d_inter;', 'radar::FractionalResampler d_inter;')
replace(header, 'float d_samp_inc;', 'double d_samp_inc;')
replace(header, 'float d_samp_phase;', 'double d_samp_phase;')
replace(header, 'float d_phase;', 'double d_phase;')
replace(header, '    float d_total_freq_error;', '''    float d_total_freq_error;
    std::vector<gr_complex> radar_previous_pilots;
    std::vector<std::complex<double>> radar_anchor_pilots;
    unsigned radar_anchor_count = 0;
    void radar_track_continual_pilots(const gr_complex* in);''')
replace(code, 'd_inter{ gr::filter::mmse_fir_interpolator_cc() }', 'd_inter{}')
replace(code, 'float loop_bw_timing = GR_M_PI / 10000;', 'float loop_bw_timing = GR_M_PI / 200;')
replace(code, 'd_cp_start_offset{ -10 }', 'd_cp_start_offset{ -radar::fft_window_margin }')
replace(code, 'd_cp_start_offset = -10;', 'd_cp_start_offset = -radar::fft_window_margin;')
replace(code, 'd_samp_inc = 1.0 - d_est_delta;', 'd_samp_inc = 1.0 - static_cast<double>(d_est_delta);')
replace(code, 'interpolate_input(&in[d_consumed + low - 3]', 'interpolate_input(&in[d_consumed + low - 47]')
# Telemetry describes the exact corrections applied to THIS symbol, before loop updates.
replace(code, '            if (d_interpolate) {\n                // I interpolate', '''            pmt::pmt_t timing = pmt::make_tuple(
                pmt::from_double(static_cast<double>(nitems_read(0) + d_consumed + low) + d_samp_phase),
                pmt::from_double(d_samp_inc), pmt::from_double(d_phase),
                pmt::from_double(d_total_freq_error / (2 * GR_M_PI) * 8000000.0 / d_fft_length));
            add_item_tag(0, nitems_written(0) + i, pmt::intern("radar_timing"), timing);
            if (d_cp_moved)
                add_item_tag(0, nitems_written(0) + i, pmt::intern("radar_boundary"), pmt::PMT_T);

            if (d_interpolate) {
                // I interpolate''')
# Port 1 is the actual un-equalized spectrum; do not undo equalization by multiplication.
replace(code, '                       &d_channel_gain[0],\n                       d_num_carriers * sizeof(gr_complex));',
              '                       d_integer_freq_derotated,\n                       d_num_carriers * sizeof(gr_complex));')
replace(code, 'estimate_fine_synchro(&d_channel_gain[0], &d_previous_channel_gain[0]);',
              'radar_track_continual_pilots(d_integer_freq_derotated);')
replace(code, 'd_initial_acquired = false;', 'd_initial_acquired = false; radar_anchor_count = 0; radar_anchor_pilots.clear();')

text = code.read_text()
begin = text.index('void dvbt_ofdm_synchronization_impl::derotate(')
end = text.index('\n\ndvbt_ofdm_synchronization::sptr', begin)
text = text[:begin] + '''// Double precision phase accumulation avoids a small systematic rotation error
// from a float unit-phasor recurrence over many millions of input samples.
void dvbt_ofdm_synchronization_impl::derotate(const gr_complex* in, gr_complex* out)
{
    d_phaseinc = -static_cast<double>(d_total_freq_error) / d_fft_length;
    auto phase = std::polar(1.0, d_phase);
    const auto step = std::polar(1.0, d_phaseinc);
    for (int i = 0; i < d_fft_length; ++i) {
        out[i] = in[i] * static_cast<gr_complex>(phase);
        phase *= step;
        if ((i & 1023) == 1023) phase /= std::abs(phase);
    }
    d_phase = std::remainder(d_phase + d_phaseinc * (d_fft_length + d_cp_length), 2 * GR_M_PI);
}

// Track the SAME physical carriers across symbols. Scattered-pilot interpolation
// changes with symbol modulo four and can bias a long-term clock estimate.
void dvbt_ofdm_synchronization_impl::radar_track_continual_pilots(const gr_complex* in)
{
    const bool have_previous = radar_previous_pilots.size() == static_cast<size_t>(d_num_cpilots);
    if (!have_previous) radar_previous_pilots.resize(d_num_cpilots);
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int j = 0; j < d_num_cpilots; ++j) {
        const int k = d_cpilots[j];
        const auto z = std::complex<double>(in[k]) * std::conj(std::complex<double>(radar_previous_pilots[j]));
        const double weight = std::abs(z);
        const double x = k - (d_num_carriers - 1) / 2.0;
        const double angle = std::arg(z);
        sw += weight; sx += weight*x; sy += weight*angle;
        sxx += weight*x*x; sxy += weight*x*angle;
        radar_previous_pilots[j] = in[k];
    }
    if (have_previous && !d_cp_moved && sw > 0 && sw*sxx > sx*sx) {
        const double slope = (sw*sxy - sx*sy) / (sw*sxx - sx*sx);
        const double intercept = (sy - slope*sx) / sw;
        const double duration = 1.0 + static_cast<double>(d_cp_length) / d_fft_length;
        if (radar_anchor_count < 160) {
            advance_freq_loop(static_cast<float>(intercept / duration));
            advance_delta_loop(static_cast<float>(slope / (2*GR_M_PI*duration)));
            if (radar_anchor_count == 128) radar_anchor_pilots.assign(d_num_cpilots, {});
            if (radar_anchor_count >= 128)
                for (int j = 0; j < d_num_cpilots; ++j)
                    radar_anchor_pilots[j] += std::complex<double>(in[d_cpilots[j]]) / 32.0;
            if (++radar_anchor_count == 160) {
                // Change from differential acquisition to an absolute phase PLL without
                // changing the current clock rate. The reference channel stays fixed.
                d_delta_aux = d_est_delta;
                d_freq_aux = d_fine_freq;
            }
        } else {
            sw = sx = sy = sxx = sxy = 0;
            for (int j = 0; j < d_num_cpilots; ++j) {
                const int k = d_cpilots[j];
                const auto z = std::complex<double>(in[k]) * std::conj(radar_anchor_pilots[j]);
                const double weight = std::abs(z), x = k - (d_num_carriers-1)/2.0;
                const double angle = std::arg(z);
                sw += weight; sx += weight*x; sy += weight*angle;
                sxx += weight*x*x; sxy += weight*x*angle;
            }
            const double anchor_slope = (sw*sxy-sx*sy)/(sw*sxx-sx*sx);
            const double anchor_phase = (sy-anchor_slope*sx)/sw;
            const double timing_error = anchor_slope/(2*GR_M_PI*duration);
            const double phase_error = anchor_phase/duration;
            d_delta_aux += d_beta_timing*timing_error;
            d_est_delta = d_delta_aux + d_alpha_timing*timing_error;
            d_freq_aux += d_beta_freq*phase_error;
            d_fine_freq = d_freq_aux + d_alpha_freq*phase_error;
        }
    }
    d_samp_inc = 1.0 - static_cast<double>(d_est_delta);
}
''' + text[end:]
code.write_text(text)

# Production timing belongs to the common ADC stream. Synchronizers observe
# pilots but never independently retime a channel that has already been warped.
replace(header, '#include "fractional_resampler.hpp"',
        '#include "fractional_resampler.hpp"\n#include "sample_clock.hpp"')
replace(header, '    unsigned radar_anchor_count = 0;', '''    unsigned radar_anchor_count = 0;
    std::shared_ptr<radar::SampleClock> radar_common_clock;
    size_t radar_clock_channel = 0;
    uint64_t radar_clock_origin = 0;
    unsigned radar_clock_decimation = 1;
    double radar_clock_coordinate = 0;''')
replace(header, 'public:\n', '''public:
    void radar_use_common_clock(std::shared_ptr<radar::SampleClock> clock, size_t channel,
                               uint64_t origin, unsigned decimation) {
        radar_common_clock = std::move(clock); radar_clock_channel = channel;
        radar_clock_origin = origin; radar_clock_decimation = decimation;
        d_interpolate = false; d_samp_inc = 1; d_samp_phase = 0;
    }
''')
replace(code, 'd_initial_acquired = false; radar_anchor_count = 0; radar_anchor_pilots.clear();',
        '''d_initial_acquired = false; radar_anchor_count = 0; radar_anchor_pilots.clear();
                radar_previous_pilots.clear();
                if (radar_common_clock) radar_common_clock->reset_observer(radar_clock_channel);''')
replace(code, '        if (radar_anchor_count < 160) {', '''        if (radar_common_clock)
            radar_common_clock->observe_rate(radar_clock_channel, radar_clock_coordinate,
                slope / (2*GR_M_PI*duration));
        if (radar_anchor_count < 160) {''')
replace(code, '            const double timing_error = anchor_slope/(2*GR_M_PI*duration);', '''            if (radar_common_clock)
                radar_common_clock->observe_offset(radar_clock_channel, radar_clock_coordinate,
                    anchor_slope*d_fft_length/(2*GR_M_PI)*radar_clock_decimation);
            const double timing_error = anchor_slope/(2*GR_M_PI*duration);''')
replace(code, '    d_samp_inc = 1.0 - static_cast<double>(d_est_delta);',
        '    d_samp_inc = radar_common_clock ? 1.0 : 1.0 - static_cast<double>(d_est_delta);')
replace(code, '            pmt::pmt_t timing = pmt::make_tuple(', '''            radar_clock_coordinate = static_cast<double>(radar_clock_origin) +
                (static_cast<double>(nitems_read(0) + d_consumed + low) + d_fft_length/2.0)*radar_clock_decimation;
            pmt::pmt_t timing = pmt::make_tuple(''')

