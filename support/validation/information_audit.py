#!/usr/bin/env python3
"""Conditional information-loss calculations, independent of the GPU kernels.

Run information_probe first. These models deliberately include counterexamples;
the output is an audit, not a certificate of real-capture optimality.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def loss(eta):
    return float(-10 * np.log10(eta))


def linear_compression():
    rng = np.random.default_rng(57921)
    n, dimensions = 64, 3
    a = (rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))) / np.sqrt(n)
    covariance = a @ a.conj().T + np.eye(n)
    derivative = rng.normal(size=(n, dimensions)) + 1j*rng.normal(size=(n, dimensions))
    white = np.linalg.solve(np.linalg.cholesky(covariance), derivative)
    full = 2*(white.conj().T @ white).real
    # Exact score compression for this known Gaussian mean/covariance model.
    score = derivative.conj().T @ np.linalg.inv(covariance)
    results = {}
    for name, operator in [('score', score), ('arbitrary_crop', np.eye(n)[:16])]:
        compressed_d = operator @ derivative
        compressed_r = operator @ covariance @ operator.conj().T
        information = 2*(compressed_d.conj().T @ np.linalg.solve(compressed_r, compressed_d)).real
        lower = np.linalg.cholesky(full)
        relative = np.linalg.solve(lower, information)
        relative = np.linalg.solve(lower, relative.T).T
        eigenvalues = np.linalg.eigvalsh(relative)
        assert eigenvalues[0] > 0 and eigenvalues[-1] < 1 + 1e-12
        results[name] = {'relative_fisher_eigenvalues': eigenvalues.tolist()}
    assert np.max(np.abs(np.asarray(results['score']['relative_fisher_eigenvalues']) - 1)) < 1e-12
    return results


def moment_compression():
    n = 8192
    t = (np.arange(n) - (n-1)/2)/8e6
    basis = np.stack([np.ones(n), t], axis=1)
    q, _ = np.linalg.qr(basis)
    rows = []
    for frequency in [0, 25, 50, 100, 150, 200, 300, 400]:
        signal = np.exp(2j*np.pi*frequency*t)
        implemented = 1 + 2j*np.pi*frequency*t
        exact = np.vdot(signal, signal).real
        eta = abs(np.vdot(implemented, signal))**2 / (np.vdot(implemented, implemented).real*exact)
        available = np.linalg.norm(q.conj().T @ signal)**2 / exact
        # Within-symbol delay/frequency nuisance test; this is not the Fisher
        # information of a full multi-second coherent aperture.
        derivative = np.stack([signal, 1j*signal, 2j*np.pi*t*signal], axis=1)
        full = 2*(derivative.conj().T @ derivative).real
        reduced = q.conj().T @ derivative
        reduced = 2*(reduced.conj().T @ reduced).real
        def frequency_information(matrix):
            return matrix[2, 2] - matrix[2, :2] @ np.linalg.solve(matrix[:2, :2], matrix[:2, 2])
        rows.append({'doppler_hz': frequency, 'first_order_snr_loss_db': loss(eta),
                     'best_two_moment_snr_loss_db': loss(available),
                     'within_symbol_frequency_information_fraction':
                         float(frequency_information(reduced)/frequency_information(full))})
    return rows


def observation_selection():
    """Exact energy/SNR in a known-waveform, iid complex-white-noise model."""
    rng = np.random.default_rng(645923)
    fft, cp, count = 8192, 512, 36
    bins = np.arange(-3408, 3409)
    levels = np.arange(-7, 8, 2)/np.sqrt(42)
    spectrum = np.zeros((count, fft), dtype=complex)
    carriers = rng.choice(levels, (count, len(bins))) + 1j*rng.choice(levels, (count, len(bins)))
    pilot_values = rng.choice([-4/3, 4/3], len(bins))
    # Coding constraints have no role when the transmitted waveform is known.
    # This is an OFDM observation-selection model, not a DVB-T decoder test.
    for row in range(count):
        selected = np.arange(len(bins)) % 12 == (row % 4)*3
        carriers[row, selected] = pilot_values[selected]
    spectrum[:, bins % fft] = carriers
    useful = np.fft.ifft(spectrum, axis=1, norm='ortho')
    source = np.concatenate([useful[:, -cp:], useful], axis=1).ravel()
    length = fft+cp
    whole = np.arange(2*length, (count-2)*length)
    windows = (np.arange(2, count-2)[:, None]*length + cp - 128 + np.arange(fft)).ravel()
    result = []
    for delay in [0, 16, 700]:
        for frequency in [0, 100, 200]:
            full = source[whole-delay]*np.exp(2j*np.pi*frequency*whole/8e6)
            selected = source[windows-delay]*np.exp(2j*np.pi*frequency*windows/8e6)
            transformed = np.fft.fft(selected.reshape(-1, fft), axis=1, norm='ortho')
            energy = float(np.vdot(full, full).real)
            window_energy = float(np.vdot(selected, selected).real)
            occupied_energy = float(np.sum(abs(transformed[:, bins % fft])**2))
            result.append({'delay_samples': delay, 'doppler_hz': frequency,
                           'window_selection_loss_db': loss(window_energy/energy),
                           'occupied_bins_additional_loss_db': loss(occupied_energy/window_energy),
                           'total_loss_db': loss(occupied_energy/energy)})
    return {'model': 'known finite OFDM waveform; iid complex Gaussian noise at 8 MS/s; no analogue filter',
            'guard_1_16_equal_energy_duty_loss_db': float(10*np.log10(17/16)), 'cases': result}


def covariance_counterexample():
    n = 256
    t = (np.arange(n)-(n-1)/2)*3/n
    nuisance = np.exp(2j*np.pi*12*t)/np.sqrt(n)
    signal = np.exp(2j*np.pi*12.4*t)/np.sqrt(n)
    overlap = abs(np.vdot(nuisance, signal))**2
    result = []
    for power in [0, 10, 100, 1000, 10000]:
        optimum = 1-power/(1+power)*overlap
        diagonal_snr = 1/(1+power*overlap)
        result.append({'coherent_interference_eigenvalue_excess': power,
                       'diagonal_weight_snr_loss_db': loss(diagonal_snr/optimum)})
    return {'model': 'R=I+P*b*b^H; 3 s; target 12.4 Hz; random complex interference amplitude at 12 Hz',
            'squared_template_overlap': float(overlap), 'cases': result}


def minimum_phase_information():
    # Local high-SNR limit at H=1 for log H = a0+i*phi+(a1+i*b1)e^-iw.
    # Eliminate unknown scalar gain and phase in the full complex likelihood;
    # compare the magnitude-only fit's a1/b1 information with the same nuisance.
    w = 2*np.pi*np.arange(-3408, 3409)/8192
    d = np.stack([np.ones(len(w)), 1j*np.ones(len(w)), np.exp(-1j*w), 1j*np.exp(-1j*w)], axis=1)
    j = 2*(d.conj().T @ d).real
    full = j[2:, 2:] - j[2:, :2] @ np.linalg.solve(j[:2, :2], j[:2, 2:])
    magnitude_d = np.stack([np.ones(len(w)), np.cos(w), np.sin(w)], axis=1)
    j = 2*magnitude_d.T @ magnitude_d
    magnitude = j[1:, 1:] - j[1:, :1] @ np.linalg.solve(j[:1, :1], j[:1, 1:])
    lower = np.linalg.cholesky(full)
    relative = np.linalg.solve(lower, magnitude)
    relative = np.linalg.solve(lower, relative.T).T
    retention = np.linalg.eigvalsh(relative)
    return {'model': 'local high-SNR, correctly specified order-one minimum-phase response; white complex noise',
            'magnitude_only_relative_fisher_eigenvalues': retention.tolist(),
            'variance_bound_penalty_db': (-10*np.log10(retention)).tolist(),
            'qualification': 'Not a reason to fit arbitrary multipath as hardware; that needs a joint identifiable model.'}


def transmitter_grid_mismatch():
    # A common receive clock cannot remove different transmitter symbol rates.
    # Isolate the resulting timing ramp on a flat occupied spectrum. Even the
    # best complex scalar cannot predict a delayed source on the wrong grid.
    omega = 2*np.pi*np.arange(-3408, 3409)/8192
    fractional_rate = 40e-9
    cases = []
    for elapsed in [1, 3, 10]:
        delay = fractional_rate*elapsed*8e6
        response = np.exp(-1j*omega*delay)
        scalar = np.mean(response)
        residual = np.mean(abs(response-scalar)**2)
        cases.append({'elapsed_seconds': elapsed, 'timing_difference_channel_samples': delay,
                      'best_scalar_residual_power_db': float(10*np.log10(residual))})
    return {'relative_transmitter_rate_ppb': fractional_rate*1e9,
            'model': 'analytic delayed flat occupied spectrum; a hypothetical transmitter rate bias, not measured hardware',
            'qualification': 'Illustrates missing per-transmitter waveform timing, not resampler numerical error or radar SNR loss.',
            'cases': cases}


def cancellation():
    # Gaussian nuisance-amplitude elimination, a comparison to the unattainable
    # known-nuisance bound as well as to the correct unknown-nuisance bound.
    delays = np.linspace(0, 1200, 401)
    frequencies = np.linspace(0, 2, 401)
    bandwidth, cpi = 6.65625e6, 3
    overlap = np.sinc(bandwidth*delays[None, :]/299792458)**2 * np.sinc(cpi*frequencies[:, None])**2
    retention = np.maximum(0, 1-overlap)
    return delays, frequencies, retention


def response_fit(path):
    with path.open() as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    region = [row for row in rows if row['delay_samples'] >= 16 and row['doppler_hz'] <= 100]
    minimum = min(region, key=lambda row: row['refit_echo_voltage_gain'])
    return {'model': 'noiseless paired echo injection into the diagonal OFDM response model; live 408-row batch',
            'qualification': 'Voltage retention only, not noise-normalised SNR or a finite beyond-guard echo test.',
            'delay_at_minimum_samples': minimum['delay_samples'],
            'doppler_at_minimum_hz': minimum['doppler_hz'],
            'minimum_echo_voltage_gain_at_delay_ge_16_samples': minimum['refit_echo_voltage_gain'],
            'minimum_voltage_gain_db': float(20*np.log10(minimum['refit_echo_voltage_gain'])),
            'cases': rows}


def response_convergence(path):
    cases = []
    for line in path.read_text().splitlines():
        if line.startswith('fit_order='):
            cases.append({key: float(value) for key, value in (field.split('=') for field in line.split())})
    assert len(cases) == 8
    omega = 2*np.pi*np.arange(-3408, 3409)/8192
    basis = np.stack([np.ones(len(omega))] +
                     [v for d in range(1, 9) for v in (np.cos(d*omega), np.sin(d*omega))], axis=1)
    return {'model': 'noiseless known order-one response, production GPU solver; relative power error from known truth',
            'order_eight_magnitude_basis_condition_number': float(np.linalg.cond(basis)), 'cases': cases}


def prototype_bounds():
    text = (ROOT/'src/channel_filters.hpp').read_text()
    result = []
    for decimation in range(1, 9):
        match = re.search(rf'decimate_{decimation} = \{{(.*?)\}};', text, flags=re.S)
        h = np.asarray([float(v) for v in re.findall(r'([-+\deE.]+)F', match.group(1))], dtype=np.float32).astype(float)
        response = np.fft.rfft(h, 1 << 20)
        f = np.fft.rfftfreq(1 << 20, 1/(8e6*decimation))
        pass_error = np.max(np.abs(abs(response[f <= 3328125])-1))
        stop = np.max(abs(response[f >= 3671875]))
        result.append({'input_msps': 8*decimation, 'taps': len(h),
                       'passband_amplitude_error': float(pass_error), 'stopband_db': float(20*np.log10(stop))})
    rate, bandwidth, bin_width = 61.44e6, 6.65625e6, 8e6/8192
    total_noise = (2/(2**12))**2/6
    return {'float32_prototype_dense_grid': result,
            'ideal_12_bit_complex_adc_noise_dbfs': float(10*np.log10(total_noise)),
            'in_channel_integrated_noise_dbfs': float(10*np.log10(total_noise*bandwidth/rate)),
            'one_976_5625_hz_bin_noise_dbfs': float(10*np.log10(total_noise*bin_width/rate)),
            'model': 'ideal nonoverloaded 12-bit I/Q ADC; independent uniform quantisation error; full-scale complex tone power=1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimators', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--response-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True);args.figures.mkdir(parents=True, exist_ok=True)
    estimators = json.loads(args.estimators.read_text())
    audit = {'linear_fisher_information': linear_compression(), 'moments': moment_compression(),
             'observations': observation_selection(), 'correlated_noise': covariance_counterexample(),
             'minimum_phase_estimator': minimum_phase_information(),
             'transmitter_grid_mismatch': transmitter_grid_mismatch(),
             'live_response_fit': response_fit(args.response),
             'response_convergence': response_convergence(args.response_log),
             'channel_filter': prototype_bounds(),
             'phase_scale': {'illustrative_rms_degrees': .37, 'full_vhf_occupied_span_hz': 55.65625e6,
                 'equivalent_time_rms_ps': .37/360/55.65625e6*1e12,
                 'equivalent_64msps_samples': .37/360/55.65625e6*64e6,
                 'qualification': 'A scale conversion, not a new acceptance threshold or measured map coherence loss.'},
             'status': 'NOT CERTIFIED: current estimator and archive lack general information-preservation guarantees'}
    (args.output/'calculations.json').write_text(json.dumps(audit, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    cases = estimators['pilot_likelihood']
    labels = ['White 10 dB', 'White 20 dB', 'White 40 dB', 'Coloured', 'Coloured\n32-row anchor']
    at = np.arange(len(cases))
    for offset, key, label in [(-.18, 'production', 'Production'), (.18, 'known_covariance_comparator', 'Known covariance comparator')]:
        ax[0, 0].bar(at+offset, [c[key]['mse_over_crlb'] for c in cases], width=.36, label=label)
    ax[0, 0].axhline(1, color='black', linestyle='--', label='CRLB');ax[0, 0].set_yscale('log')
    ax[0, 0].set_xticks(at, labels, fontsize=8);ax[0, 0].set_ylabel('Timing MSE / model CRLB');ax[0, 0].legend(fontsize=8)
    clocks = estimators['common_clock']
    bars = ax[0, 1].bar(['3° loop setting', '0.1° loop setting'], [c['phase_rms_degrees'] for c in clocks])
    ax[0, 1].bar_label(bars, fmt='%.3f°', padding=3)
    ax[0, 1].set(ylabel='3 s differential phase RMS (degrees)',
                 title='Five-channel clock stress model; no pass/fail threshold', ylim=(0, 2.1))
    moments = audit['moments']
    ax[1, 0].plot([m['doppler_hz'] for m in moments], [m['first_order_snr_loss_db'] for m in moments], 'o-', label='Current first-order weighting')
    ax[1, 0].plot([m['doppler_hz'] for m in moments], [m['best_two_moment_snr_loss_db'] for m in moments], 'o-', label='Best possible from two moments')
    ax[1, 0].axvline(100, color='black', linestyle=':', label='Current display limit')
    ax[1, 0].set(xlabel='Within-symbol Doppler (Hz)', ylabel='SNR loss (dB)', title='Uniform energy, white noise, 1.024 ms', xlim=(0, 200), ylim=(-.001, .018));ax[1, 0].legend(fontsize=8)
    corr = audit['correlated_noise']['cases'][1:]
    ax[1, 1].semilogx([r['coherent_interference_eigenvalue_excess'] for r in corr], [r['diagonal_weight_snr_loss_db'] for r in corr], 'o-')
    ax[1, 1].set(xlabel='Correlated interference eigenvalue excess', ylabel='Diagonal-filter SNR loss (dB)', title='Explicit temporal covariance counterexample')
    fig.suptitle('Conditional audit models — not measurements of recorded-target SNR')
    fig.savefig(args.figures/'estimation_and_compression.png', dpi=170);plt.close(fig)
    delays, frequencies, retention = cancellation()
    fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    show = ax.pcolormesh(delays, frequencies, 10*np.log10(np.maximum(retention, 1e-4)), vmin=-20, vmax=0, shading='auto')
    ax.set(xlabel='Excess path (m)', ylabel='Doppler (Hz)', title='Information retained after eliminating one unknown direct-path amplitude')
    fig.colorbar(show, ax=ax, label='Relative to known-direct-path bound (dB)')
    fig.savefig(args.figures/'nuisance_information.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    rows = audit['live_response_fit']['cases']
    for ax, minimum in zip(axes, [1, 16]):
        for frequency in [0, .1, 1, 10]:
            selected = [row for row in rows if row['doppler_hz'] == frequency and row['delay_samples'] >= minimum]
            ax.plot([row['delay_samples']*299792458/8e6 for row in selected],
                    [20*np.log10(row['refit_echo_voltage_gain']) for row in selected], 'o-', label=f'{frequency:g} Hz')
        ax.set(xscale='log', xlabel='Excess path (m)', ylabel='Echo voltage retention (dB)')
        ax.legend(fontsize=8)
    axes[0].set(ylim=(-60, 1), title='Full diagnostic span')
    axes[0].axvline(600, color='black', linestyle=':')
    axes[1].set(ylim=(-.15, .005), title='Detail from approximately 600 m')
    fig.suptitle('Live 408-row response refit — paired injection, not SNR')
    fig.savefig(args.figures/'response_retention.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for order in [1, 8]:
        rows = [row for row in audit['response_convergence']['cases'] if row['fit_order'] == order]
        axes[0].plot([row['iterations'] for row in rows],
                     [10*np.log10(row['relative_response_power_error']) for row in rows], 'o-', label=f'Order {order}')
        axes[1].plot([row['iterations'] for row in rows], [row['phase_rms_degrees'] for row in rows], 'o-', label=f'Order {order}')
    axes[0].set(xlabel='Coordinate descent iterations', ylabel='Relative response power error (dB)')
    axes[1].set(xlabel='Coordinate descent iterations', ylabel='Response phase error RMS (degrees)')
    for ax in axes: ax.legend()
    fig.suptitle('Noiseless representable channel — production uses order 8 and 16 iterations')
    fig.savefig(args.figures/'response_convergence.png', dpi=170);plt.close(fig)
    print(json.dumps({'output': str(args.output/'calculations.json'), 'figures': str(args.figures), 'status': audit['status']}))


if __name__ == '__main__':
    main()
