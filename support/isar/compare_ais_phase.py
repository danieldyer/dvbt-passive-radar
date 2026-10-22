"""Direct measured interband phase comparison against the AIS geometry.

No motion, path rate, antenna location or phase trajectory is fitted. The
existing crop has AIS range migration and carrier deramp applied. We extract
one scalar phase history per band with a fixed range filter, undo only that
known carrier deramp, divide each phase by its own RF wavenumber, then compare
path changes. Only constant phase origins are removed. A single range-cell
control tests sensitivity to the fixed spatial filter.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

C = 299792458.
NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
COLOURS = ['#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00']


def path_geometry(position, velocity, transmitter, receiver):
    u = position-transmitter
    v = position-receiver
    rt, rr = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
    path = rt+rr-np.linalg.norm(transmitter-receiver)
    rate = np.sum((u/rt[..., None]+v/rr[..., None])*velocity, axis=-1)
    return path, rate


def phase_to_path(time, aligned_signal, k, path_time, ais_path, grid, anchor):
    """Return measured/expected changes after removing a constant only.

    aligned_signal already contains exp(+i k L_AIS). Its residual phase is
    therefore -k (L_actual-L_AIS), up to a constant scattering/filter phase.
    """
    phase = np.interp(grid, time, np.unwrap(np.angle(aligned_signal)))
    expected = np.interp(grid, path_time, ais_path)
    expected -= np.mean(expected[anchor])
    residual = -phase/k
    residual -= np.mean(residual[anchor])
    return expected+residual, expected, residual


def slope(time, values):
    centred = time-time.mean()
    return np.einsum('n,...n->...', centred, values)/np.sum(centred**2)


def extract(source, interval, training, grid, anchor):
    with np.load(source) as data:
        metadata = json.loads(str(data['metadata_json']))
        time = data['time']
        keep = (time >= interval[0]-1)&(time <= interval[1]+1)&(data['information'] > 0)
        time = time[keep]
        raw = data['value'][keep]
        information = data['information'][keep]
        train = (time >= training[0])&(time < training[1])
        if np.count_nonzero(train) < 8:
            raise ValueError('Insufficient training samples')
        offsets = data['offset_samples']
        lag = offsets[:, None]-offsets[None, :]
        covariance = data['pulse_samples'][lag+64]
        covariance = (covariance+covariance.conj().T)/2
        whiten = solve_triangular(np.linalg.cholesky(covariance), np.eye(len(offsets)), lower=True)
        white = raw@whiten.T
        template = np.average(white[train], axis=0, weights=information[train])
        norm = np.vdot(template, template).real
        signal = white@template.conj()/norm
        # Select the single-cell control without its subsequent phase samples.
        mean = np.average(raw[train], axis=0, weights=information[train])
        peak = int(np.argmax(abs(mean)**2))
        central = int(np.flatnonzero(offsets == 0)[0])
        k = 2*np.pi*metadata['rf_hz']/C
        modes = {}
        for name, scalar in [('fixed_range_pattern', signal), ('strongest_range_cell', raw[:, peak]),
                             ('common_ais_range_cell', raw[:, central])]:
            measured, expected, residual = phase_to_path(time, scalar, k, data['row_time'],
                                                         data['centre_path_m'], grid, anchor)
            amplitude = np.interp(grid, time, abs(scalar))
            amplitude /= np.mean(amplitude[anchor])
            modes[name] = {'measured_path_change_m': measured.tolist(),
                           'ais_path_change_m': expected.tolist(),
                           'residual_path_change_m': residual.tolist(),
                           'relative_amplitude': amplitude.tolist()}
        midpoint = np.mean(interval)
        position = np.array([np.interp(midpoint, data['row_time'], data['centre_ecef_m'][:, j]) for j in range(3)])
        velocity = np.array([np.interp(midpoint, data['row_time'], data['centre_velocity_ecef_mps'][:, j]) for j in range(3)])
        transmitter, receiver = data['transmitter_ecef_m'], data['receiver_ecef_m']
        _, rate = path_geometry(position, velocity, transmitter, receiver)
        dt = .001
        plus = path_geometry(position+dt*velocity, velocity, transmitter, receiver)[0]
        minus = path_geometry(position-dt*velocity, velocity, transmitter, receiver)[0]
        finite_difference_error = float(abs((plus-minus)/(2*dt)-rate))
        if finite_difference_error > 1e-5:
            raise ValueError('Geometric path-rate derivative check failed')
        with source.open('rb') as handle:
            source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        return {'name': NAMES[metadata['channel']], 'rf_hz': metadata['rf_hz'], 'k_rad_per_m': k,
                'source': str(source), 'source_sha256': source_hash, 'peak_offset_samples': int(offsets[peak]),
                'peak_offset_m': float(offsets[peak]*C/8e6),
                'range_offsets_m': (offsets*C/8e6).tolist(),
                'training_range_profile_relative_db': (20*np.log10(abs(mean)/max(abs(mean)))).tolist(),
                'transmitter_ecef_m': transmitter.tolist(), 'receiver_ecef_m': receiver.tolist(),
                'midpoint_position_ecef_m': position.tolist(), 'midpoint_velocity_ecef_mps': velocity.tolist(),
                'ais_excess_path_rate_mps': float(rate), 'ais_doppler_hz': float(-rate*metadata['rf_hz']/C),
                'path_rate_derivative_check_error_mps': finite_difference_error, 'modes': modes}


def compare(channels, grid, mode):
    observed = np.array([c['modes'][mode]['measured_path_change_m'] for c in channels])
    expected = np.array([c['modes'][mode]['ais_path_change_m'] for c in channels])
    residual = observed-expected
    abc_observed = observed[4]-np.mean(observed[:4], axis=0)
    abc_expected = expected[4]-np.mean(expected[:4], axis=0)
    pairs = []
    for a, b in itertools.combinations(range(5), 2):
        error = residual[a]-residual[b]
        pairs.append({'pair': NAMES[a]+'/'+NAMES[b],
                      'phase_at_ABC_rf_rms_deg': float(np.sqrt(np.mean(error**2))*channels[4]['k_rad_per_m']*180/np.pi),
                      'observed_differential_rate_mps': float(slope(grid, observed[a]-observed[b])),
                      'expected_differential_rate_mps': float(slope(grid, expected[a]-expected[b]))})
    return {'mode': mode, 'abc_minus_other_four_measured_m': abc_observed.tolist(),
            'abc_minus_other_four_expected_m': abc_expected.tolist(),
            'abc_minus_other_four_observed_rate_mps': float(slope(grid, abc_observed)),
            'abc_minus_other_four_expected_rate_mps': float(slope(grid, abc_expected)),
            'abc_difference_error_rms_m': float(np.sqrt(np.mean((abc_observed-abc_expected)**2))),
            'per_band_observed_rate_mps': slope(grid, observed).tolist(),
            'per_band_expected_rate_mps': slope(grid, expected).tolist(), 'pairs': pairs}


def plot(result, output):
    time = np.array(result['time_s'])
    channels = result['channels']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    main = 'fixed_range_pattern'
    for c, colour in zip(channels, COLOURS):
        d = c['modes'][main]
        axes[0, 0].plot(time, np.array(d['residual_path_change_m'])*100, color=colour, label=c['name'])
        axes[1, 0].plot(time, d['relative_amplitude'], color=colour, label=c['name'])
    axes[0, 0].set_title('Independent measured phase minus each channel’s AIS prediction')
    axes[0, 0].set_ylabel('Residual excess-path change (cm)')
    axes[0, 0].legend(ncol=5, fontsize=8)
    axes[1, 0].set_title('Amplitude of the fixed-filter phase histories')
    axes[1, 0].set_ylabel('Amplitude / first-half mean')
    for mode, colour, label in [(main, '#222222', 'Measured: fixed range pattern'),
                                ('strongest_range_cell', '#8a4fa3', 'Measured: strongest range cell'),
                                ('common_ais_range_cell', '#0072b2', 'Measured: same AIS range coordinate')]:
        d = result['comparisons'][mode]
        axes[0, 1].plot(time, np.array(d['abc_minus_other_four_measured_m'])*100,
                        color=colour, label=label, lw=1.5 if mode == main else 1)
    expected = np.array(result['comparisons'][main]['abc_minus_other_four_expected_m'])
    axes[0, 1].plot(time, expected*100, color='#009e73', lw=2.5, ls='--', label='Predicted: separated TX geometry + AIS')
    axes[0, 1].set_ylabel('ABC minus mean of other four (cm)')
    axes[0, 1].set_title('Direct test: measured difference versus geometric prediction')
    axes[0, 1].legend(fontsize=8)
    axis = axes[0, 1].secondary_yaxis('right', functions=(
        lambda cm: -cm/100*channels[4]['k_rad_per_m']*180/np.pi,
        lambda deg: -deg*np.pi/180/channels[4]['k_rad_per_m']*100))
    axis.set_ylabel('Equivalent phase at ABC RF (degrees)')
    for c, colour in zip(channels, COLOURS):
        d = c['modes'][main]
        cell = c['modes']['strongest_range_cell']
        delta = np.array(cell['residual_path_change_m'])-np.array(d['residual_path_change_m'])
        axes[1, 1].plot(time, delta*100, color=colour, label=c['name'])
    axes[1, 1].set_ylabel('Cell minus pattern phase, as path (cm)')
    axes[1, 1].set_title('Range-extraction sensitivity: no trajectory refitted')
    for ax in axes.flat:
        ax.axvspan(*result['training_interval_s'], color='#999999', alpha=.12)
        ax.axhline(0, color='#999999', lw=.6)
        ax.set_xlim(result['interval_s'])
        ax.set_xlabel('Recording time (s)')
    fig.suptitle('Ship phase versus AIS geometry · range migration aligned separately for each transmitter\n'
                 'Only constant phase offsets removed; no fitted phase rate, motion or antenna correction')
    fig.savefig(output/'ais-phase-comparison.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), layout='constrained', sharex=True)
    mode = 'common_ais_range_cell'
    observed = np.array([c['modes'][mode]['measured_path_change_m'] for c in channels])
    for c, colour in zip(channels, COLOURS):
        axes[0].plot(time, 100*np.array(c['modes'][mode]['residual_path_change_m']), color=colour, label=c['name'])
        axes[2].plot(time, c['modes'][mode]['relative_amplitude'], color=colour, label=c['name'])
    axes[0].set_title('Same AIS-aligned range coordinate in all five channels: no phase-rate fit')
    axes[0].set_ylabel('Measured − AIS path change (cm)')
    axes[0].legend(ncol=5)
    for c in range(4):
        axes[1].plot(time, 100*(observed[4]-observed[c]), color=COLOURS[c], lw=.8, alpha=.75,
                     label='ABC − '+NAMES[c])
    d = result['comparisons'][mode]
    axes[1].plot(time, 100*np.array(d['abc_minus_other_four_measured_m']), color='#222222', lw=2, label='ABC − four-band mean')
    axes[1].plot(time, 100*np.array(d['abc_minus_other_four_expected_m']), color='#009e73', lw=2.5, ls='--', label='AIS + geometry prediction')
    axes[1].set_ylabel('Differential excess-path change (cm)')
    axes[1].set_title('Does the measured ABC difference follow the separated-transmitter prediction?')
    axes[1].legend(ncol=3, fontsize=9)
    axes[2].set_ylabel('Amplitude / first-half mean')
    axes[2].set_title('Return amplitude: phase becomes less reliable when a return fades')
    axes[-1].set_xlabel('Recording time (s)')
    for ax in axes:
        ax.axvspan(*result['training_interval_s'], color='#999999', alpha=.12)
        ax.axhline(0, color='#999999', lw=.6)
        ax.set_xlim(result['interval_s'])
    fig.suptitle('Direct interband comparison · constant phase origins removed only\n'
                 'Each phase is divided by its own RF wavenumber before comparing path changes')
    fig.savefig(output/'common-cell-phase.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    for c, colour in zip(channels, COLOURS):
        ax.plot(c['range_offsets_m'], c['training_range_profile_relative_db'], '.-', color=colour,
                label=c['name']+f" (strongest cell: {c['peak_offset_m']:+.0f} m)")
    ax.axvspan(-C/8e6/2, C/8e6/2, color='#777777', alpha=.12, label='Common AIS-centred cell')
    ax.set_xlabel('Excess-path offset from channel-specific AIS prediction (m)')
    ax.set_ylabel('Training mean profile magnitude (dB relative to each band’s peak)')
    start, stop = result['training_interval_s']
    ax.set_title(f'Range selection audit · {start:g}–{stop:g} s · no phase or motion fitted\n'
                 'Separate peaks in this crop need not be the same physical scattering feature')
    ax.legend(ncol=3, fontsize=9)
    fig.savefig(output/'range-selection.png', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval', type=float, nargs=2, default=[164, 165])
    args = p.parse_args()
    first, last = args.interval
    training = [first, (first+last)/2]
    grid = np.linspace(first, last, round((last-first)*50)+1)
    anchor = (grid >= training[0])&(grid < training[1])
    channels = [extract(args.fields/f'channel-{c}.npz', args.interval, training, grid, anchor) for c in range(5)]
    result = {'scope': __doc__, 'interval_s': args.interval, 'training_interval_s': training,
              'time_s': grid.tolist(), 'channels': channels,
              'geometry_scope': 'Nominal licensed-main hypothesis: Seven/SBS/Nine/Ten at Ornata, ABC at Ridge. Active sites and phase centres are not established. AIS Hermite position/SOG/COG interpolation from the existing fixed observation crop. No previous autofocus or refined antenna fit used.',
              'limitations': ['A range patch may contain multiple unresolved physical scatterers.',
                              'Fixed spatial filter learned on first-half data; second half is independent in time. No stationary term is fitted inside this diagnostic.',
                              'Input archive already subtracted its capture-time frozen stationary template. No further DC notch or temporal background subtraction is applied here.',
                              'Range alignment is the fixed AIS alignment, not true target migration if AIS or the assumed scattering position is inaccurate.',
                              'Reported slopes summarize measured curves; they are not corrections applied to them, and do not provide formal statistical confidence intervals.'],
              'comparisons': {mode: compare(channels, grid, mode) for mode in
                              ['fixed_range_pattern', 'strongest_range_cell', 'common_ais_range_cell']}}
    source = Path(__file__)
    result['implementation_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'ais-phase-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    plot(result, args.output)
    for mode, comparison in result['comparisons'].items():
        print(json.dumps({k: v for k, v in comparison.items() if k not in
                         ('abc_minus_other_four_measured_m', 'abc_minus_other_four_expected_m', 'pairs')}), flush=True)


if __name__ == '__main__':
    main()
