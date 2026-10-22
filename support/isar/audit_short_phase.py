"""Measure stationary phase drift and short bright ship phase transfer.

Reads the unnotched IQ replay; no production samples or fit inputs are changed.
The first half of each window selects brightness and fits all coefficients.
The second half is a genuinely subsequent prediction, not an interleaved fit.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from short_phase import Patch, fit_rate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from geographic_io import read_profiles

C = 299792458.
NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
COLOURS = ['#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stationary(source, output):
    d = json.loads(source.read_text())
    t = np.array(d['times_s'])
    use = t >= 90
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout='constrained', sharex=True)
    result = {'source': str(source), 'sha256': digest(source),
              'evaluation_interval_s': [90, float(t[-1])],
              'coherent_interval_s': d['coherence_seconds'], 'regions': []}
    for region, ax in enumerate(axes):
        curves = np.array([c['regions'][region]['phase_degrees'] for c in d['channels']])
        pairs = []
        for a, b in itertools.combinations(range(5), 2):
            phase = curves[a]-curves[b]
            phase -= np.mean(phase[use])
            rms = float(np.sqrt(np.mean(phase[use]**2)))
            label = NAMES[a]+'/'+NAMES[b]
            pairs.append({'pair': label, 'rms_deg': rms,
                          'p05_p95_deg': np.percentile(phase[use], [5, 95]).tolist()})
            ax.plot(t[use], phase[use], lw=.75, label=f'{label}: {rms:.2f}°')
        ranges = d['regions_m'][region]
        result['regions'].append({'range_m': ranges, 'pairs': pairs})
        ax.set_title(f'{ranges[0]/1000:g}–{ranges[1]/1000:g} km: relative temporal phase drift')
        ax.set_ylabel('Phase difference (degrees)')
        ax.legend(ncol=5, fontsize=7, loc='upper left')
        ax.set_ylim(-5, 5)
    axes[-1].set_xlabel('Recording time (s)')
    fig.suptitle('Stationary returns: subtracting one constant per channel pair, no drift correction\n'
                 'Separate fixed range templates in each band; this is not absolute cross-band calibration')
    fig.savefig(output/'stationary-pair-drift.png', dpi=160)
    plt.close(fig)
    return result


def patch(rows, geometry, first, duration, rf):
    last = first+duration
    use = (rows['time'] >= first)&(rows['time'] < last)&(rows['information'] > 0)
    selected = rows[use]
    t = selected['time']
    midpoint = (first+last)/2
    centre = np.interp(midpoint, geometry['time'], geometry['path'])
    range_index = int(np.rint(centre*8e6/C))+np.arange(-8, 9)
    covariance_psf = np.average(selected['psf'], axis=0, weights=selected['information'])
    lag = range_index[:, None]-range_index[None, :]
    covariance = covariance_psf[lag+64]
    covariance = (covariance+covariance.conj().T)*.5
    whiten = solve_triangular(np.linalg.cholesky(covariance), np.eye(len(range_index)), lower=True)
    y = selected['values'][:, range_index]@whiten.T
    path = np.interp(t, geometry['time'], geometry['path'])-centre
    result = Patch(t, y, 8192*selected['information'], path, 2*np.pi*rf/C,
                   midpoint, t < midpoint)
    if min(np.count_nonzero(result.training), np.count_nonzero(~result.training)) < 8:
        raise ValueError('Too few observations for phase prediction')
    return result


def summary(result):
    return {k: v for k, v in result.items() if k not in
            ('time_s', 'training', 'phase_residual_deg', 'projected_amplitude')}


def archived_patch(rows, geometry, first, duration, rf):
    t = rows['time']
    use = (t >= first)&(t < first+duration)&(rows['information'] > 0)
    t = t[use]
    midpoint = first+duration/2
    path = np.interp(t, geometry['time'], geometry['path'])
    centre = np.interp(midpoint, geometry['time'], geometry['path'])
    # A constant phase rotation makes the local coefficients well scaled;
    # it changes no relative phase or Doppler. The archived AIS deramp remains.
    rotate = np.exp(-1j*(2*np.pi*rf/C)*centre)
    result = Patch(t, rows['value'][use]*rotate, 8192*rows['information'][use],
                   np.zeros(len(t)), 2*np.pi*rf/C, midpoint, t < midpoint,
                   np.exp(1j*(2*np.pi*rf/C)*(path-centre)))
    return result


def scan(rows, geometry, frequencies, duration, output, make_patch):
    windows = []
    for index, first in enumerate(np.arange(6, 600-duration, duration)):
        patches = [make_patch(rows[c], geometry[c], first, duration, frequencies[c]) for c in range(5)]
        common = fit_rate(patches)
        shared = [p.evaluate(common) for p in patches]
        independent = [p.evaluate(fit_rate([p])) for p in patches]
        windows.append({'first_s': float(first), 'last_s': float(first+duration),
                        'brightness': [p.brightness() for p in patches],
                        'common_rate_mps': common,
                        'common': [summary(e) for e in shared],
                        'independent': [summary(e) for e in independent]})
        if index % 40 == 0:
            print(json.dumps({'duration_s': duration, 'completed_windows': index+1}), flush=True)
    # Require high signal variation in every band. Selection uses only training
    # variance; neither phase-fit quality nor any held-out measurement is used.
    ranking = np.argsort([min(w['brightness']) for w in windows])[::-1]
    selected = []
    for index in ranking:
        w = windows[index]
        if all(abs(w['first_s']-x['first_s']) >= 12 for x in selected):
            selected.append(dict(w))
        if len(selected) == 6:
            break
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout='constrained')
    for w, ax in zip(selected, axes.flat):
        patches = [make_patch(rows[c], geometry[c], w['first_s'], duration, frequencies[c]) for c in range(5)]
        w['phase_curves'] = [p.evaluate(w['common_rate_mps']) for p in patches]
        for c, curve in enumerate(w['phase_curves']):
            ax.plot(curve['time_s'], curve['phase_residual_deg'], color=COLOURS[c], lw=1,
                    label=NAMES[c]+f" ({curve['test_phase_rms_deg']:.1f}°)")
        middle = (w['first_s']+w['last_s'])/2
        ax.axvspan(w['first_s'], middle, color='#dddddd', alpha=.5)
        ax.axhline(0, color='#555555', lw=.6)
        ax.set_ylim(-180, 180)
        separation = min(e['tone_static_separation'] for e in w['common'])
        ax.set_title(f"{w['first_s']:g}–{w['last_s']:g} s · residual path rate {w['common_rate_mps']:+.3f} m/s\n"
                     f"min. static/moving separation = {separation:.3g}"+(' (nearly degenerate)' if separation < .01 else ''))
        ax.set_xlabel('Recording time (s)')
        ax.set_ylabel('Phase residual (degrees)')
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle(f'Bright {duration:g} s ship intervals: phase predicted into the second half\n'
                 'Grey: fit/brightness selection. White: frozen prediction. Legend: holdout phase RMS.')
    fig.savefig(output/f'ship-{duration:g}s-phase-transfer.png', dpi=160)
    plt.close(fig)
    return {'duration_s': duration, 'windows': windows, 'brightest_separated_windows': selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--stationary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observation-source', choices=['iq-replay', 'archived-ship'], default='iq-replay')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    header, records = read_profiles(args.profiles)
    if header['stationary_subtraction']:
        raise ValueError('This control requires the unnotched IQ replay')
    rows = []
    geometry, fields = [], []
    for c in range(5):
        path = args.fields/f'channel-{c}.npz'
        with np.load(path) as d:
            geometry.append({'time': d['row_time'], 'path': d['centre_path_m']})
            if args.observation_source == 'archived-ship':
                lag = d['offset_samples'][:, None]-d['offset_samples'][None, :]
                covariance = d['pulse_samples'][lag+64]
                covariance = (covariance+covariance.conj().T)*.5
                whiten = solve_triangular(np.linalg.cholesky(covariance), np.eye(len(lag)), lower=True)
                rows.append({'time': d['time'], 'information': d['information'],
                             'value': d['value']@whiten.T})
            else:
                rows.append(records[records['channel'] == c])
        fields.append({'path': str(path), 'sha256': digest(path)})
    frequencies = [c['rf_hz'] for c in header['applied_channels']]
    result = {'scope': __doc__, 'observation_source': args.observation_source, 'profiles': str(args.profiles),
              'profiles_sha256': digest(args.profiles), 'geometry_sources': fields,
              'operator': 'Information-weighted complex least squares with conditional PSF range covariance; 17 range cells about the AIS path; separate stationary and rotating patterns per band. One common scalar path-rate residual within +/-3 m/s, first-half training, second-half prediction. No fitted antenna offsets or autofocus corrections. IQ replay uses fixed range cells with stationary clutter retained; archived-ship uses saved profiles with the capture-time frozen stationary template subtracted, and the already-declared AIS range alignment/phase deramp.',
              'limitations': ['Local constant-pattern diagnostic, not a full range-migration-aware image former.',
                              'A bright range patch is not proof of one physical scatterer.',
                              'Conditional noise weights; no Gaussian significance claims.',
                              'Separate complex band patterns absorb unknown constant phases; this does not estimate a transferable absolute interband calibration.',
                              'Near-zero-Doppler moving and static components can be poorly separated; separation is reported, not notched.',
                              'IQ-replayed C0 profiles average 30 symbols (32.64 ms); usable for this low-Doppler diagnostic, not general moving-target processing.'],
              'stationary': stationary(args.stationary, args.output), 'ship': [],
              'script_sha256': {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('short_phase.py')]}}
    for duration in (1., 3.):
        make_patch = archived_patch if args.observation_source == 'archived-ship' else patch
        result['ship'].append(scan(rows, geometry, frequencies, duration, args.output, make_patch))
        (args.output/'phase-diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
