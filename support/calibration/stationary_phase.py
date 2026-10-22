"""Audit temporal phase stability using retained, stationary complex returns.

This is a diagnostic only. No fitted phase, gain or normalization is applied
to production samples or to the moving-target calibration likelihood.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from geographic_io import read_profiles
from geometry import C


def compare(profiles, reference):
    cross = profiles@reference.conj()
    norm = np.vdot(reference, reference).real
    if norm <= 0:
        raise ValueError('Stationary reference has no energy')
    amplitude = cross/norm
    coherence = abs(cross)**2/(norm*np.sum(abs(profiles)**2, axis=1))
    return amplitude, coherence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, records = read_profiles(args.profiles)
    duration = 92*30*8704/8e6
    count = int(records['time'].max()/duration)
    times = (np.arange(count)+.5)*duration
    profiles = np.zeros((5, count, 1024), complex)
    for channel in range(5):
        rows = records[(records['channel'] == channel)&(records['information'] > 0)]
        block = np.floor(rows['time']/duration).astype(int)
        valid = block < count
        rows, block = rows[valid], block[valid]
        norm = np.bincount(block, weights=rows['information'], minlength=count)
        if np.any(norm <= 0):
            raise ValueError('Empty stationary integration interval')
        np.add.at(profiles[channel], block, rows['values']*rows['information'][:, None])
        profiles[channel] /= norm[:, None]
    ranges = np.arange(1024)*C/8e6
    regions = [(600, 5000), (5000, 15000), (15000, 30000)]
    reference_times = (times >= 30)&(times < 90)
    result = {'source': str(args.profiles), 'coherence_seconds': duration,
        'reference_interval_s': [30, 90], 'regions_m': regions, 'channels': [],
        'scope': 'Complex least-squares comparison with a 30–90 s stationary range profile, evaluated in three separate range regions. No correction is applied. Correlation is an empirical pattern-stability diagnostic, not a thermal-noise confidence interval or independent evidence of absolute cross-band phase.'}
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    colours = ['#2373bb', '#228d59', '#c57812']
    fig, axes = plt.subplots(5, 2, figsize=(14, 13), layout='constrained', sharex=True)
    for ch in range(5):
        channel = {'channel': ch, 'name': names[ch], 'regions': []}
        for index, (lo, hi) in enumerate(regions):
            selected = (ranges >= lo)&(ranges < hi)
            data = profiles[ch][:, selected]
            reference = data[reference_times].mean(axis=0)
            amplitude, coherence = compare(data, reference)
            phase = np.unwrap(np.angle(amplitude))
            phase -= np.median(phase[reference_times])
            degrees = np.degrees(phase)
            label = f'{lo/1000:g}–{hi/1000:g} km'
            axes[ch, 0].plot(times, degrees, color=colours[index], label=label, lw=.8)
            axes[ch, 1].plot(times, coherence, color=colours[index], label=label, lw=.8)
            channel['regions'].append({'range_m': [lo, hi], 'phase_degrees': degrees.tolist(),
                'pattern_squared_coherence': coherence.tolist(),
                'median_squared_coherence': float(np.median(coherence)),
                'phase_p05_p50_p95_degrees': np.percentile(degrees, [5, 50, 95]).tolist()})
        result['channels'].append(channel)
        axes[ch, 0].set_ylabel(names[ch]+' phase (degrees)')
        axes[ch, 1].set_ylabel('Pattern coherence²')
        axes[ch, 1].set_ylim(0, 1.02)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel('Recording time (s)')
    axes[-1, 1].set_xlabel('Recording time (s)')
    fig.suptitle('Stationary-return phase stability · three independent range regions\nDiagnostic of the recorded complex profiles; no phase correction applied')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    result['times_s'] = times.tolist()
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    for ch in result['channels']:
        print(ch['name'], [(r['range_m'], round(r['median_squared_coherence'], 3),
                           np.round(r['phase_p05_p50_p95_degrees'], 1).tolist()) for r in ch['regions']], flush=True)


if __name__ == '__main__':
    main()
