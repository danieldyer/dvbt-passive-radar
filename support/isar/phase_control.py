"""Uncorrected stationary-return phase control over the ship aperture."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from geographic_io import read_profiles
from stationary_phase import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, records = read_profiles(args.profiles)
    regions = [(600, 5000), (5000, 10000)]
    ranges = np.arange(1024)*299792458/8e6
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    fig, axes = plt.subplots(5, 2, figsize=(13, 11), layout='constrained', sharex=True)
    result = {'source': str(args.profiles), 'reference_s': [30, 90], 'evaluation_s': [90, 210],
              'profile_mean_rows': 30, 'profile_interval_s': 30*8704/8e6,
              'scope': 'Uncorrected stationary control, using a disjoint reference interval. Least-squares phase relative to a reference profile, not absolute transmitter calibration or a statistical confidence bound.',
              'channels': []}
    for channel, name in enumerate(names):
        good = (records['channel'] == channel)&(records['information'] > 0)
        reference_rows = records[good&(records['time'] >= 30)&(records['time'] < 90)]
        rows = records[good&(records['time'] >= 90)&(records['time'] < 210)]
        if len(reference_rows) < 100 or len(rows) < 100:
            raise ValueError('Insufficient independent control coverage')
        entry = {'channel': channel, 'name': name, 'regions': []}
        for lo, hi in regions:
            keep = (ranges >= lo)&(ranges < hi)
            reference = np.average(reference_rows['values'][:, keep], axis=0,
                                   weights=reference_rows['information'])
            gain, coherence = compare(rows['values'][:, keep], reference)
            phase = np.degrees(np.unwrap(np.angle(gain)))
            label = f'{lo/1000:g}–{hi/1000:g} km'
            axes[channel, 0].plot(rows['time'], phase, lw=.5, alpha=.7, label=label)
            axes[channel, 1].plot(rows['time'], coherence, lw=.5, alpha=.7)
            entry['regions'].append({'range_m': [lo, hi], 'phase_p01_p50_p99_deg': np.percentile(phase, [1, 50, 99]).tolist(),
                                     'adjacent_phase_step_absolute_p99_deg': float(np.percentile(abs(np.diff(phase)), 99)),
                                     'pattern_coherence_squared_p01_p50_p99': np.percentile(coherence, [1, 50, 99]).tolist()})
        axes[channel, 0].set_ylabel(name+' phase (°)')
        axes[channel, 1].set_ylabel('Pattern coherence²')
        result['channels'].append(entry)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel('Recording time (s)')
    axes[-1, 1].set_xlabel('Recording time (s)')
    fig.suptitle('Stationary phase control at 32.64 ms cadence · no correction applied\nReference profile from 30–90 s; ship aperture evaluated independently')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
