"""Compare ordinary and stationary-nuisance-profiled ship matched images.

This uses the already checked cached field operator and conditional motion /
channel gains. It is a likelihood diagnostic, not a newly calibrated image.
No empirical image normalization or fixed-width Doppler notch is used.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint
from stationary import Stationary, matched_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_path = args.dictionary/'dictionary.json'
    source = json.loads(source_path.read_text())
    if not source['complete']:
        raise ValueError('Finish the exact field dictionary first')
    points = np.array(source['points_body_m'])
    numerators, information = [], []
    for entry in source['channels']:
        c = entry['channel']
        with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
            data, background = loaded['data'].ravel(), loaded['background']
        nuisance = Stationary(background, np.ones(len(data), bool))
        target = np.load(args.dictionary/f'channel-{c}-design.npy', mmap_mode='r')
        n, w = np.empty((2, len(points)), complex), np.empty((2, len(points)))
        for first in range(0, len(points), 128):
            section = slice(first, min(first+128, len(points)))
            design = target[:, section]
            n[0, section], w[0, section] = design.conj().T@data, np.sum(abs(design)**2, axis=0)
            n[1, section], w[1, section] = matched_projection(design, data, nuisance)
        numerators.append(n); information.append(w)
        print(json.dumps({'channel': c, 'stationary_rank': nuisance.rank,
            'minimum_information_fraction': float(np.min(w[1]/w[0]))}), flush=True)
    n, w = np.array(numerators), np.array(information)
    n, w = np.concatenate([n, n.sum(axis=0, keepdims=True)]), np.concatenate([w, w.sum(axis=0, keepdims=True)])
    if np.any(w <= 0) or not np.all(np.isfinite(n)):
        raise ValueError('A pixel is unidentifiable or contains invalid matched statistics')
    power = abs(n)**2/w
    x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
    image = np.full((6, 2, len(y), len(x)), np.nan)
    image[:, :, np.searchsorted(y, points[:, 1]), np.searchsorted(x, points[:, 0])] = 10*np.log10(np.maximum(power, 1e-30))
    upper, lower = float(np.nanmax(image)), float(np.nanmax(image)-45.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.npz'), points_body_m=points, numerator=n, information=w,
        power=power, source_dictionary=str(source_path))
    fig, axes = plt.subplots(6, 2, figsize=(14, 19), sharex=True, sharey=True, layout='constrained')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Conditional coherent sum']
    for c, name in enumerate(names):
        for column, label in enumerate(('Ordinary matched response', 'Stationary nuisance profiled')):
            ax = axes[c, column]
            im = ax.imshow(image[c, column], origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                aspect='equal', cmap='viridis', vmin=lower, vmax=upper)
            ax.set(title=f'{name} / {label}', xlabel='Forward of AIS antenna prior (m)', ylabel='Port (m)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Conditional matched power (dB)', shrink=.7)
    fig.suptitle('Same 30 s field, motion, channel gains and fixed colour scale\n'
        'Right: data and templates both projected; template information recomputed. Unsampled regions are blank.')
    fig.savefig(args.output, dpi=150); plt.close(fig)
    result = {'scope': __doc__, 'source_dictionary': str(source_path), 'source_sha256': fingerprint(source_path),
        'interval_s': source['interval_s'], 'colour_scale_db': [lower, upper], 'calibration_accepted': False,
        'operator': 'P = I - QQ^H for whitened stationary nuisance; z = a^H P y, I = a^H P a, displayed power = |z|^2/I. Independent stationary coefficients per band; coherent sum combines z and I before magnitude.',
        'minimum_retained_information_fraction_by_channel': np.min(w[:5, 1]/w[:5, 0], axis=1).tolist(),
        'maximum_power_db_by_channel_and_variant': np.max(10*np.log10(power), axis=2).tolist()}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
